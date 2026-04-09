"""Dgrammar: grammar-constrained decoding for diffusion LLMs.

Core algorithm: adaptive batch unmasking + incremental token-level grammar
checking + async overlap of constraint compute with GPU forward.

Public API:
- generate(model, prompt, checker, prompt_len, ...)
    Generator yielding (x, resamples, is_complete, violations, remasks, grammar_checks)
    after each step. The main diffusion loop.
- autocomplete_greedy(model, x, checker, ...) -> (x, ac_steps, ac_mask_ms, ac_fwd_ms)
    Grammar-guided greedy fallback for when the main loop fails to complete.
- TimingStats: optional per-operation timing collector. Pass to generate()
    to profile constraint overhead.
"""

import time
import threading

import numpy as np
import torch
import torch.nn.functional as F

from dgrammar.checker import TokenChecker


class TimingStats:
    """Per-operation timing for constraint overhead profiling."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []
        self.grammar_check_times = []
        self.token_select_times = []
        self.mask_compute_times = []
        self.mask_wait_times = []
        self.resample_count = 0
        self.tokens_unmasked = 0
        self.batch_sizes = []
        self.overlap_count = 0

    def summary(self):
        fwd = self.forward_times
        gc = self.grammar_check_times
        ts = self.token_select_times
        mc = self.mask_compute_times
        mw = self.mask_wait_times
        return {
            "forward_count": len(fwd),
            "forward_total_ms": sum(fwd) * 1000,
            "forward_mean_ms": (sum(fwd) / len(fwd) * 1000) if fwd else 0,
            "grammar_check_count": len(gc),
            "grammar_check_total_ms": sum(gc) * 1000,
            "grammar_check_mean_ms": (sum(gc) / len(gc) * 1000) if gc else 0,
            "grammar_check_median_ms": (sorted(gc)[len(gc) // 2] * 1000) if gc else 0,
            "grammar_check_p95_ms": (sorted(gc)[int(len(gc) * 0.95)] * 1000) if len(gc) >= 20 else 0,
            "mask_compute_count": len(mc),
            "mask_compute_total_ms": sum(mc) * 1000,
            "mask_compute_mean_ms": (sum(mc) / len(mc) * 1000) if mc else 0,
            "mask_wait_count": len(mw),
            "mask_wait_total_ms": sum(mw) * 1000,
            "mask_wait_mean_ms": (sum(mw) / len(mw) * 1000) if mw else 0,
            "overlap_count": self.overlap_count,
            "token_select_count": len(ts),
            "token_select_total_ms": sum(ts) * 1000,
            "token_select_mean_ms": (sum(ts) / len(ts) * 1000) if ts else 0,
            "resample_count": self.resample_count,
            "tokens_unmasked": self.tokens_unmasked,
            "avg_batch_size": (sum(self.batch_sizes) / len(self.batch_sizes)) if self.batch_sizes else 0,
        }


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def compute_mask_async(checker, vocab_size):
    """Run checker.compute_mask in a background thread for GPU/CPU overlap.

    Returns (thread, result_holder). result_holder is [mask, compute_time],
    populated when the thread finishes. Caller must thread.join() before reading.
    """
    result = [None, 0.0]

    def _run():
        t0 = time.perf_counter()
        result[0] = checker.compute_mask(vocab_size=vocab_size)
        result[1] = time.perf_counter() - t0

    thread = threading.Thread(target=_run)
    thread.start()
    return thread, result


def extend_prefix(checker: TokenChecker, x, consume_idx: int, mask_id: int, stats=None):
    """Try to consume contiguous non-mask tokens starting from consume_idx.

    Returns (new_consume_idx, violator_pos_or_neg1).
    - If all contiguous tokens are valid: (advanced_idx, -1)
    - If a token fails: (advanced_past_valid, violator_absolute_pos)
    Checker state is advanced past all valid tokens in both cases.
    """
    t0 = time.perf_counter() if stats is not None else None
    tokens_to_consume = []
    pos = consume_idx
    while pos < x.shape[1]:
        tid = x[0, pos].item()
        if tid == mask_id:
            break
        tokens_to_consume.append(tid)
        pos += 1

    if not tokens_to_consume:
        if stats is not None:
            stats.grammar_check_times.append(time.perf_counter() - t0)
        return consume_idx, -1

    count = checker.matcher.try_consume_tokens(tokens_to_consume)
    if stats is not None:
        stats.grammar_check_times.append(time.perf_counter() - t0)

    if count == len(tokens_to_consume):
        return consume_idx + count, -1
    violator_pos = consume_idx + count
    return consume_idx + count, violator_pos


@torch.no_grad()
def autocomplete_greedy(model, x, checker, consume_idx, gen_start, mask_id, eos_id,
                        refresh_interval=8):
    """Grammar-guided greedy completion from consume_idx forward.

    Diffusion models give logits for all positions at once. We exploit this by
    doing one forward pass and completing multiple positions from the same logits.
    Logits are refreshed every `refresh_interval` steps to avoid staleness.

    Returns (x, autocomplete_steps, autocomplete_mask_ms, autocomplete_fwd_ms).
    """
    ac_steps = 0
    ac_fwd_ms = 0.0
    ac_mask_ms = 0.0
    seq_len = x.shape[1]
    steps_since_refresh = refresh_interval  # force initial forward

    while consume_idx < seq_len:
        if checker.is_accepting():
            for j in range(consume_idx, seq_len):
                x[0, j] = eos_id
            break

        # Refresh logits periodically
        if steps_since_refresh >= refresh_interval:
            t_fwd = time.perf_counter()
            logits = model(x).logits
            ac_fwd_ms += (time.perf_counter() - t_fwd) * 1000
            steps_since_refresh = 0

        # compute_mask at frontier
        t_mc = time.perf_counter()
        bias = checker.compute_mask(vocab_size=logits.shape[-1])
        ac_mask_ms += (time.perf_counter() - t_mc) * 1000

        logits[0, consume_idx, bias] = -np.inf
        best = torch.argmax(logits[0, consume_idx]).item()

        if logits[0, consume_idx, best] == -np.inf:
            break

        x[0, consume_idx] = best

        c = checker.matcher.try_consume_tokens([best])
        if c == 1:
            consume_idx += 1
            ac_steps += 1
            steps_since_refresh += 1
        else:
            break

        # Extend past already-placed non-mask tokens
        while consume_idx < seq_len:
            tid = x[0, consume_idx].item()
            if tid == mask_id:
                break
            c = checker.matcher.try_consume_tokens([tid])
            if c == 1:
                consume_idx += 1
                ac_steps += 1
                steps_since_refresh += 1
            else:
                x[0, consume_idx] = mask_id
                break

    return x, ac_steps, ac_mask_ms, ac_fwd_ms


@torch.no_grad()
def generate(
    model,
    prompt,
    checker: TokenChecker,
    prompt_len: int,
    steps=128,
    gen_length=256,
    block_length=32,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_id=126081,
    eot_id=126348,
    max_batch_size=8,
    max_resamples=100,
    stats=None,
):
    """Dgrammar v2+AC4: async overlap + adaptive batch + frontier masking.

    Main diffusion loop with incremental token-level grammar checking.
    Pass a TimingStats instance to `stats` to collect per-operation profiling.

    Yields (x, resamples, is_complete, total_violations, total_remasks, total_grammar_checks)
    after each block-step. The final yield carries the terminal is_complete flag.
    """
    start_time = time.monotonic()

    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    total_violations = 0
    total_remasks = 0
    total_grammar_checks = 0
    resamples = []
    current_batch = 1

    gen_start = prompt.shape[1]
    consume_idx = gen_start

    if prompt_len < gen_start:
        prefix_tokens = x[0, prompt_len:gen_start].tolist()
        checker.consume_tokens(prefix_tokens)

    # Pending async mask result from previous iteration
    pending_mask = None  # (thread, result_list)

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end = gen_start + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            # Precompute mask async if frontier is a mask: overlaps with forward
            mask_index_pre = x == mask_id
            need_mask = (
                consume_idx < x.shape[1]
                and mask_index_pre[0, consume_idx]
                and pending_mask is None
            )
            if need_mask:
                vocab_size = 126464  # corrected after logits are available
                thread, result_holder = compute_mask_async(checker, vocab_size)
                pending_mask = (thread, result_holder)

            # Model forward
            t_fwd = time.perf_counter() if stats is not None else None
            logits = model(x).logits
            if stats is not None:
                stats.forward_times.append(time.perf_counter() - t_fwd)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            n_scheduled = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            tokens_placed_this_step = 0
            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break

                t_sel = time.perf_counter() if stats is not None else None
                mask_index = x == mask_id
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, block_end:] = -np.inf

                # Frontier masking: use async result if available, else sync fallback
                if consume_idx < x.shape[1] and mask_index[0, consume_idx]:
                    if pending_mask is not None:
                        t_wait = time.perf_counter() if stats is not None else None
                        thread, result_holder = pending_mask
                        thread.join()
                        if stats is not None:
                            stats.mask_wait_times.append(time.perf_counter() - t_wait)
                            stats.mask_compute_times.append(result_holder[1])
                            stats.overlap_count += 1
                        bias = result_holder[0]
                        pending_mask = None
                        # Truncate/pad if vocab sizes differ
                        actual_vocab = logits_with_noise.shape[-1]
                        if bias.shape[0] > actual_vocab:
                            bias = bias[:actual_vocab]
                        elif bias.shape[0] < actual_vocab:
                            pad = torch.ones(actual_vocab - bias.shape[0], dtype=torch.bool)
                            bias = torch.cat([bias, pad])
                    else:
                        # Sync fallback: state changed since last forward (e.g. after resampling)
                        t_mask = time.perf_counter() if stats is not None else None
                        bias = checker.compute_mask(vocab_size=logits_with_noise.shape[-1])
                        if stats is not None:
                            stats.mask_compute_times.append(time.perf_counter() - t_mask)

                    logits_with_noise[0, consume_idx, bias] = -np.inf
                    x0[0, consume_idx] = torch.argmax(logits_with_noise[0, consume_idx])

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                n_available = mask_index[0].sum().item()
                if n_available == 0:
                    break

                remaining = n_scheduled - tokens_placed_this_step
                batch_k = min(current_batch, remaining, n_available)
                if batch_k == 0:
                    break

                _, select_indices = torch.topk(confidence[0], k=batch_k)
                if stats is not None:
                    stats.token_select_times.append(time.perf_counter() - t_sel)
                    stats.batch_sizes.append(batch_k)

                if select_indices.shape[0] == 0:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                positions = []
                for idx in select_indices:
                    pos = idx.item()
                    vocab_idx = x0[0, pos].item()
                    if logits_with_noise[0, pos, vocab_idx] == -np.inf:
                        continue
                    x[0, pos] = x0[0, pos]
                    positions.append(pos)

                if not positions:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                tokens_placed_this_step += len(positions)
                if stats is not None:
                    stats.tokens_unmasked += len(positions)

                # Grammar check: extend prefix, resolve violations
                total_grammar_checks += 1
                new_idx, violator = extend_prefix(checker, x, consume_idx, mask_id, stats=stats)

                if violator < 0:
                    consume_idx = new_idx
                    current_batch = min(current_batch * 2, max_batch_size)
                else:
                    total_violations += 1
                    consume_idx = new_idx

                    # Grammar already satisfied: truncate and finish
                    if checker.is_accepting():
                        for j in range(violator, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True
                        current_batch = 1
                        continue

                    # Remask violator and try next-best candidates from same logits
                    bad_token = x[0, violator].item()
                    logits_with_noise[0, violator, bad_token] = -np.inf
                    x[0, violator] = mask_id
                    total_remasks += 1
                    tokens_placed_this_step -= 1
                    if stats is not None:
                        stats.resample_count += 1
                        stats.tokens_unmasked -= 1
                    resamples.append((violator, time.monotonic() - start_time))

                    if len(resamples) >= max_resamples:
                        yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                        return

                    while len(resamples) < max_resamples:
                        next_vocab = torch.argmax(logits_with_noise[0, violator]).item()
                        if logits_with_noise[0, violator, next_vocab] == -np.inf:
                            break
                        t_gc_retry = time.perf_counter() if stats is not None else None
                        total_grammar_checks += 1
                        c = checker.matcher.try_consume_tokens([next_vocab])
                        if stats is not None:
                            stats.grammar_check_times.append(time.perf_counter() - t_gc_retry)
                        if c == 1:
                            x[0, violator] = next_vocab
                            consume_idx += 1
                            tokens_placed_this_step += 1
                            if stats is not None:
                                stats.tokens_unmasked += 1
                            further_idx, _ = extend_prefix(
                                checker, x, consume_idx, mask_id, stats=stats
                            )
                            total_grammar_checks += 1
                            consume_idx = further_idx
                            break
                        logits_with_noise[0, violator, next_vocab] = -np.inf
                        total_remasks += 1
                        if stats is not None:
                            stats.resample_count += 1
                        resamples.append((violator, time.monotonic() - start_time))
                    current_batch = 1

                # Early grammar-accepting completion
                if not complete and checker.is_accepting():
                    gen_ids = x[0, gen_start:].tolist()
                    first_mask = next((j for j, t in enumerate(gen_ids) if t == mask_id), len(gen_ids))
                    if first_mask >= consume_idx - gen_start:
                        for j in range(consume_idx, x.shape[1]):
                            x[0, j] = eos_id
                        complete = True

                # EOS/EOT termination
                if not complete:
                    gen_ids = x[0, gen_start:].tolist()
                    if (eos_id in gen_ids or eot_id in gen_ids) and checker.is_accepting():
                        eos_pos = next(
                            (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)),
                            None,
                        )
                        if eos_pos is not None and mask_id not in gen_ids[:eos_pos]:
                            for j in range(eos_pos, len(gen_ids)):
                                x[0, gen_start + j] = x[0, gen_start + eos_pos]
                            complete = True

            yield x, resamples, False, total_violations, total_remasks, total_grammar_checks

    # Clean up any pending async thread
    if pending_mask is not None:
        thread, _ = pending_mask
        thread.join()

    gen_ids = x[0, gen_start:].tolist()
    is_complete = False
    if (eos_id in gen_ids or eot_id in gen_ids) and checker.is_accepting():
        eos_pos = next(
            (j for j, t in enumerate(gen_ids) if t in (eos_id, eot_id)), None
        )
        is_complete = eos_pos is not None and mask_id not in gen_ids[:eos_pos]

    yield x, resamples, is_complete, total_violations, total_remasks, total_grammar_checks
