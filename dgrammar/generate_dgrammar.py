"""dGrammar generation using ETH's token-level grammar infrastructure.

Key difference from ETH's IG-CD:
- IG-CD: unmask 1 token at a time, reject if invalid, try next best
- dGrammar: adaptive batch unmasking with selective remasking
  - Start with batch=1 (safe, like IG-CD)
  - If valid, double batch size (up to max)
  - If violation detected, selectively remask violators and reset batch to 1

Crucially, ALL scheduled tokens per step are processed (not just batch-many).
The batch size controls how many tokens are placed before a grammar check,
not the total throughput per step.
"""

import json
import time
from typing import List, Optional, Tuple
from collections import defaultdict

import frozendict
import numpy as np
import torch
import torch.nn.functional as F

from constrained_diffusion.constrain_utils import (
    EOS,
    EOSType,
    generated_language,
    preprocessed_generate_stuff,
    all_lexings_mask,
    CompiledLexMap,
    LexMap,
)
from rustformlang.cfg import CFG, is_intersection_empty_threaded


def compute_no_lexing_mask(tokenizer, lex_map, model, strip_chars=None, prelex=None, trace=False):
    """Compute context-independent mask of tokens with no valid grammar lexing."""
    vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)
    all_tokens_decoded = tokenizer.batch_decode(torch.arange(0, vocab_size))
    _, no_lexing_arr = all_lexings_mask(
        all_tokens_decoded, lex_map, trace=trace,
        strip_chars=strip_chars, prelex=prelex,
    )
    # Whitelist EOS/EOT tokens
    eos_token = tokenizer.special_tokens_map.get("eos_token", "")
    for tok_id, tok_obj in tokenizer.added_tokens_decoder.items():
        if tok_obj.content in (
            eos_token, "<|eot_id|>", "<|endoftext|>",
            "<\uff5cend\u2581of\u2581sentence\uff5c>",
        ):
            no_lexing_arr[tok_id] = 0
    mask = torch.tensor(no_lexing_arr > 0.5, dtype=torch.bool, device=model.device)
    # Pad to match model logit dimension
    logit_vocab_size = model.config.vocab_size if hasattr(model, 'config') else vocab_size
    if mask.shape[0] < logit_vocab_size:
        pad = torch.ones(logit_vocab_size - mask.shape[0], dtype=torch.bool, device=model.device)
        mask = torch.cat([mask, pad])
    if trace:
        n = mask.sum().item()
        print(f"Pre-filter: blocking {n}/{logit_vocab_size} tokens ({n/logit_vocab_size*100:.1f}%)")
    return mask


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


def check_local_valid(
    generated_words,
    pos,
    constraint_lang,
    lex_map,
    terminals,
    prelex=None,
    subtokens=frozendict.frozendict(),
    supertokens=frozendict.frozendict(),
    strip_chars=None,
    window=5,
):
    """Check validity using only a local window around position pos.

    Instead of building a DFA for the full sequence, we use a window of
    tokens around the target position. Tokens outside the window are
    replaced with None (gaps), which the DFA treats as wildcards.
    This is much cheaper than full-sequence checking while still
    capturing local grammar constraints.
    """
    start = max(0, pos - window)
    end = min(len(generated_words), pos + window + 1)
    # Build local view: gaps outside window
    local_words = [None] * len(generated_words)
    for i in range(start, end):
        local_words[i] = generated_words[i]
    local_lang = generated_language(
        local_words,
        lex_map,
        terminals,
        trace=False,
        prelex=prelex,
        subtokens=subtokens,
        supertokens=supertokens,
        strip_chars=strip_chars,
    )
    return is_intersection_empty_threaded(
        constraint_lang, local_lang, timeout=100
    )


def check_valid(
    generated_words,
    constraint_lang,
    lex_map,
    terminals,
    trace=False,
    prelex=None,
    subtokens=frozendict.frozendict(),
    supertokens=frozendict.frozendict(),
    strip_chars=None,
):
    generated_lang = generated_language(
        generated_words,
        lex_map,
        terminals,
        trace=trace,
        prelex=prelex,
        subtokens=subtokens,
        supertokens=supertokens,
        strip_chars=strip_chars,
    )
    intersection_empty = is_intersection_empty_threaded(
        constraint_lang, generated_lang, timeout=100
    )
    return intersection_empty


@torch.no_grad()
def generate_dgrammar(
    model,
    prompt,
    tokenizer,
    constraint_lang: CFG,
    lex_map: CompiledLexMap,
    prompt_len: int,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    trace=False,
    prelex=None,
    subtokens=frozendict.frozendict(),
    strip_chars=None,
    additional_stuff=None,
    constrain=True,
    # dGrammar-specific parameters
    max_batch_size=8,
    max_remask_attempts=3,
    max_resamples=100,
    no_lexing_mask=None,
    template_ids=None,
    type_masker=None,
):
    """dGrammar adaptive batch generation with selective remasking.

    Processes ALL scheduled tokens per diffusion step (like ETH's IG-CD).
    The adaptive batch size controls how many tokens are placed before
    running a grammar check, not total tokens per step.

    Adaptive batching strategy:
    - Start with current_batch=1 (same as IG-CD)
    - After each successful grammar check, double batch: 1 -> 2 -> 4 -> 8
    - On violation, selectively remask violating tokens and reset to 1
    """
    start_time = time.monotonic()

    if additional_stuff is None and constrain:
        additional_stuff = preprocessed_generate_stuff(
            tokenizer, constraint_lang, lex_map,
            trace=trace, prelex=prelex,
            subtokens=subtokens, strip_chars=strip_chars,
        )
    elif additional_stuff is None:
        additional_stuff = None, None, {}

    all_possible_lexings, _no_lexing, supertokens_map = additional_stuff
    resamples = []

    if constrain:
        terminals = constraint_lang.get_terminals()
    else:
        terminals = None

    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    # Template presetting: pre-fill structural tokens in the generation area
    if template_ids is not None:
        t = torch.tensor(template_ids[:gen_length], dtype=torch.long, device=model.device)
        x[0, prompt.shape[1]:prompt.shape[1] + len(t)] = t
        if trace:
            n_filled = sum(1 for tid in template_ids[:gen_length] if tid != mask_id)
            print(f"  Template: {n_filled}/{gen_length} positions pre-filled")

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    generated_words = tokenizer.batch_decode(x.squeeze())
    mask_decoded = tokenizer.decode(mask_id)
    generated_words = [w if w != mask_decoded else None for w in generated_words]
    eos_token = tokenizer.special_tokens_map["eos_token"]
    eot_token = "<|eot_id|>"

    total_violations = 0
    total_remasks = 0
    total_grammar_checks = 0

    # Adaptive batch size: start at 1, grow on success, reset on violation
    current_batch = 1

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        complete = False
        for i in range(steps_per_block):
            if complete:
                break

            # One forward pass per diffusion step
            logits = model(x).logits

            # Pre-filter: mask out tokens with no valid grammar lexing
            if no_lexing_mask is not None:
                logits[:, :, no_lexing_mask] = -np.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            # Type-aware masking: block tokens that don't match expected JSON type
            if type_masker is not None:
                n_type_masked = type_masker.apply_masks(
                    logits_with_noise, generated_words, prompt_len,
                    mask_id, block_start, block_end,
                )
                if trace and n_type_masked > 0:
                    print(f"  Type-masked {n_type_masked} positions")

            n_scheduled = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            # Inner loop: process ALL scheduled tokens, in groups of current_batch
            tokens_placed_this_step = 0
            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break

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
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                n_available = mask_index[0].sum().item()
                if n_available == 0:
                    break

                # How many tokens to place before grammar check
                remaining = n_scheduled - tokens_placed_this_step
                batch_k = min(current_batch, remaining, n_available)
                if batch_k == 0:
                    break

                _, select_indices = torch.topk(confidence[0], k=batch_k)

                if select_indices.shape[0] == 0:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                # Place the selected tokens
                positions = []
                for idx in select_indices:
                    pos = idx.item()
                    vocab_idx = x0[0, pos].item()

                    if logits_with_noise[0, pos, vocab_idx] == -np.inf:
                        continue

                    word = tokenizer.decode(vocab_idx)
                    if word in (eos_token, eot_token):
                        word = EOS

                    generated_words[pos] = word
                    x[0, pos] = x0[0, pos]
                    positions.append(pos)

                if not positions:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                tokens_placed_this_step += len(positions)

                # Grammar check after placing tokens
                if constrain:
                    total_grammar_checks += 1
                    is_empty = check_valid(
                        generated_words[prompt_len:],
                        constraint_lang, lex_map, terminals,
                        trace=trace, prelex=prelex,
                        subtokens=subtokens, supertokens=supertokens_map,
                        strip_chars=strip_chars,
                    )

                    if is_empty:
                        total_violations += 1

                        if len(positions) == 1:
                            # Single token rejection (like IG-CD)
                            pos = positions[0]
                            logits_with_noise[0, pos, x[0, pos]] = -np.inf
                            x[0, pos] = mask_id
                            generated_words[pos] = None
                            total_remasks += 1
                            tokens_placed_this_step -= 1
                            resamples.append((pos, time.monotonic() - start_time))

                            if len(resamples) >= max_resamples:
                                yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                                return

                            # Retry next-best tokens at this position
                            # Use local window check for fast rejection,
                            # confirm with full check on local pass
                            rel_pos = pos - prompt_len
                            while len(resamples) < max_resamples:
                                next_vocab_idx = torch.argmax(logits_with_noise[0, pos]).item()
                                if logits_with_noise[0, pos, next_vocab_idx] == -np.inf:
                                    break

                                word = tokenizer.decode(next_vocab_idx)
                                if word in (eos_token, eot_token):
                                    word = EOS

                                generated_words[pos] = word
                                x[0, pos] = next_vocab_idx

                                # Fast local check first
                                total_grammar_checks += 1
                                local_empty = check_local_valid(
                                    generated_words[prompt_len:],
                                    rel_pos,
                                    constraint_lang, lex_map, terminals,
                                    prelex=prelex,
                                    subtokens=subtokens, supertokens=supertokens_map,
                                    strip_chars=strip_chars,
                                )
                                if local_empty:
                                    # Locally invalid, reject immediately
                                    logits_with_noise[0, pos, next_vocab_idx] = -np.inf
                                    x[0, pos] = mask_id
                                    generated_words[pos] = None
                                    total_remasks += 1
                                    resamples.append((pos, time.monotonic() - start_time))
                                    continue

                                # Local check passed, confirm with full check
                                total_grammar_checks += 1
                                still_empty = check_valid(
                                    generated_words[prompt_len:],
                                    constraint_lang, lex_map, terminals,
                                    trace=False, prelex=prelex,
                                    subtokens=subtokens, supertokens=supertokens_map,
                                    strip_chars=strip_chars,
                                )
                                if not still_empty:
                                    tokens_placed_this_step += 1
                                    break
                                logits_with_noise[0, pos, next_vocab_idx] = -np.inf
                                x[0, pos] = mask_id
                                generated_words[pos] = None
                                total_remasks += 1
                                resamples.append((pos, time.monotonic() - start_time))

                            # Reset batch to 1 after violation
                            current_batch = 1
                        else:
                            # Multiple tokens: find which ones cause the violation
                            violators = []
                            for pos in positions:
                                saved = generated_words[pos]
                                generated_words[pos] = None
                                total_grammar_checks += 1
                                still_empty = check_valid(
                                    generated_words[prompt_len:],
                                    constraint_lang, lex_map, terminals,
                                    trace=False, prelex=prelex,
                                    subtokens=subtokens, supertokens=supertokens_map,
                                    strip_chars=strip_chars,
                                )
                                generated_words[pos] = saved
                                if not still_empty:
                                    violators.append(pos)

                            if not violators:
                                violators = positions

                            for pos in violators:
                                # Mask out this vocab entry so it won't be repicked
                                logits_with_noise[0, pos, x[0, pos]] = -np.inf
                                x[0, pos] = mask_id
                                generated_words[pos] = None
                                total_remasks += 1
                                tokens_placed_this_step -= 1
                                resamples.append((pos, time.monotonic() - start_time))

                            # Reset batch to 1 after violation
                            current_batch = 1
                    else:
                        # Success! Grow batch size
                        current_batch = min(current_batch * 2, max_batch_size)

                # Check for EOS completion
                if EOS in generated_words:
                    eos_idx = generated_words.index(EOS)
                    if None not in generated_words[:eos_idx]:
                        for pos in range(eos_idx, len(generated_words)):
                            if generated_words[pos] is None:
                                generated_words[pos] = EOS
                                x[0, pos] = x0[0, eos_idx]
                        complete = True

            yield x, resamples, False, total_violations, total_remasks, total_grammar_checks

    is_complete = (
        EOS in generated_words
        and None not in generated_words[: generated_words.index(EOS)]
    )
    yield x, resamples, is_complete, total_violations, total_remasks, total_grammar_checks
