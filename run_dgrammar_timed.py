"""Run dGrammar v1 with per-operation timing instrumentation.

Same timing breakdown as run_igcd_timed.py for fair comparison:
  - model_forward_ms: time per model(x).logits call
  - grammar_check_ms: time per CFG ∩ DFA emptiness check
  - token_select_ms: time per argmax + confidence computation
  - autocompletion_ms: time for autocompletion post-processing
"""

import json
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import frozendict

from constrained_diffusion.constrain_utils import (
    compile_lex_map, preprocessed_generate_stuff,
    EOS, autocomplete_valid, partial_output_from_tokens,
    generated_language, derive_supertokens,
)
from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from rustformlang.cfg import CFG, is_intersection_empty_threaded

from dgrammar.generate_dgrammar import (
    add_gumbel_noise, get_num_transfer_tokens, check_valid, check_local_valid,
)


class TimingStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []
        self.grammar_check_times = []
        self.token_select_times = []
        self.resample_count = 0
        self.tokens_unmasked = 0
        self.batch_sizes = []

    def summary(self):
        fwd = self.forward_times
        gc = self.grammar_check_times
        ts = self.token_select_times
        return {
            "forward_count": len(fwd),
            "forward_total_ms": sum(fwd) * 1000,
            "forward_mean_ms": (sum(fwd) / len(fwd) * 1000) if fwd else 0,
            "grammar_check_count": len(gc),
            "grammar_check_total_ms": sum(gc) * 1000,
            "grammar_check_mean_ms": (sum(gc) / len(gc) * 1000) if gc else 0,
            "grammar_check_median_ms": (sorted(gc)[len(gc) // 2] * 1000) if gc else 0,
            "grammar_check_p95_ms": (sorted(gc)[int(len(gc) * 0.95)] * 1000) if len(gc) >= 20 else 0,
            "token_select_count": len(ts),
            "token_select_total_ms": sum(ts) * 1000,
            "token_select_mean_ms": (sum(ts) / len(ts) * 1000) if ts else 0,
            "resample_count": self.resample_count,
            "tokens_unmasked": self.tokens_unmasked,
            "avg_batch_size": (sum(self.batch_sizes) / len(self.batch_sizes)) if self.batch_sizes else 0,
        }


STATS = TimingStats()


@torch.no_grad()
def generate_dgrammar_timed(
    model, prompt, tokenizer,
    constraint_lang, lex_map, prompt_len,
    steps=128, gen_length=256, block_length=32,
    temperature=0.0, remasking="low_confidence",
    mask_id=126336, prelex=None,
    subtokens=frozendict.frozendict(),
    strip_chars=None, additional_stuff=None,
    constrain=True, max_batch_size=8,
    max_remask_attempts=3, max_resamples=100,
):
    """dGrammar generate with per-operation timing."""
    start_time = time.monotonic()

    if additional_stuff is None and constrain:
        additional_stuff = preprocessed_generate_stuff(
            tokenizer, constraint_lang, lex_map,
            prelex=prelex, subtokens=subtokens, strip_chars=strip_chars,
        )
    elif additional_stuff is None:
        additional_stuff = None, None, {}

    _, _, supertokens_map = additional_stuff
    resamples = []

    if constrain:
        terminals = constraint_lang.get_terminals()
    else:
        terminals = None

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

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

            # --- Model forward (timed) ---
            t_fwd = time.perf_counter()
            logits = model(x).logits
            STATS.forward_times.append(time.perf_counter() - t_fwd)

            logits_for_confidence = logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            n_scheduled = num_transfer_tokens[0, i].item()
            if n_scheduled == 0:
                continue

            tokens_placed_this_step = 0
            while tokens_placed_this_step < n_scheduled:
                if complete:
                    break

                # --- Token selection (timed) ---
                t_sel = time.perf_counter()
                mask_index = x == mask_id
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits_for_confidence.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                x0_p[:, block_end:] = -np.inf
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
                STATS.token_select_times.append(time.perf_counter() - t_sel)
                STATS.batch_sizes.append(batch_k)

                if select_indices.shape[0] == 0:
                    yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                    return

                # Place tokens
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
                STATS.tokens_unmasked += len(positions)

                # --- Grammar check (timed) ---
                if constrain:
                    t_gc = time.perf_counter()
                    total_grammar_checks += 1
                    is_empty = check_valid(
                        generated_words[prompt_len:],
                        constraint_lang, lex_map, terminals,
                        prelex=prelex, subtokens=subtokens,
                        supertokens=supertokens_map, strip_chars=strip_chars,
                    )
                    STATS.grammar_check_times.append(time.perf_counter() - t_gc)

                    if is_empty:
                        total_violations += 1

                        if len(positions) == 1:
                            pos = positions[0]
                            logits_with_noise[0, pos, x[0, pos]] = -np.inf
                            x[0, pos] = mask_id
                            generated_words[pos] = None
                            total_remasks += 1
                            STATS.resample_count += 1
                            tokens_placed_this_step -= 1
                            STATS.tokens_unmasked -= 1
                            resamples.append((pos, time.monotonic() - start_time))

                            if len(resamples) >= max_resamples:
                                yield x, resamples, False, total_violations, total_remasks, total_grammar_checks
                                return

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

                                t_gc2 = time.perf_counter()
                                total_grammar_checks += 1
                                local_empty = check_local_valid(
                                    generated_words[prompt_len:], rel_pos,
                                    constraint_lang, lex_map, terminals,
                                    prelex=prelex, subtokens=subtokens,
                                    supertokens=supertokens_map, strip_chars=strip_chars,
                                )
                                STATS.grammar_check_times.append(time.perf_counter() - t_gc2)

                                if local_empty:
                                    logits_with_noise[0, pos, next_vocab_idx] = -np.inf
                                    x[0, pos] = mask_id
                                    generated_words[pos] = None
                                    total_remasks += 1
                                    STATS.resample_count += 1
                                    resamples.append((pos, time.monotonic() - start_time))
                                    continue

                                t_gc3 = time.perf_counter()
                                total_grammar_checks += 1
                                still_empty = check_valid(
                                    generated_words[prompt_len:],
                                    constraint_lang, lex_map, terminals,
                                    prelex=prelex, subtokens=subtokens,
                                    supertokens=supertokens_map, strip_chars=strip_chars,
                                )
                                STATS.grammar_check_times.append(time.perf_counter() - t_gc3)

                                if not still_empty:
                                    tokens_placed_this_step += 1
                                    STATS.tokens_unmasked += 1
                                    break
                                logits_with_noise[0, pos, next_vocab_idx] = -np.inf
                                x[0, pos] = mask_id
                                generated_words[pos] = None
                                total_remasks += 1
                                STATS.resample_count += 1
                                resamples.append((pos, time.monotonic() - start_time))

                            current_batch = 1
                        else:
                            for pos in positions:
                                t_gc4 = time.perf_counter()
                                saved = generated_words[pos]
                                generated_words[pos] = None
                                total_grammar_checks += 1
                                still_empty = check_valid(
                                    generated_words[prompt_len:],
                                    constraint_lang, lex_map, terminals,
                                    prelex=prelex, subtokens=subtokens,
                                    supertokens=supertokens_map, strip_chars=strip_chars,
                                )
                                STATS.grammar_check_times.append(time.perf_counter() - t_gc4)
                                generated_words[pos] = saved
                                if not still_empty:
                                    logits_with_noise[0, pos, x[0, pos]] = -np.inf
                                    x[0, pos] = mask_id
                                    generated_words[pos] = None
                                    total_remasks += 1
                                    STATS.resample_count += 1
                                    tokens_placed_this_step -= 1
                                    STATS.tokens_unmasked -= 1
                                    resamples.append((pos, time.monotonic() - start_time))
                            current_batch = 1
                    else:
                        current_batch = min(current_batch * 2, max_batch_size)

                # Check EOS
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
        and None not in generated_words[:generated_words.index(EOS)]
    )
    yield x, resamples, is_complete, total_violations, total_remasks, total_grammar_checks


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    tag = "dgrammar_timed"
    ds_safe = dataset_name.replace("/", "_")
    suffix = f"_off{offset}" if offset > 0 else ""
    output_file = f"results/{tag}_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl"

    dataset = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances = all_instances[offset:offset + limit]
    print(f"dGrammar timed: {len(instances)} instances, seed={seed}, T={steps}")

    lang, lex_map, subtokens = None, None, None
    prelex = None
    additional_stuff = None
    orig_lex_map = None

    for i, instance in enumerate(instances):
        if lang is None or dataset.different_grammar_per_instance:
            lang, lex_map, subtokens = instance.language_lex_subtokens()
            orig_lex_map = lex_map.copy()
            lang = lang.concatenate(CFG.from_text("S -> lexFence | $", "S"))
            if instance.strip_chars() is not None and "\n" not in instance.strip_chars():
                lex_map["lexFence"] = r"\n?```"
            else:
                lex_map["lexFence"] = "```"
            orig_lex_map["lexFence"] = lex_map["lexFence"]
            lang = lang.to_normal_form()
            lex_map = compile_lex_map(lex_map, subtokens=subtokens)
            additional_stuff = None
            prelex = instance.prelex()

        if additional_stuff is None:
            additional_stuff = preprocessed_generate_stuff(
                tokenizer, lang, lex_map,
                prelex=prelex, subtokens=subtokens,
                strip_chars=instance.strip_chars(),
            )

        prompt_ids, prompt_len, suffix_str, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )

        STATS.reset()
        torch.manual_seed(seed)
        start_time = time.monotonic()

        out = None
        resamples = []
        valid = False
        total_violations = 0
        total_remasks = 0
        total_grammar_checks = 0

        for out, resamples, valid, violations, remasks, grammar_checks in generate_dgrammar_timed(
            model, prompt_ids, tokenizer,
            constraint_lang=lang, lex_map=lex_map,
            prompt_len=prompt_len,
            steps=steps, gen_length=256, block_length=32,
            temperature=0.2, remasking="low_confidence",
            prelex=prelex, subtokens=subtokens,
            strip_chars=instance.strip_chars(),
            additional_stuff=additional_stuff,
            constrain=True, max_batch_size=8,
            max_remask_attempts=3, max_resamples=100,
        ):
            total_violations = violations
            total_remasks = remasks
            total_grammar_checks = grammar_checks

        elapsed = time.monotonic() - start_time

        if out is None:
            code = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]

        extracted = instance.extract_result(suffix_str + start_line + code)

        # Autocompletion
        autocompletion = None
        ac_time = 0.0
        supertokens_map = derive_supertokens(subtokens) if subtokens else {}
        if out is not None and not valid:
            ac_start = time.perf_counter()
            mask_id = 126336
            mask_decoded = tokenizer.decode(mask_id)
            eos_tk = tokenizer.special_tokens_map["eos_token"]
            generated_words = tokenizer.batch_decode(out.squeeze())
            generated_words = [
                None if w == mask_decoded
                else EOS if w in (eos_tk, "<|eot_id|>", "<|endoftext|>")
                else w
                for w in generated_words[prompt_len:]
            ]
            partial_output, first_token_gap, last_token_eos_adj = (
                partial_output_from_tokens(generated_words, prelex)
            )
            valid_completion = autocomplete_valid(
                partial_output=partial_output,
                first_token_gap=first_token_gap,
                last_token_eos_adj=last_token_eos_adj,
                generated_lang=generated_language(
                    generated_words,
                    lex_map, lang.get_terminals(),
                    prelex=prelex, subtokens=subtokens,
                    supertokens=supertokens_map,
                    strip_chars=instance.strip_chars(),
                ),
                subtokens=subtokens,
                lex_map=orig_lex_map,
                constraint_lang=lang,
            )
            ac_time = (time.perf_counter() - ac_start) * 1000
            if valid_completion is not None:
                autocompletion = instance.extract_result(suffix_str + valid_completion)

        timing = STATS.summary()
        total_constraint_ms = timing["grammar_check_total_ms"] + timing["token_select_total_ms"]
        total_forward_ms = timing["forward_total_ms"]

        result = {
            "instance_id": instance.instance_id(),
            "method": "dgrammar_v1",
            "valid": valid,
            "extracted": extracted,
            "autocompletion": autocompletion,
            "time_taken": elapsed,
            "resamples": len(resamples),
            "timing": {
                **timing,
                "autocompletion_ms": ac_time,
                "total_constraint_ms": total_constraint_ms,
                "total_forward_ms": total_forward_ms,
                "constraint_pct": (total_constraint_ms / (total_constraint_ms + total_forward_ms) * 100)
                    if (total_constraint_ms + total_forward_ms) > 0 else 0,
                "per_token_constraint_ms": (total_constraint_ms / timing["tokens_unmasked"])
                    if timing["tokens_unmasked"] > 0 else 0,
                "per_token_total_ms": (elapsed * 1000 / timing["tokens_unmasked"])
                    if timing["tokens_unmasked"] > 0 else 0,
            },
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        gc_mean = timing["grammar_check_mean_ms"]
        fwd_mean = timing["forward_mean_ms"]
        pct = result["timing"]["constraint_pct"]
        avg_batch = timing["avg_batch_size"]
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, time={elapsed:.1f}s, "
            f"fwd={fwd_mean:.0f}ms(x{timing['forward_count']}), "
            f"gc={gc_mean:.1f}ms(x{timing['grammar_check_count']}), "
            f"constraint={pct:.0f}%, batch={avg_batch:.1f}"
        )


if __name__ == "__main__":
    main()
