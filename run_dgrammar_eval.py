"""Run dGrammar on the same benchmarks as ETH/LAVE for comparison.

Uses ETH's dataset loading + grammar infrastructure with our selective remasking.
Outputs in the same JSONL format so we can use ETH's checkers.
"""

import json
import time
import sys
from pathlib import Path

import torch

from constrained_diffusion.constrain_utils import (
    compile_lex_map, preprocessed_generate_stuff,
    EOS, autocomplete_valid, partial_output_from_tokens,
    generated_language, derive_supertokens,
)
from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from rustformlang.cfg import CFG
from dgrammar.generate_dgrammar import generate_dgrammar, compute_no_lexing_mask


def run_benchmark(
    dataset_name: str,
    seed: int = 0,
    limit: int = 1000,
    offset: int = 0,
    output_file: str = "results/dgrammar.jsonl",
    steps: int = 32,
    max_tokens: int = 256,
    temperature: float = 0.2,
    max_batch_size: int = 8,
    max_remask_attempts: int = 3,
    max_retries: int = 3,
    trace: bool = False,
    device: str = "cuda",
):
    dataset = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")

    torch.manual_seed(seed)

    done = set()
    if Path(output_file).exists():
        with open(output_file) as f:
            for line in f:
                d = json.loads(line)
                done.add(d["instance_id"])

    tokenizer = eval_model.tokenizer(device)
    model = eval_model.model(device)

    # Load grammar
    lang, lex_map, subtokens = None, None, None
    prelex = None
    additional_stuff = None
    cached_no_lexing_mask = None

    instances = sorted(dataset, key=lambda x: x.instance_id())[offset:offset + limit]
    print(f"Running dGrammar on {len(instances)} {dataset_name} instances, seed={seed}")

    for i, instance in enumerate(instances):
        if instance.instance_id() in done:
            continue

        # Load grammar per instance if needed
        if lang is None or dataset.different_grammar_per_instance:
            lang, lex_map, subtokens = instance.language_lex_subtokens()
            orig_lex_map = lex_map
            lang = lang.concatenate(CFG.from_text("S -> lexFence | $", "S"))
            if instance.strip_chars() is not None and "\n" not in instance.strip_chars():
                lex_map["lexFence"] = r"\n?```"
            else:
                lex_map["lexFence"] = "```"
            lang = lang.to_normal_form()
            assert not lang.is_empty(), "Language is empty"
            lex_map = compile_lex_map(lex_map, subtokens=subtokens)
            additional_stuff = None
            cached_no_lexing_mask = None
            prelex = instance.prelex()

        if additional_stuff is None:
            additional_stuff = preprocessed_generate_stuff(
                tokenizer, lang, lex_map,
                trace=trace, prelex=prelex,
                subtokens=subtokens, strip_chars=instance.strip_chars(),
            )
        # no_lexing_mask disabled: only blocks ~236/126K tokens (0.19%),
        # not worth the 7-46s computation cost per grammar instance
        # if cached_no_lexing_mask is None:
        #     cached_no_lexing_mask = compute_no_lexing_mask(
        #         tokenizer, lex_map, model,
        #         strip_chars=instance.strip_chars(), prelex=prelex,
        #     )

        # Prepare prompt
        prompt_ids, prompt_len, suffix, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=trace)
        )

        start_time = time.monotonic()

        # Retry loop with escalating parameters
        retry_tokens = [max_tokens, 384, 512]
        retry_resamples = [100, 200, 200]
        attempts_used = 0

        out = None
        resamples_list = []
        total_violations = 0
        total_remasks = 0
        total_grammar_checks = 0
        valid = False
        code = "TIMEOUT"
        code_raw = "TIMEOUT"
        extracted = None
        autocompletion_raw = None
        autocompletion = None
        time_taken_autocompletion = None
        supertokens_map = derive_supertokens(subtokens) if subtokens else {}

        for attempt in range(1 + max_retries):
            attempt_tokens = retry_tokens[min(attempt, len(retry_tokens) - 1)]
            attempt_resamples = retry_resamples[min(attempt, len(retry_resamples) - 1)]

            # Use different seed for retries
            if attempt > 0:
                torch.manual_seed(seed + attempt * 1000)
                if trace:
                    print(f"  Retry {attempt}: seed={seed + attempt * 1000}, "
                          f"max_tokens={attempt_tokens}, max_resamples={attempt_resamples}")

            out = None
            resamples_list = []
            total_violations = 0
            total_remasks = 0
            total_grammar_checks = 0

            for out, resamples_list, valid, violations, remasks, grammar_checks in generate_dgrammar(
                model, prompt_ids, tokenizer,
                constraint_lang=lang,
                lex_map=lex_map,
                prompt_len=prompt_len,
                steps=steps,
                gen_length=attempt_tokens,
                block_length=32,
                temperature=temperature,
                remasking="low_confidence",
                trace=trace,
                prelex=prelex,
                subtokens=subtokens,
                strip_chars=instance.strip_chars(),
                additional_stuff=additional_stuff,
                constrain=True,
                max_batch_size=max_batch_size,
                max_remask_attempts=max_remask_attempts,
                max_resamples=attempt_resamples,
                no_lexing_mask=cached_no_lexing_mask,
            ):
                total_violations = violations
                total_remasks = remasks
                total_grammar_checks = grammar_checks

            attempts_used = attempt + 1

            if out is None:
                code = "TIMEOUT"
                code_raw = "TIMEOUT"
            else:
                code = tokenizer.batch_decode(
                    out[:, prompt_ids.shape[1]:], skip_special_tokens=True
                )[0]
                code_raw = tokenizer.batch_decode(out.squeeze(), skip_special_tokens=False)

            extracted = instance.extract_result(suffix + start_line + code)

            # If valid, no need to retry
            if valid:
                break

            # Try autocompletion before retrying
            autocompletion_raw = None
            autocompletion = None
            time_taken_autocompletion = None

            if out is not None:
                ac_start = time.monotonic()
                mask_id = 126336
                mask_decoded = tokenizer.decode(mask_id)
                eos_token = tokenizer.special_tokens_map["eos_token"]
                generated_words = tokenizer.batch_decode(out.squeeze())
                generated_words = [
                    None if x == mask_decoded
                    else EOS if x in (eos_token, "<|eot_id|>", "<|endoftext|>")
                    else x
                    for x in generated_words[prompt_len:]
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
                        trace=trace, prelex=prelex,
                        subtokens=subtokens,
                        supertokens=supertokens_map,
                        strip_chars=instance.strip_chars(),
                    ),
                    subtokens=subtokens,
                    lex_map=orig_lex_map,
                    constraint_lang=lang,
                    trace=trace,
                )
                time_taken_autocompletion = time.monotonic() - ac_start
                if valid_completion is not None:
                    autocompletion_raw = valid_completion
                    autocompletion = instance.extract_result(
                        suffix + valid_completion
                    )
                    break  # Autocompletion succeeded, no need to retry

        # Restore original seed after retries
        if attempts_used > 1:
            torch.manual_seed(seed)

        elapsed = time.monotonic() - start_time

        result = {
            "dataset": dataset_name,
            "instance_id": instance.instance_id(),
            "prompt": prompt_raw,
            "constrained": "dgrammar",
            "model_name": "GSAI-ML/LLaDA-8B-Instruct",
            "code": code,
            "code_raw": code_raw,
            "extracted": extracted,
            "trace": trace,
            "timeout": 600,
            "seed": seed,
            "timed_out": False,
            "resamples": resamples_list,
            "autocompletion_raw": autocompletion_raw,
            "autocompletion": autocompletion,
            "time_taken_autocompletion": time_taken_autocompletion,
            "temp": temperature,
            "max_tokens": max_tokens,
            "time_taken": elapsed,
            "dgrammar_stats": {
                "violations": total_violations,
                "remasks": total_remasks,
                "grammar_checks": total_grammar_checks,
                "max_batch_size": max_batch_size,
                "attempts": attempts_used,
            },
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        retry_info = f", attempts={attempts_used}" if attempts_used > 1 else ""
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"violations={total_violations}, remasks={total_remasks}, "
            f"checks={total_grammar_checks}, time={elapsed:.1f}s{retry_info}"
        )


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    dataset = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    tag = sys.argv[6] if len(sys.argv) > 6 else ""

    ds_safe = dataset.replace("/", "_")
    suffix = f"_off{offset}" if offset > 0 else ""
    if tag:
        suffix += f"_{tag}"
    run_benchmark(
        dataset_name=dataset,
        seed=seed,
        limit=limit,
        offset=offset,
        steps=steps,
        output_file=f"results/dgrammar_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl",
    )


if __name__ == "__main__":
    main()
