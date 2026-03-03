"""Run dGrammar with soft constrained decoding on ETH benchmarks.

Instead of hard-masking invalid tokens (-inf), adds a positive bias to
grammar-valid tokens. This preserves the diffusion model's joint distribution
while gently steering toward valid outputs.
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
from dgrammar.generate_dgrammar import generate_dgrammar
from dgrammar.token_dfa import TokenDFATable


def run_benchmark(
    dataset_name: str,
    seed: int = 0,
    limit: int = 10,
    offset: int = 0,
    output_file: str = "results/soft.jsonl",
    steps: int = 128,
    max_tokens: int = 256,
    temperature: float = 0.2,
    max_batch_size: int = 8,
    max_remask_attempts: int = 3,
    soft_bias: float = 5.0,
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

    lang, lex_map, subtokens = None, None, None
    prelex = None
    additional_stuff = None
    token_mask_table = None
    orig_lex_map = None

    instances = sorted(dataset, key=lambda x: x.instance_id())[offset:offset + limit]
    print(f"Running soft-bias={soft_bias} on {len(instances)} {dataset_name} instances, seed={seed}")

    for i, instance in enumerate(instances):
        if instance.instance_id() in done:
            continue

        if lang is None or dataset.different_grammar_per_instance:
            lang, lex_map, subtokens = instance.language_lex_subtokens()
            lang = lang.concatenate(CFG.from_text("S -> lexFence | $", "S"))
            if instance.strip_chars() is not None and "\n" not in instance.strip_chars():
                lex_map["lexFence"] = r"\n?```"
            else:
                lex_map["lexFence"] = "```"
            orig_lex_map = lex_map.copy()
            lang = lang.to_normal_form()
            assert not lang.is_empty(), "Language is empty"

            raw_lex_map = lex_map.copy()
            lex_map = compile_lex_map(lex_map, subtokens=subtokens)
            additional_stuff = None
            prelex = instance.prelex()

            t0 = time.monotonic()
            token_mask_table = TokenDFATable(
                tokenizer, raw_lex_map, lex_map, model.device,
                constraint_lang=lang,
                prelex=prelex,
                strip_chars=instance.strip_chars(),
                trace=trace,
            )
            if trace:
                print(f"  TokenDFATable built in {time.monotonic() - t0:.2f}s")

        if additional_stuff is None:
            additional_stuff = preprocessed_generate_stuff(
                tokenizer, lang, lex_map,
                trace=trace, prelex=prelex,
                subtokens=subtokens, strip_chars=instance.strip_chars(),
            )

        prompt_ids, prompt_len, suffix, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=trace)
        )

        start_time = time.monotonic()

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
            gen_length=max_tokens,
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
            token_mask_table=token_mask_table,
            soft_bias=soft_bias,
        ):
            total_violations = violations
            total_remasks = remasks
            total_grammar_checks = grammar_checks

        elapsed = time.monotonic() - start_time

        if out is None:
            code = "TIMEOUT"
            code_raw = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]
            code_raw = tokenizer.batch_decode(out.squeeze(), skip_special_tokens=False)

        extracted = instance.extract_result(suffix + start_line + code)

        # Autocompletion
        autocompletion_raw = None
        autocompletion = None
        time_taken_autocompletion = None
        supertokens_map = derive_supertokens(subtokens) if subtokens else {}

        if out is not None and not valid:
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

        result = {
            "dataset": dataset_name,
            "instance_id": instance.instance_id(),
            "prompt": prompt_raw,
            "constrained": f"soft_bias_{soft_bias}",
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
                "soft_bias": soft_bias,
            },
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"violations={total_violations}, remasks={total_remasks}, "
            f"checks={total_grammar_checks}, time={elapsed:.1f}s, valid={valid}"
        )


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    dataset = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    soft_bias = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0

    ds_safe = dataset.replace("/", "_")
    bias_tag = f"sb{soft_bias}"
    suffix = f"_off{offset}" if offset > 0 else ""
    run_benchmark(
        dataset_name=dataset,
        seed=seed,
        limit=limit,
        offset=offset,
        steps=steps,
        soft_bias=soft_bias,
        output_file=f"results/soft_{bias_tag}_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl",
    )


if __name__ == "__main__":
    main()
