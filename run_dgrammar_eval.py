"""Run dGrammar on the same benchmarks as ETH/LAVE for comparison.

Uses ETH's dataset loading + grammar infrastructure with our selective remasking.
Outputs in the same JSONL format so we can use ETH's checkers.
"""

import json
import time
import sys
from pathlib import Path

import torch

from constrained_diffusion.constrain_utils import compile_lex_map, preprocessed_generate_stuff
from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from rustformlang.cfg import CFG
from dgrammar.generate_dgrammar import generate_dgrammar


def run_benchmark(
    dataset_name: str,
    seed: int = 0,
    limit: int = 1000,
    output_file: str = "results/dgrammar.jsonl",
    steps: int = 32,
    max_tokens: int = 256,
    temperature: float = 0.2,
    max_batch_size: int = 8,
    max_remask_attempts: int = 3,
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

    instances = sorted(dataset, key=lambda x: x.instance_id())[:limit]
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
            prelex = instance.prelex()

        if additional_stuff is None:
            additional_stuff = preprocessed_generate_stuff(
                tokenizer, lang, lex_map,
                trace=trace, prelex=prelex,
                subtokens=subtokens, strip_chars=instance.strip_chars(),
            )

        # Prepare prompt
        prompt_ids, prompt_len, suffix, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=trace)
        )

        start_time = time.monotonic()

        # Run dGrammar generation
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
            "autocompletion_raw": None,
            "autocompletion": None,
            "time_taken_autocompletion": None,
            "temp": temperature,
            "max_tokens": max_tokens,
            "time_taken": elapsed,
            "dgrammar_stats": {
                "violations": total_violations,
                "remasks": total_remasks,
                "grammar_checks": total_grammar_checks,
                "max_batch_size": max_batch_size,
            },
        }

        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"violations={total_violations}, remasks={total_remasks}, "
            f"checks={total_grammar_checks}, time={elapsed:.1f}s"
        )


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    dataset = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"

    ds_safe = dataset.replace("/", "_")
    run_benchmark(
        dataset_name=dataset,
        seed=seed,
        limit=limit,
        output_file=f"results/dgrammar_{ds_safe}_s{seed}.jsonl",
    )


if __name__ == "__main__":
    main()
