"""Run AdaGram v2 (llguidance) on jsonschema benchmark."""

import json
import time
import sys
from pathlib import Path

import torch

from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from dgrammar.checker import TokenChecker
from dgrammar.generate_v2 import generate_v2


def run_benchmark(
    dataset_name: str,
    seed: int = 0,
    limit: int = 1000,
    offset: int = 0,
    output_file: str = "results/adagram_v2.jsonl",
    steps: int = 128,
    max_tokens: int = 256,
    temperature: float = 0.2,
    max_batch_size: int = 8,
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

    instances = sorted(dataset, key=lambda x: x.instance_id())[offset:offset + limit]
    print(f"Running AdaGram v2 on {len(instances)} {dataset_name} instances, seed={seed}")

    cached_checker = None

    for i, instance in enumerate(instances):
        if instance.instance_id() in done:
            continue

        # Build checker from raw JSON schema
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            print(f"  Skipping {instance.instance_id()}: no schema")
            continue

        try:
            if cached_checker is None or dataset.different_grammar_per_instance:
                cached_checker = TokenChecker(schema_str)
            checker = cached_checker.clone()
        except (AssertionError, Exception) as e:
            print(f"  Skipping {instance.instance_id()}: {e}")
            continue

        # Prepare prompt
        prompt_ids, prompt_len, suffix, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=trace)
        )

        start_time = time.monotonic()

        out = None
        resamples_list = []
        total_violations = 0
        total_remasks = 0
        total_grammar_checks = 0

        for out, resamples_list, valid, violations, remasks, grammar_checks in generate_v2(
            model, prompt_ids, tokenizer,
            checker=checker,
            prompt_len=prompt_len,
            steps=steps,
            gen_length=max_tokens,
            block_length=32,
            temperature=temperature,
            remasking="low_confidence",
            trace=trace,
            max_batch_size=max_batch_size,
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
            "constrained": "adagram_v2",
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

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
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
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
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
        output_file=f"results/adagram_v2_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl",
    )


if __name__ == "__main__":
    main()
