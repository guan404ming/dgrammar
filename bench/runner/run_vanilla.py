"""Run vanilla (unconstrained) LLaDA-8B with timing.

Uses the llada model's generate_unconstrained path (constrain=False). Validity is
checked post-hoc via jsonschema.validate against each instance's JSON schema.
"""

import json
import sys
import time
from pathlib import Path

import jsonschema
import torch

from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
import jsb_dataset  # noqa: F401 - registers jsb_* datasets


def validate_json(extracted: str, schema_str: str) -> tuple[bool, bool]:
    """Return (syntax_ok, schema_ok)."""
    try:
        obj = json.loads(extracted)
    except (json.JSONDecodeError, ValueError, TypeError):
        return False, False
    try:
        schema = json.loads(schema_str)
        jsonschema.validate(instance=obj, schema=schema)
        return True, True
    except jsonschema.ValidationError:
        return True, False
    except Exception:
        return True, False


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    out_rel = sys.argv[6]  # relative path under results/, set by modal_bench

    output_file = f"results/{out_rel}"

    dataset = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances = all_instances[offset:offset + limit]
    print(f"Vanilla LLaDA timed: {len(instances)} instances, seed={seed}, T={steps}")

    for i, instance in enumerate(instances):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            print(f"  Skipping {instance.instance_id()}: no schema")
            continue

        torch.manual_seed(seed)
        start_time = time.monotonic()

        try:
            _, code, extracted, _ = eval_model.generate_unconstrained(
                instance, model, tokenizer,
                steps=steps, gen_length=256, temperature=0.2, trace=False,
            )
        except Exception as e:
            print(f"  Error on {instance.instance_id()}: {e}")
            continue

        elapsed = time.monotonic() - start_time

        syntax_ok, schema_ok = validate_json(extracted, schema_str)
        valid = schema_ok

        result = {
            "instance_id": instance.instance_id(),
            "method": "vanilla_llada",
            "valid": valid,
            "syntax_ok": syntax_ok,
            "extracted": extracted,
            "time_taken": elapsed,
            "resamples": 0,
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, syntax={syntax_ok}, time={elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
