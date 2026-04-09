"""Run Dgrammar benchmark with per-operation timing.

Thin runner around dgrammar.generate; all algorithm code lives in dgrammar/.
"""

import json
import sys
import time
from pathlib import Path

import torch

from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
import jsb_dataset  # noqa: F401 - registers jsb_* datasets
from dgrammar import TimingStats, TokenChecker, autocomplete_greedy, generate


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 272
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else "jsonschema"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    block_ar = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    max_resamples = int(sys.argv[7]) if len(sys.argv) > 7 else 100
    out_rel = sys.argv[8]  # relative path under results/, set by modal_bench

    output_file = f"results/{out_rel}"

    dataset = load_dataset(dataset_name)
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances = all_instances[offset:offset + limit]
    bl = 32 if block_ar else 256
    print(f"Dgrammar timed: {len(instances)} instances, seed={seed}, T={steps}, block_length={bl}")

    cached_checker = None
    stats = TimingStats()

    for i, instance in enumerate(instances):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            print(f"  Skipping {instance.instance_id()}: no schema")
            continue

        try:
            if cached_checker is None or dataset.different_grammar_per_instance:
                cached_checker = TokenChecker(schema_str)
            checker = cached_checker.clone()
        except Exception as e:
            print(f"  Skipping {instance.instance_id()}: {e}")
            continue

        prompt_ids, prompt_len, suffix_str, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )

        stats.reset()
        torch.manual_seed(seed)
        start_time = time.monotonic()

        out = None
        resamples = []
        valid = False
        total_violations = 0
        total_remasks = 0
        total_grammar_checks = 0

        ac_steps = 0
        ac_mask_ms = 0.0
        ac_fwd_ms = 0.0

        for out, resamples, valid, violations, remasks, grammar_checks in generate(
            model, prompt_ids, checker=checker,
            prompt_len=prompt_len, steps=steps, gen_length=256,
            block_length=bl, temperature=0.2, remasking="low_confidence",
            max_batch_size=8, max_resamples=max_resamples,
            stats=stats,
        ):
            total_violations = violations
            total_remasks = remasks
            total_grammar_checks = grammar_checks

        # Autocompletion fallback: if generation is incomplete, greedily complete
        if out is not None and not valid:
            gen_start = prompt_ids.shape[1]
            mask_id_val = 126336
            eos_id_val = 126081
            eot_id_val = 126348

            # Use a fresh checker to find the grammar frontier: feed the
            # valid prefix (everything before the first mask) and see where
            # llguidance rejects. For phantom cases (no masks but grammar
            # not accepting), this catches the real frontier.
            fresh = cached_checker.clone()
            if prompt_len < gen_start:
                fresh.consume_tokens(out[0, prompt_len:gen_start].tolist())

            gen_tokens = out[0, gen_start:].tolist()
            first_mask = next((j for j, t in enumerate(gen_tokens) if t == mask_id_val), len(gen_tokens))
            prefix_tokens = gen_tokens[:first_mask]
            count = fresh.matcher.try_consume_tokens(prefix_tokens)

            # If grammar consumed everything but still isn't accepting,
            # roll back a short window so AC has room to regenerate.
            if count == len(prefix_tokens) and first_mask == len(gen_tokens) and not fresh.matcher.is_accepting():
                rollback_n = min(32, count)
                fresh.matcher.rollback(rollback_n)
                count -= rollback_n

            frontier = gen_start + count
            # Re-mask from frontier onward so autocomplete_greedy can fill it
            for j in range(frontier, out.shape[1]):
                out[0, j] = mask_id_val

            out, ac_steps, ac_mask_ms, ac_fwd_ms = autocomplete_greedy(
                model, out, fresh, frontier, gen_start,
                mask_id=mask_id_val, eos_id=eos_id_val,
            )
            # Re-check validity after autocompletion
            gen_ids_ac = out[0, gen_start:].tolist()
            if (eos_id_val in gen_ids_ac or eot_id_val in gen_ids_ac) and fresh.is_accepting():
                eos_pos = next((j for j, t in enumerate(gen_ids_ac) if t in (eos_id_val, eot_id_val)), None)
                valid = eos_pos is not None and mask_id_val not in gen_ids_ac[:eos_pos]

        elapsed = time.monotonic() - start_time

        if out is None:
            code = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]

        extracted = instance.extract_result(suffix_str + start_line + code)

        timing = stats.summary()
        timing["autocomplete_steps"] = ac_steps
        timing["autocomplete_mask_ms"] = ac_mask_ms
        timing["autocomplete_fwd_ms"] = ac_fwd_ms
        # For async, effective constraint = gc + token_select + mask_wait + ac_mask
        # mask_compute was hidden behind forward pass
        effective_constraint_ms = (
            timing["grammar_check_total_ms"]
            + timing["token_select_total_ms"]
            + timing["mask_wait_total_ms"]
            + ac_mask_ms
        )
        total_constraint_ms = (
            timing["grammar_check_total_ms"]
            + timing["token_select_total_ms"]
            + timing["mask_compute_total_ms"]
        )
        total_forward_ms = timing["forward_total_ms"]

        result = {
            "instance_id": instance.instance_id(),
            "method": "dgrammar",
            "valid": valid,
            "extracted": extracted,
            "time_taken": elapsed,
            "resamples": len(resamples),
            "timing": {
                **timing,
                "total_constraint_ms": total_constraint_ms,
                "effective_constraint_ms": effective_constraint_ms,
                "total_forward_ms": total_forward_ms,
                "constraint_pct": (total_constraint_ms / (total_constraint_ms + total_forward_ms) * 100)
                    if (total_constraint_ms + total_forward_ms) > 0 else 0,
                "effective_constraint_pct": (effective_constraint_ms / (effective_constraint_ms + total_forward_ms) * 100)
                    if (effective_constraint_ms + total_forward_ms) > 0 else 0,
                "per_token_constraint_ms": (effective_constraint_ms / timing["tokens_unmasked"])
                    if timing["tokens_unmasked"] > 0 else 0,
                "per_token_total_ms": (elapsed * 1000 / timing["tokens_unmasked"])
                    if timing["tokens_unmasked"] > 0 else 0,
                "mask_time_saved_ms": timing["mask_compute_total_ms"] - timing["mask_wait_total_ms"],
            },
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        gc_mean = timing["grammar_check_mean_ms"]
        fwd_mean = timing["forward_mean_ms"]
        eff_pct = result["timing"]["effective_constraint_pct"]
        avg_batch = timing["avg_batch_size"]
        wait_mean = timing["mask_wait_mean_ms"]
        mc_mean = timing["mask_compute_mean_ms"]
        saved = result["timing"]["mask_time_saved_ms"]
        ac_info = f", AC={ac_steps}steps/{ac_mask_ms:.0f}ms" if ac_steps > 0 else ""
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, time={elapsed:.1f}s, "
            f"fwd={fwd_mean:.0f}ms(x{timing['forward_count']}), "
            f"gc={gc_mean:.3f}ms(x{timing['grammar_check_count']}), "
            f"mask={mc_mean:.2f}ms(x{timing['mask_compute_count']}), "
            f"wait={wait_mean:.2f}ms(x{timing['mask_wait_count']}), "
            f"overlap={timing['overlap_count']}, saved={saved:.0f}ms, "
            f"eff_constraint={eff_pct:.1f}%, batch={avg_batch:.1f}{ac_info}"
        )


if __name__ == "__main__":
    main()
