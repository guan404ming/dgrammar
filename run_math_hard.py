"""Mathematical approaches to the 11 hard instances.

Approach 1: Rejection sampling - generate K samples, pick first valid.
  This samples from p(x | grammar_valid(x)).

Approach 2: KL projection - at each step, renormalize logits to valid tokens
  for token selection, but keep original logits for confidence ranking.
  This is the proper KL-closest distribution to p_model with support on C.
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

HARD_IDS = [
    "jsonschema_27", "jsonschema_28", "jsonschema_33", "jsonschema_35",
    "jsonschema_119", "jsonschema_127", "jsonschema_148", "jsonschema_149",
    "jsonschema_160", "jsonschema_256", "jsonschema_261",
]


def run_rejection_sampling(k_samples: int):
    """Generate K samples per instance, pick first valid."""
    tag = f"reject_k{k_samples}"
    output_file = f"results/math_{tag}.jsonl"

    dataset = load_dataset("jsonschema")
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    hard_set = set(HARD_IDS)
    instances = [inst for inst in all_instances if inst.instance_id() in hard_set]
    print(f"Rejection sampling K={k_samples} on {len(instances)} hard instances")

    for i, instance in enumerate(instances):
        lang, lex_map, subtokens = instance.language_lex_subtokens()
        lang = lang.concatenate(CFG.from_text("S -> lexFence | $", "S"))
        if instance.strip_chars() is not None and "\n" not in instance.strip_chars():
            lex_map["lexFence"] = r"\n?```"
        else:
            lex_map["lexFence"] = "```"
        orig_lex_map = lex_map.copy()
        lang = lang.to_normal_form()
        lex_map = compile_lex_map(lex_map, subtokens=subtokens)
        prelex = instance.prelex()
        additional_stuff = preprocessed_generate_stuff(
            tokenizer, lang, lex_map,
            prelex=prelex, subtokens=subtokens,
            strip_chars=instance.strip_chars(),
        )

        prompt_ids, prompt_len, suffix, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )

        start_time = time.monotonic()
        best_out = None
        best_valid = False
        best_violations = 999
        best_resamples = []
        best_code = "TIMEOUT"
        best_sample_idx = -1

        for k in range(k_samples):
            torch.manual_seed(k * 1000)

            out = None
            for out, resamples_list, valid, violations, remasks, grammar_checks in generate_dgrammar(
                model, prompt_ids, tokenizer,
                constraint_lang=lang,
                lex_map=lex_map,
                prompt_len=prompt_len,
                steps=128,
                gen_length=256,
                block_length=32,
                temperature=0.2,
                remasking="low_confidence",
                prelex=prelex,
                subtokens=subtokens,
                strip_chars=instance.strip_chars(),
                additional_stuff=additional_stuff,
                constrain=True,
                max_batch_size=8,
                max_remask_attempts=3,
            ):
                pass

            if valid:
                best_out = out
                best_valid = True
                best_violations = violations
                best_resamples = resamples_list
                best_sample_idx = k
                break

            # Keep the one with fewest violations
            if out is not None and violations < best_violations:
                best_out = out
                best_violations = violations
                best_resamples = resamples_list
                best_sample_idx = k

        elapsed = time.monotonic() - start_time

        if best_out is not None:
            best_code = tokenizer.batch_decode(
                best_out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]

        extracted = instance.extract_result(suffix + start_line + best_code)

        # Try autocompletion on best result if not valid
        autocompletion = None
        supertokens_map = derive_supertokens(subtokens) if subtokens else {}
        if best_out is not None and not best_valid:
            mask_id = 126336
            mask_decoded = tokenizer.decode(mask_id)
            eos_token = tokenizer.special_tokens_map["eos_token"]
            generated_words = tokenizer.batch_decode(best_out.squeeze())
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
                    prelex=prelex, subtokens=subtokens,
                    supertokens=supertokens_map,
                    strip_chars=instance.strip_chars(),
                ),
                subtokens=subtokens,
                lex_map=orig_lex_map,
                constraint_lang=lang,
            )
            if valid_completion is not None:
                autocompletion = instance.extract_result(suffix + valid_completion)

        result = {
            "instance_id": instance.instance_id(),
            "method": tag,
            "valid": best_valid,
            "sample_idx": best_sample_idx,
            "violations": best_violations,
            "extracted": extracted,
            "autocompletion": autocompletion,
            "time_taken": elapsed,
            "code_tail": best_code[-100:] if best_code else None,
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        ac_tag = "+AC" if autocompletion else ""
        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={best_valid}{ac_tag}, best_k={best_sample_idx}/{k_samples}, "
            f"violations={best_violations}, time={elapsed:.1f}s"
        )


def run_kl_projection():
    """KL projection: renormalize to valid tokens for selection,
    original distribution for confidence."""
    tag = "kl_proj"
    output_file = f"results/math_{tag}.jsonl"

    dataset = load_dataset("jsonschema")
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")
    torch.manual_seed(0)

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    hard_set = set(HARD_IDS)
    instances = [inst for inst in all_instances if inst.instance_id() in hard_set]
    print(f"KL projection on {len(instances)} hard instances")

    for i, instance in enumerate(instances):
        lang, lex_map, subtokens = instance.language_lex_subtokens()
        orig_lex_map = lex_map.copy()
        lang = lang.concatenate(CFG.from_text("S -> lexFence | $", "S"))
        if instance.strip_chars() is not None and "\n" not in instance.strip_chars():
            lex_map["lexFence"] = r"\n?```"
        else:
            lex_map["lexFence"] = "```"
        lang = lang.to_normal_form()

        raw_lex_map = lex_map.copy()
        lex_map = compile_lex_map(lex_map, subtokens=subtokens)
        prelex = instance.prelex()

        token_mask_table = TokenDFATable(
            tokenizer, raw_lex_map, lex_map, model.device,
            constraint_lang=lang,
            prelex=prelex,
            strip_chars=instance.strip_chars(),
        )

        additional_stuff = preprocessed_generate_stuff(
            tokenizer, lang, lex_map,
            prelex=prelex, subtokens=subtokens,
            strip_chars=instance.strip_chars(),
        )

        prompt_ids, prompt_len, suffix, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )

        start_time = time.monotonic()

        out = None
        for out, resamples_list, valid, violations, remasks, grammar_checks in generate_dgrammar(
            model, prompt_ids, tokenizer,
            constraint_lang=lang,
            lex_map=lex_map,
            prompt_len=prompt_len,
            steps=128,
            gen_length=256,
            block_length=32,
            temperature=0.2,
            remasking="low_confidence",
            prelex=prelex,
            subtokens=subtokens,
            strip_chars=instance.strip_chars(),
            additional_stuff=additional_stuff,
            constrain=True,
            max_batch_size=8,
            max_remask_attempts=3,
            token_mask_table=token_mask_table,
            soft_bias="kl_project",  # special mode
        ):
            pass

        elapsed = time.monotonic() - start_time

        if out is None:
            code = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1]:], skip_special_tokens=True
            )[0]

        extracted = instance.extract_result(suffix + start_line + code)

        result = {
            "instance_id": instance.instance_id(),
            "method": tag,
            "valid": valid,
            "violations": violations,
            "extracted": extracted,
            "time_taken": elapsed,
            "code_tail": code[-100:] if code else None,
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, violations={violations}, time={elapsed:.1f}s"
        )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "reject"

    if mode == "reject":
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        run_rejection_sampling(k)
    elif mode == "kl":
        run_kl_projection()


if __name__ == "__main__":
    main()
