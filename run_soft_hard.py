"""Run soft constrained decoding on the 11 hard instances only."""

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


def main():
    soft_bias = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    tag = f"hard_sb{soft_bias}"
    output_file = f"results/soft_{tag}.jsonl"

    dataset = load_dataset("jsonschema")
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    seed = 0
    torch.manual_seed(seed)

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")

    # Filter to hard instances only
    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    hard_set = set(HARD_IDS)
    instances = [inst for inst in all_instances if inst.instance_id() in hard_set]
    print(f"Running soft_bias={soft_bias} on {len(instances)} hard instances")

    lang, lex_map, subtokens = None, None, None
    prelex = None
    additional_stuff = None
    token_mask_table = None
    orig_lex_map = None

    for i, instance in enumerate(instances):
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

            # Only build TokenDFATable if soft_bias > 0
            if soft_bias > 0:
                t0 = time.monotonic()
                token_mask_table = TokenDFATable(
                    tokenizer, raw_lex_map, lex_map, model.device,
                    constraint_lang=lang,
                    prelex=prelex,
                    strip_chars=instance.strip_chars(),
                )
                print(f"  TokenDFATable built in {time.monotonic() - t0:.2f}s")
            else:
                token_mask_table = None

        if additional_stuff is None:
            additional_stuff = preprocessed_generate_stuff(
                tokenizer, lang, lex_map,
                prelex=prelex,
                subtokens=subtokens, strip_chars=instance.strip_chars(),
            )

        prompt_ids, prompt_len, suffix, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
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
            soft_bias=soft_bias if soft_bias > 0 else None,
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
                    prelex=prelex,
                    subtokens=subtokens,
                    supertokens=supertokens_map,
                    strip_chars=instance.strip_chars(),
                ),
                subtokens=subtokens,
                lex_map=orig_lex_map,
                constraint_lang=lang,
            )
            time_taken_autocompletion = time.monotonic() - ac_start
            if valid_completion is not None:
                autocompletion_raw = valid_completion
                autocompletion = instance.extract_result(
                    suffix + valid_completion
                )

        result = {
            "instance_id": instance.instance_id(),
            "constrained": f"soft_bias_{soft_bias}",
            "code": code,
            "extracted": extracted,
            "autocompletion": autocompletion,
            "time_taken": elapsed,
            "valid": valid,
            "dgrammar_stats": {
                "violations": total_violations,
                "remasks": total_remasks,
                "grammar_checks": total_grammar_checks,
                "soft_bias": soft_bias,
            },
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"valid={valid}, violations={total_violations}, remasks={total_remasks}, "
            f"time={elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
