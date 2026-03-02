"""Run dGrammar with type-aware masking on jsonschema benchmark."""

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
from dgrammar.type_mask import TypeAwareMasker


def run_benchmark(
    dataset_name: str,
    seed: int = 0,
    limit: int = 1000,
    offset: int = 0,
    output_file: str = "results/dgrammar_typemask.jsonl",
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
    mask_id = 126336

    lang, lex_map, subtokens = None, None, None
    prelex = None
    additional_stuff = None

    # Cache type maskers per schema to avoid recomputing
    type_masker_cache = {}

    instances = sorted(dataset, key=lambda x: x.instance_id())[offset:offset + limit]
    print(f"Running dGrammar+typemask on {len(instances)} {dataset_name} instances, seed={seed}")

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

        # Build type masker from schema
        schema_str = instance.data.get("schema", "")
        type_masker = None
        if schema_str:
            if schema_str not in type_masker_cache:
                try:
                    type_masker_cache[schema_str] = TypeAwareMasker(
                        schema_str, tokenizer, device=device,
                    )
                except Exception as e:
                    if trace:
                        print(f"  TypeAwareMasker failed: {e}")
                    type_masker_cache[schema_str] = None
            type_masker = type_masker_cache[schema_str]

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
            type_masker=type_masker,
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
            "constrained": "dgrammar_typemask",
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
                "type_masker": type_masker is not None,
            },
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            print(json.dumps(result), flush=True, file=f)

        print(
            f"  [{i+1}/{len(instances)}] {instance.instance_id()}: "
            f"violations={total_violations}, remasks={total_remasks}, "
            f"checks={total_grammar_checks}, typemask={'yes' if type_masker else 'no'}, "
            f"time={elapsed:.1f}s"
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
        output_file=f"results/dgrammar_typemask_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl",
    )


if __name__ == "__main__":
    main()
