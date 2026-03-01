"""Measure grammar-aware pre-filter coverage and cost."""
import time, torch, numpy as np
from constrained_diffusion.constrain_utils import (
    compile_lex_map, all_lexings_mask, generated_language, derive_supertokens,
)
from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from rustformlang.cfg import CFG
import frozendict

dataset = load_dataset("jsonschema")
instance = sorted(dataset, key=lambda x: x.instance_id())[0]
lang, lex_map_raw, subtokens = instance.language_lex_subtokens()
lang = lang.concatenate(CFG.from_text("S -> lexFence | $", "S"))
strip_chars = instance.strip_chars()
if strip_chars is not None and "\n" not in strip_chars:
    lex_map_raw["lexFence"] = r"\n?```"
else:
    lex_map_raw["lexFence"] = "```"
lang = lang.to_normal_form()
lex_map = compile_lex_map(lex_map_raw, subtokens=subtokens)
prelex = instance.prelex()
terminals = lang.get_terminals()
supertokens = derive_supertokens(subtokens)

eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
tokenizer = eval_model.tokenizer("cpu")
vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)

print(f"Vocab size: {vocab_size}")
print(f"Terminals: {len(terminals)}")

# Step 1: all_lexings_mask (Rust-accelerated)
t0 = time.monotonic()
all_tokens_decoded = tokenizer.batch_decode(torch.arange(0, vocab_size))
all_possible_lexings, no_lexing_arr = all_lexings_mask(
    all_tokens_decoded, lex_map, trace=False,
    strip_chars=strip_chars, prelex=prelex,
)
t1 = time.monotonic()
n_basic = int((no_lexing_arr > 0.5).sum())
n_lexings = len(all_possible_lexings)
print(f"\nStep 1 - Lexing ({t1-t0:.1f}s):")
print(f"  No-lexing tokens: {n_basic}/{vocab_size} ({n_basic/vocab_size*100:.1f}%)")
print(f"  Unique lexings: {n_lexings}")

# Step 2: grammar-aware pruning
t2 = time.monotonic()
pruned = 0
lexings_list = list(all_possible_lexings.keys())
for i, lexing in enumerate(lexings_list):
    lexing_lang = generated_language(
        [None, lexing, None],
        lex_map, terminals,
        prelex=prelex,
        subtokens=subtokens,
        supertokens=supertokens,
        strip_chars=strip_chars,
    )
    intersection_empty = lang.is_intersection_empty(lexing_lang, 100)
    if intersection_empty:
        no_lexing_arr += all_possible_lexings[lexing]
        del all_possible_lexings[lexing]
        pruned += 1
    if (i + 1) % 100 == 0:
        print(f"  Checked {i+1}/{n_lexings}, pruned {pruned} so far...")

t3 = time.monotonic()
n_grammar = int((no_lexing_arr > 0.5).sum())
print(f"\nStep 2 - Grammar pruning ({t3-t2:.1f}s):")
print(f"  Lexings pruned: {pruned}/{n_lexings}")
print(f"  Remaining lexings: {len(all_possible_lexings)}")
print(f"  Total blocked tokens: {n_grammar}/{vocab_size} ({n_grammar/vocab_size*100:.1f}%)")
print(f"\nTotal time: {t3-t0:.1f}s")
