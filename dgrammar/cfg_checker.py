"""rustformlang-backed CFG checker for Dgr on cpp/smiles benchmarks.

Mirrors the TokenChecker interface (matcher.try_consume_tokens, compute_mask,
clone, rollback, is_accepting) but uses CFG-DFA intersection-emptiness via
rustformlang. Per-token mask is not produced (returns zero bias), so Dgr falls
back to selective remasking only.
"""

import torch

from constrained_diffusion.constrain_utils import EOS, lex
from constrained_diffusion.eval.dllm.models.llada.generate_constrained import (
    check_valid,
)


class _CFGMatcher:
    def __init__(self, parent):
        self.parent = parent

    def try_consume_tokens(self, token_ids):
        words = self.parent._tokens_to_words(token_ids)
        old_len = len(self.parent.generated_words)
        self.parent.generated_words.extend(words)
        try:
            invalid = check_valid(
                self.parent.generated_words,
                self.parent.constraint_lang,
                self.parent.lex_map,
                self.parent.terminals,
                prelex=self.parent.prelex,
                subtokens=self.parent.subtokens,
                strip_chars=self.parent.strip_chars,
            )
        except Exception:
            invalid = True
        if invalid:
            del self.parent.generated_words[old_len:]
            return 0
        return len(token_ids)

    def is_accepting(self):
        return self.parent.is_accepting()

    def rollback(self, n):
        if n > 0:
            del self.parent.generated_words[-n:]


class CFGTokenChecker:
    """rustformlang-backed checker compatible with TokenChecker's surface."""

    def __init__(self, constraint_lang, lex_map, terminals, prelex, subtokens,
                 strip_chars, tokenizer, eos_decoded, eot_decoded,
                 vocab_size: int):
        self.constraint_lang = constraint_lang
        self.lex_map = lex_map
        self.terminals = terminals
        self.prelex = prelex
        self.subtokens = subtokens
        self.strip_chars = strip_chars or ""
        self.tokenizer = tokenizer
        self.eos_decoded = eos_decoded
        self.eot_decoded = eot_decoded
        self.vocab_size = vocab_size
        self.generated_words = []
        self.matcher = _CFGMatcher(self)

    def _tokens_to_words(self, token_ids):
        out = []
        for tid in token_ids:
            w = self.tokenizer.decode(tid)
            if w in (self.eos_decoded, self.eot_decoded):
                out.append(EOS)
            else:
                out.append(w)
        return out

    def consume_tokens(self, token_ids):
        return self.matcher.try_consume_tokens(token_ids)

    def is_accepting(self):
        text = "".join(w for w in self.generated_words if isinstance(w, str))
        try:
            for lexed_word, unfin, unfin_prefix in lex(
                text, self.lex_map, is_first=True
            ):
                if not unfin and not unfin_prefix:
                    if self.constraint_lang.accepts(lexed_word):
                        return True
            return False
        except Exception:
            return False

    def is_stopped(self):
        return False

    def compute_mask(self, vocab_size=None):
        return torch.zeros(vocab_size or self.vocab_size, dtype=torch.bool)

    def clone(self):
        new = CFGTokenChecker.__new__(CFGTokenChecker)
        new.constraint_lang = self.constraint_lang
        new.lex_map = self.lex_map
        new.terminals = self.terminals
        new.prelex = self.prelex
        new.subtokens = self.subtokens
        new.strip_chars = self.strip_chars
        new.tokenizer = self.tokenizer
        new.eos_decoded = self.eos_decoded
        new.eot_decoded = self.eot_decoded
        new.vocab_size = self.vocab_size
        new.generated_words = list(self.generated_words)
        new.matcher = _CFGMatcher(new)
        return new


def build_cfg_checker(instance, tokenizer, vocab_size: int):
    """Construct a CFGTokenChecker from an Instance providing CFG.

    Mirrors the preprocessing pipeline used by IG-CD's runner so the
    constraint matches the same lexFence-augmented grammar.
    """
    from constrained_diffusion.constrain_utils import compile_lex_map
    from rustformlang.cfg import CFG

    lang, lex_map, subtokens = instance.language_lex_subtokens()
    lang = lang.concatenate(CFG.from_text("S -> lexFence | $", "S"))
    sc = instance.strip_chars()
    if sc is not None and "\n" not in sc:
        lex_map["lexFence"] = r"\n?```"
    else:
        lex_map["lexFence"] = "```"
    lang = lang.to_normal_form()
    compiled = compile_lex_map(lex_map, subtokens=subtokens)
    terminals = lang.get_terminals()
    eos_decoded = tokenizer.decode(tokenizer.eos_token_id) if hasattr(tokenizer, "eos_token_id") else ""
    eot_decoded = "<|eot_id|>"
    return CFGTokenChecker(
        constraint_lang=lang,
        lex_map=compiled,
        terminals=terminals,
        prelex=instance.prelex(),
        subtokens=subtokens,
        strip_chars=sc,
        tokenizer=tokenizer,
        eos_decoded=eos_decoded,
        eot_decoded=eot_decoded,
        vocab_size=vocab_size,
    )
