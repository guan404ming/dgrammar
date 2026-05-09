"""Token-level grammar checker using llguidance.

Replaces the character-level DFA intersection approach (rustformlang) with
llguidance's incremental token-level parser. ~9000x faster per check.
"""

import torch
from llguidance import LLMatcher, LLTokenizer, LLParserLimits
from huggingface_hub import hf_hub_download


class TokenChecker:
    """Grammar checker operating directly on token IDs."""

    def __init__(self, grammar: str, model_name: str = "GSAI-ML/LLaDA-8B-Instruct"):
        try:
            tokenizer_path = hf_hub_download(
                repo_id=model_name, filename="tokenizer.json"
            )
            self.tokenizer = LLTokenizer(tokenizer_path)
        except Exception:
            self.tokenizer = LLTokenizer(self._derive_tokenizer_json(model_name))

        import json
        try:
            json.loads(grammar)
            grm = LLMatcher.grammar_from_json_schema(grammar)
        except (json.JSONDecodeError, TypeError):
            grm = LLMatcher.grammar_from_lark(grammar)

        is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)
        assert not is_err, "Grammar is not valid"

        self._grammar = grm
        limits = LLParserLimits(max_items_in_row=20000, step_max_items=600000)
        self.matcher = LLMatcher(self.tokenizer, grm, log_level=1, limits=limits)
        self._limits = limits

    def _derive_tokenizer_json(self, model_name: str) -> str:
        """Build a tokenizer.json for repos that don't ship one (e.g. Dream).

        Dream-7B reuses the Qwen2.5-7B-Instruct vocabulary plus a `<|mask|>`
        added token at id 151666. We download Qwen's tokenizer.json and patch
        in the extra special token.
        """
        import json
        import os
        import tempfile

        name_lower = model_name.lower()
        if "dream" in name_lower or "diffucoder" in name_lower:
            base_path = hf_hub_download(
                repo_id="Qwen/Qwen2.5-7B-Instruct", filename="tokenizer.json"
            )
            with open(base_path) as f:
                spec = json.load(f)
            mask_tok = {
                "id": 151666,
                "content": "<|mask|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            spec.setdefault("added_tokens", [])
            if not any(t.get("id") == 151666 for t in spec["added_tokens"]):
                spec["added_tokens"].append(mask_tok)
            self._tok_tmp = tempfile.TemporaryDirectory()
            out_path = os.path.join(self._tok_tmp.name, "tokenizer.json")
            with open(out_path, "w") as f:
                json.dump(spec, f)
            return out_path
        raise NotImplementedError(
            f"No tokenizer.json available for {model_name} and no fallback registered"
        )

    def validate_tokens(self, token_ids: list[int]) -> bool:
        """Check if a token sequence is valid (can be extended to a complete valid string)."""
        if not token_ids:
            return True
        count = self.matcher.validate_tokens(token_ids)
        return count == len(token_ids)

    def compute_mask(self, vocab_size: int = None) -> torch.Tensor:
        """Compute context-dependent valid token mask for the current state.

        Returns a bool tensor of shape [vocab_size] where True = blocked.
        """
        import numpy as np
        bias = self.matcher.compute_logit_bias()
        arr = np.frombuffer(bias, dtype=np.uint8)
        if vocab_size is None:
            vocab_size = len(arr)
        # 0 = blocked, >0 = allowed
        mask = torch.ones(vocab_size, dtype=torch.bool)
        blocked = torch.from_numpy((arr == 0).copy())
        n = min(len(arr), vocab_size)
        mask[:n] = blocked[:n]
        return mask

    def consume_tokens(self, token_ids: list[int]) -> bool:
        """Advance parser state by consuming tokens. Returns False if invalid."""
        if not token_ids:
            return True
        count = self.matcher.try_consume_tokens(token_ids)
        if count != len(token_ids):
            self.matcher.rollback(count)
            return False
        return True

    def rollback(self, count: int) -> bool:
        """Roll back the last `count` consumed tokens."""
        if count <= 0:
            return True
        return self.matcher.rollback(count)

    def is_accepting(self) -> bool:
        """Check if current state is a valid end state."""
        return self.matcher.is_accepting()

    def is_stopped(self) -> bool:
        """Check if parser has reached a terminal state."""
        return self.matcher.is_stopped()

    def reset(self):
        """Reset parser to initial state."""
        self.matcher.reset()

    def clone(self):
        """Create a new checker with the same grammar (fresh state)."""
        checker = TokenChecker.__new__(TokenChecker)
        checker.tokenizer = self.tokenizer
        checker._grammar = self._grammar
        checker._limits = self._limits
        checker.matcher = LLMatcher(
            self.tokenizer, self._grammar, log_level=1, limits=self._limits
        )
        return checker
