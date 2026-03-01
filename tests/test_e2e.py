"""End-to-end tests for the dGrammar decode loop using a mock dLLM.

The mock model simulates a diffusion LLM that predicts tokens for masked
positions based on a predefined target sequence. This lets us test the full
decode loop (unmask, grammar check, selective remask, fallback) without
loading a real model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import pytest

from dgrammar._core import GrammarChecker, MASK_TOKEN
from dgrammar.decode import DecodeConfig, decode, _ids_to_token_strings
from dgrammar.baselines import BaselineConfig, decode_no_cd, decode_fscd, decode_lave


# --- Mock model and tokenizer ---

class MockTokenizer:
    """Simple tokenizer where each word is a token."""

    def __init__(self, vocab: list[str], mask_token: str = "[MASK]"):
        self.vocab = vocab
        self.mask_token = mask_token
        self._tok2id = {tok: i for i, tok in enumerate(vocab)}
        self.mask_token_id = self._tok2id[mask_token]

    def encode(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        ids = [self._tok2id[w] for w in text.split()]
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens = []
        for i in ids:
            tok = self.vocab[i]
            if skip_special_tokens and tok == self.mask_token:
                continue
            tokens.append(tok)
        return " ".join(tokens)


@dataclass
class MockLogits:
    logits: torch.Tensor


class MockDLLM:
    """Mock diffusion LLM that predicts a target sequence.

    For each masked position, it produces high logits for the corresponding
    target token and low logits elsewhere. For non-masked positions, logits
    are irrelevant (already unmasked).

    If bad_positions is set, those target positions will have a "wrong" token
    with the highest confidence on the first call, simulating grammar violations
    that the selective remask should fix.
    """

    def __init__(
        self,
        vocab_size: int,
        target_ids: list[int],
        prompt_len: int,
        bad_positions: list[int] | None = None,
        bad_token_id: int | None = None,
    ):
        self.vocab_size = vocab_size
        self.target_ids = target_ids
        self.prompt_len = prompt_len
        self.bad_positions = set(bad_positions or [])
        self.bad_token_id = bad_token_id
        self._call_count = 0

    def __call__(self, x: torch.Tensor) -> MockLogits:
        self._call_count += 1
        batch_size, seq_len = x.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)

        for pos in range(self.prompt_len, seq_len):
            gen_pos = pos - self.prompt_len
            if gen_pos < len(self.target_ids):
                target_id = self.target_ids[gen_pos]

                # On first call, bad positions predict the wrong token
                if gen_pos in self.bad_positions and self._call_count == 1:
                    logits[0, pos, self.bad_token_id] = 10.0
                    logits[0, pos, target_id] = 5.0
                else:
                    logits[0, pos, target_id] = 10.0

        return MockLogits(logits=logits)


# --- Test fixtures ---

VOCAB = ["[MASK]", "{", "}", ":", ",", "key1", "key2", "val1", "val2",
         "true", "false", "null", "a", "b", "c", "d", "x", "PROMPT"]

JSON_GRAMMAR = """
start: S
S -> "{" Pairs "}"
Pairs -> Pair | Pair "," Pairs
Pair -> Key ":" Value
Key -> "key1" | "key2"
Value -> "val1" | "val2" | "true" | "false"
"""

SIMPLE_GRAMMAR = """
start: S
S -> a b c d
"""


def make_tokenizer():
    return MockTokenizer(VOCAB)


def tok_id(tokenizer, word):
    return tokenizer._tok2id[word]


# --- E2E Tests ---

class TestDecodeE2E:
    """Test the full dGrammar decode loop."""

    def test_correct_prediction_no_violations(self):
        """When model predicts all correct tokens, no remasking needed."""
        tokenizer = make_tokenizer()
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]
        prompt_ids = [tok_id(tokenizer, "PROMPT")]

        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
        )
        checker = GrammarChecker(SIMPLE_GRAMMAR)
        config = DecodeConfig(output_length=4, total_steps=10, max_unmask_per_step=4)

        output, stats = decode(model, tokenizer, "PROMPT", checker, config, device="cpu")

        assert "a b c d" == output
        assert stats.total_violations == 0
        assert stats.total_remasks == 0
        assert stats.degraded_to_fallback is False

    def test_violation_triggers_remask(self):
        """When model predicts a wrong token, selective remask should fix it."""
        tokenizer = make_tokenizer()
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]
        prompt_ids = [tok_id(tokenizer, "PROMPT")]

        # Position 1 (token "b") will be predicted as "x" on first call
        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
            bad_positions=[1],
            bad_token_id=tok_id(tokenizer, "x"),
        )
        checker = GrammarChecker(SIMPLE_GRAMMAR)
        config = DecodeConfig(
            output_length=4, total_steps=10,
            max_unmask_per_step=4, max_remask_attempts=3,
        )

        output, stats = decode(model, tokenizer, "PROMPT", checker, config, device="cpu")

        assert "a b c d" == output
        assert stats.total_violations > 0
        assert stats.total_remasks > 0

    def test_fallback_on_persistent_violation(self):
        """When violations persist, fallback to per-token mode."""
        tokenizer = make_tokenizer()
        # Target is valid
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]

        # All positions are bad on first call. Model always returns bad on call 1,
        # but correct after remask. This tests the fallback path.
        class AlwaysBadFirstModel:
            def __init__(self):
                self.call_count = 0

            def __call__(self, x):
                self.call_count += 1
                batch_size, seq_len = x.shape
                logits = torch.zeros(batch_size, seq_len, len(VOCAB))
                for pos in range(1, seq_len):
                    gen_pos = pos - 1
                    if gen_pos < len(target):
                        if self.call_count <= 2:
                            # Predict wrong token
                            logits[0, pos, tok_id(tokenizer, "x")] = 10.0
                        else:
                            # After a few calls, predict correctly
                            logits[0, pos, target[gen_pos]] = 10.0
                return MockLogits(logits=logits)

        model = AlwaysBadFirstModel()
        checker = GrammarChecker(SIMPLE_GRAMMAR)
        config = DecodeConfig(
            output_length=4, total_steps=10,
            max_unmask_per_step=4, max_remask_attempts=2,
            enable_fallback=True, fallback_max_steps=20,
        )

        output, stats = decode(model, tokenizer, "PROMPT", checker, config, device="cpu")

        # Should eventually produce valid output through fallback
        assert stats.total_violations > 0

    def test_json_grammar_decode(self):
        """Test dGrammar with a JSON-like grammar."""
        tokenizer = make_tokenizer()
        # Target: { key1 : val1 }
        target_words = ["{", "key1", ":", "val1", "}"]
        target = [tok_id(tokenizer, w) for w in target_words]

        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
        )
        checker = GrammarChecker(JSON_GRAMMAR)
        config = DecodeConfig(output_length=5, total_steps=10, max_unmask_per_step=5)

        output, stats = decode(model, tokenizer, "PROMPT", checker, config, device="cpu")

        assert output == "{ key1 : val1 }"
        assert stats.total_violations == 0

    def test_json_with_violation_and_repair(self):
        """Test JSON grammar where a value token is initially wrong."""
        tokenizer = make_tokenizer()
        target_words = ["{", "key1", ":", "val1", "}"]
        target = [tok_id(tokenizer, w) for w in target_words]

        # Position 3 (val1) initially predicted as "x"
        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
            bad_positions=[3],
            bad_token_id=tok_id(tokenizer, "x"),
        )
        checker = GrammarChecker(JSON_GRAMMAR)
        config = DecodeConfig(
            output_length=5, total_steps=10,
            max_unmask_per_step=5, max_remask_attempts=3,
        )

        output, stats = decode(model, tokenizer, "PROMPT", checker, config, device="cpu")

        assert output == "{ key1 : val1 }"
        assert stats.total_violations > 0


class TestBaselineE2E:
    """Test baseline decoders."""

    def test_no_cd_decode(self):
        tokenizer = make_tokenizer()
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]

        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
        )
        config = BaselineConfig(output_length=4, total_steps=10)

        output, stats = decode_no_cd(model, tokenizer, "PROMPT", config, device="cpu")

        assert output == "a b c d"
        assert stats.total_steps == 4  # one token per step

    def test_fscd_decode_valid(self):
        tokenizer = make_tokenizer()
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]

        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
        )
        checker = GrammarChecker(SIMPLE_GRAMMAR)
        config = BaselineConfig(output_length=4, total_steps=10)

        output, stats = decode_fscd(
            model, tokenizer, "PROMPT", checker, config, device="cpu"
        )

        assert output == "a b c d"

    def test_lave_decode_valid(self):
        tokenizer = make_tokenizer()
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]

        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
        )
        checker = GrammarChecker(SIMPLE_GRAMMAR)
        config = BaselineConfig(output_length=4, total_steps=10, lookahead_k=2)

        output, stats = decode_lave(
            model, tokenizer, "PROMPT", checker, config, device="cpu"
        )

        assert output == "a b c d"


class TestTokenMapper:
    """Test the token mapping layer."""

    def test_identity_mapper(self):
        tokenizer = make_tokenizer()
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]

        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
        )
        checker = GrammarChecker(SIMPLE_GRAMMAR)
        config = DecodeConfig(output_length=4, total_steps=10, max_unmask_per_step=4)

        # Default identity mapper
        output, stats = decode(model, tokenizer, "PROMPT", checker, config, device="cpu")
        assert output == "a b c d"

    def test_custom_mapper(self):
        """Test with a custom token mapper that lowercases."""
        tokenizer = make_tokenizer()
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]

        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
        )
        checker = GrammarChecker(SIMPLE_GRAMMAR)
        config = DecodeConfig(output_length=4, total_steps=10, max_unmask_per_step=4)

        def lower_mapper(tokens):
            return [t.lower() for t in tokens]

        output, stats = decode(
            model, tokenizer, "PROMPT", checker, config,
            token_mapper=lower_mapper, device="cpu",
        )
        assert output == "a b c d"


class TestDecodeStats:
    """Test that statistics are correctly tracked."""

    def test_stats_no_violations(self):
        tokenizer = make_tokenizer()
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]

        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
        )
        checker = GrammarChecker(SIMPLE_GRAMMAR)
        config = DecodeConfig(output_length=4, total_steps=10, max_unmask_per_step=4)

        _, stats = decode(model, tokenizer, "PROMPT", checker, config, device="cpu")

        assert stats.total_steps >= 1
        assert stats.total_violations == 0
        assert stats.total_remasks == 0
        assert stats.degraded_to_fallback is False
        assert len(stats.tokens_per_step) > 0
        assert sum(stats.tokens_per_step) == 4  # unmasked all 4 tokens

    def test_stats_with_violations(self):
        tokenizer = make_tokenizer()
        target = [tok_id(tokenizer, w) for w in ["a", "b", "c", "d"]]

        model = MockDLLM(
            vocab_size=len(VOCAB),
            target_ids=target,
            prompt_len=1,
            bad_positions=[2],
            bad_token_id=tok_id(tokenizer, "x"),
        )
        checker = GrammarChecker(SIMPLE_GRAMMAR)
        config = DecodeConfig(
            output_length=4, total_steps=10,
            max_unmask_per_step=4, max_remask_attempts=3,
        )

        _, stats = decode(model, tokenizer, "PROMPT", checker, config, device="cpu")

        assert stats.total_violations > 0
        assert stats.total_remasks > 0
