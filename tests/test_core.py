"""Tests for the Rust core (Earley parser + violation detection)."""

import pytest
from dgrammar._core import GrammarChecker, MASK_TOKEN


SIMPLE_GRAMMAR = """
start: S
S -> a b c d
"""

JSON_GRAMMAR = """
start: S
S -> "{" Pairs "}"
Pairs -> Pair | Pair "," Pairs
Pair -> Key ":" Value
Key -> "name" | "age" | "active"
Value -> "str" | "num" | "true" | "false" | "null"
"""

RECURSIVE_GRAMMAR = """
start: E
E -> E "+" E | E "*" E | "(" E ")" | n
"""


class TestGrammarChecker:
    def test_check_valid(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        assert gc.check(["a", "b", "c", "d"]) is True

    def test_check_invalid(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        assert gc.check(["a", "x", "c", "d"]) is False

    def test_check_empty_invalid(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        assert gc.check([]) is False

    def test_mask_tokens(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        assert gc.check(["a", MASK_TOKEN, "c", "d"]) is True
        assert gc.check([MASK_TOKEN, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN]) is True

    def test_error_position(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        assert gc.error_position(["a", "b", "c", "d"]) is None
        assert gc.error_position(["a", "b", "x", "d"]) == 2

    def test_json_grammar(self):
        gc = GrammarChecker(JSON_GRAMMAR)
        assert gc.check(["{", "name", ":", "str", "}"]) is True
        assert gc.check([
            "{", "name", ":", "str", ",", "age", ":", "num", "}"
        ]) is True
        assert gc.check(["{", "bad", ":", "str", "}"]) is False

    def test_recursive_grammar(self):
        gc = GrammarChecker(RECURSIVE_GRAMMAR)
        assert gc.check(["n"]) is True
        assert gc.check(["n", "+", "n"]) is True
        assert gc.check(["(", "n", "+", "n", ")", "*", "n"]) is True


class TestViolationDetection:
    def test_no_violations(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        violations = gc.find_violations(["a", "b", "c", "d"], [1, 2])
        assert violations == []

    def test_single_violation(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        violations = gc.find_violations(["a", "x", "c", "d"], [1])
        assert 1 in violations

    def test_violation_not_in_unmasked(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        # Position 1 is wrong but not in unmasked_positions
        violations = gc.find_violations(["a", "x", "c", "d"], [2, 3])
        # Medium path: masking 2 or 3 won't fix position 1
        # Falls to slow path: returns [2, 3]
        assert len(violations) > 0

    def test_greedy_detection(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        violations = gc.find_violations_greedy(
            ["a", "x", "c", "d"], [1, 2, 3]
        )
        assert 1 in violations
        assert 2 not in violations
        assert 3 not in violations

    def test_empty_unmasked(self):
        gc = GrammarChecker(SIMPLE_GRAMMAR)
        violations = gc.find_violations(["a", "x", "c", "d"], [])
        assert violations == []


class TestGrammarParsing:
    def test_invalid_grammar(self):
        with pytest.raises(ValueError):
            GrammarChecker("")

    def test_quoted_terminals(self):
        grammar = '''
        start: S
        S -> "{" "}" | "{" Body "}"
        Body -> "x"
        '''
        gc = GrammarChecker(grammar)
        assert gc.check(["{", "}"]) is True
        assert gc.check(["{", "x", "}"]) is True

    def test_epsilon_production(self):
        grammar = """
        start: S
        S -> a B c
        B -> b |
        """
        gc = GrammarChecker(grammar)
        assert gc.check(["a", "b", "c"]) is True
        assert gc.check(["a", "c"]) is True
