"""Tests for string-level validators."""

import pytest
from dgrammar.validators import (
    StringLevelChecker,
    JsonValidator,
    CppValidator,
    SmilesValidator,
)
from dgrammar._core import MASK_TOKEN


class TestJsonValidator:
    def test_valid_json(self):
        v = JsonValidator()
        assert v.validate('{"name": "John", "age": 30}') is True

    def test_invalid_json(self):
        v = JsonValidator()
        assert v.validate('{"name": "John",}') is False

    def test_error_position(self):
        v = JsonValidator()
        pos = v.error_char_position('{"name": }')
        assert pos is not None

    def test_schema_validation(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        v = JsonValidator(schema=schema)
        assert v.validate('{"name": "John"}') is True
        assert v.validate('{"age": 30}') is False
        assert v.validate('not json') is False


class TestCppValidator:
    def test_valid_cpp(self):
        v = CppValidator()
        assert v.validate("int main() { return 0; }") is True

    def test_invalid_cpp(self):
        v = CppValidator()
        # Missing closing brace
        assert v.validate("int main() { return 0;") is False


class TestSmilesValidator:
    def test_valid_smiles(self):
        v = SmilesValidator()
        assert v.validate("CCO") is True
        assert v.validate("C(=O)O") is True

    def test_invalid_smiles(self):
        v = SmilesValidator()
        assert v.validate("C(=O") is False  # unbalanced paren
        assert v.validate("") is False


class TestStringLevelChecker:
    def test_check_valid(self):
        checker = StringLevelChecker(JsonValidator())
        tokens = ["{", '"name"', ":", '"John"', "}"]
        assert checker.check(tokens) is True

    def test_check_invalid(self):
        checker = StringLevelChecker(JsonValidator())
        tokens = ["{", '"name"', ":", "INVALID", "}"]
        assert checker.check(tokens) is False

    def test_check_with_masks(self):
        checker = StringLevelChecker(JsonValidator())
        # Masks are replaced with placeholder, should form valid JSON
        tokens = ["{", '"name"', ":", MASK_TOKEN, "}"]
        # With placeholder "x" -> {"name":"x"} which is valid
        assert checker.check(tokens) is True

    def test_find_violations(self):
        checker = StringLevelChecker(JsonValidator())
        tokens = ["{", '"name"', ":", "BAD", "}"]
        violations = checker.find_violations(tokens, [3])
        assert 3 in violations

    def test_find_violations_none(self):
        checker = StringLevelChecker(JsonValidator())
        tokens = ["{", '"name"', ":", '"val"', "}"]
        violations = checker.find_violations(tokens, [3])
        assert violations == []
