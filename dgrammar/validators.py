"""String-level validators for JSON, C++, and SMILES.

These validators work on the decoded string (after concatenating BPE tokens)
rather than on individual grammar terminals. They wrap external tools like
jsonschema and tree-sitter to provide validation and error localization.

For the decode loop, use StringLevelChecker which adapts these validators
to the GrammarChecker-like interface expected by decode().
"""

from __future__ import annotations

import json
import re
from typing import Any

from dgrammar._core import MASK_TOKEN


class StringLevelChecker:
    """Wraps a string-level validator to work with the decode loop.

    Instead of checking individual token strings against a CFG,
    this concatenates tokens into a string and validates the whole string.
    For violation detection, it tests removing each token.
    """

    def __init__(self, validator: StringValidator):
        self.validator = validator

    def check(self, tokens: list[str]) -> bool:
        """Check if the token sequence forms a valid string."""
        # Mask tokens are wildcards, fill with placeholder
        filled = self._fill_masks(tokens)
        text = self._join_tokens(filled)
        return self.validator.validate(text)

    def find_violations(
        self,
        tokens: list[str],
        unmasked_positions: list[int],
    ) -> list[int]:
        """Find which unmasked positions cause validation failure."""
        if not unmasked_positions:
            return []

        filled = self._fill_masks(tokens)
        text = self._join_tokens(filled)

        if self.validator.validate(text):
            return []

        # Fast path: check error position from validator
        err_pos = self.validator.error_char_position(text)
        if err_pos is not None:
            token_pos = self._char_to_token_pos(filled, err_pos)
            if token_pos is not None and token_pos in unmasked_positions:
                return [token_pos]

        # Medium path: test each unmasked position
        violators = []
        for pos in unmasked_positions:
            if pos >= len(tokens):
                continue
            test_tokens = list(filled)
            test_tokens[pos] = self.validator.placeholder
            test_text = self._join_tokens(test_tokens)
            if self.validator.validate(test_text):
                violators.append(pos)

        if violators:
            return violators

        # Slow fallback
        return [p for p in unmasked_positions if p < len(tokens)]

    def find_violations_greedy(
        self,
        tokens: list[str],
        unmasked_positions: list[int],
    ) -> list[int]:
        """Greedy violation detection at string level."""
        if not unmasked_positions:
            return []

        filled = self._fill_masks(tokens)
        text = self._join_tokens(filled)
        if self.validator.validate(text):
            return []

        # Mask all unmasked positions, then add back one by one
        test_tokens = list(filled)
        for pos in unmasked_positions:
            if pos < len(test_tokens):
                test_tokens[pos] = self.validator.placeholder

        if not self.validator.validate(self._join_tokens(test_tokens)):
            return [p for p in unmasked_positions if p < len(tokens)]

        violators = []
        for pos in unmasked_positions:
            if pos >= len(test_tokens):
                continue
            test_tokens[pos] = filled[pos]
            if not self.validator.validate(self._join_tokens(test_tokens)):
                violators.append(pos)
                test_tokens[pos] = self.validator.placeholder
        return violators

    def _fill_masks(self, tokens: list[str]) -> list[str]:
        """Replace MASK tokens with validator-specific placeholders."""
        return [
            self.validator.placeholder if t == MASK_TOKEN else t
            for t in tokens
        ]

    def _join_tokens(self, tokens: list[str]) -> str:
        """Join token strings. Handles BPE tokens with leading spaces."""
        parts = []
        for t in tokens:
            if t.startswith(" ") or not parts:
                parts.append(t)
            else:
                parts.append(t)
        return "".join(parts)

    def _char_to_token_pos(self, tokens: list[str], char_pos: int) -> int | None:
        """Map a character position to a token index."""
        offset = 0
        for i, tok in enumerate(tokens):
            end = offset + len(tok)
            if offset <= char_pos < end:
                return i
            offset = end
        return None


class StringValidator:
    """Base class for string-level validators."""

    placeholder: str = " "

    def validate(self, text: str) -> bool:
        raise NotImplementedError

    def error_char_position(self, text: str) -> int | None:
        """Return the character position of the first error, if determinable."""
        return None


class JsonValidator(StringValidator):
    """Validates JSON strings, optionally against a JSON Schema.

    Handles markdown code fences (```json ... ```) by stripping them.
    """

    placeholder = '"x"'

    def __init__(self, schema: dict[str, Any] | None = None):
        self.schema = schema

    def validate(self, text: str) -> bool:
        text = self._strip_fences(text)
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return False

        if self.schema is not None:
            try:
                import jsonschema
                jsonschema.validate(obj, self.schema)
            except (jsonschema.ValidationError, jsonschema.SchemaError):
                return False

        return True

    def error_char_position(self, text: str) -> int | None:
        text = self._strip_fences(text)
        try:
            json.loads(text)
            return None
        except json.JSONDecodeError as e:
            return e.pos
        except ValueError:
            return None

    def _strip_fences(self, text: str) -> str:
        """Strip markdown code fences if present."""
        text = text.strip()
        # Handle ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            if lines:
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip()


class CppValidator(StringValidator):
    """Validates C++ code using tree-sitter."""

    placeholder = "0"

    def validate(self, text: str) -> bool:
        try:
            import tree_sitter_cpp as tscpp
            import tree_sitter as ts
        except ImportError:
            # Fallback: basic brace/paren matching
            return self._basic_check(text)

        lang = ts.Language(tscpp.language())
        parser = ts.Parser(lang)
        tree = parser.parse(bytes(text, "utf8"))
        return not tree.root_node.has_error

    def error_char_position(self, text: str) -> int | None:
        try:
            import tree_sitter_cpp as tscpp
            import tree_sitter as ts
        except ImportError:
            return None

        lang = ts.Language(tscpp.language())
        parser = ts.Parser(lang)
        tree = parser.parse(bytes(text, "utf8"))

        if not tree.root_node.has_error:
            return None

        # Find the first ERROR node
        return self._find_error_pos(tree.root_node)

    def _find_error_pos(self, node) -> int | None:
        if node.type == "ERROR" or node.is_missing:
            return node.start_byte
        for child in node.children:
            pos = self._find_error_pos(child)
            if pos is not None:
                return pos
        return None

    def _basic_check(self, text: str) -> bool:
        """Basic syntax check: balanced braces/parens and semicolons."""
        stack = []
        pairs = {")": "(", "]": "[", "}": "{"}
        for ch in text:
            if ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
        return len(stack) == 0


class SmilesValidator(StringValidator):
    """Validates SMILES strings."""

    placeholder = "C"

    def validate(self, text: str) -> bool:
        # Basic SMILES validation: check balanced brackets and valid chars
        valid_chars = set("CNOSPFBIHcnospb[]()=#:-+/\\@%.0123456789rl")
        text_clean = text.strip()
        if not text_clean:
            return False

        bracket_depth = 0
        paren_depth = 0
        for ch in text_clean:
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
            elif ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1

            if bracket_depth < 0 or paren_depth < 0:
                return False
            if ch not in valid_chars:
                return False

        return bracket_depth == 0 and paren_depth == 0
