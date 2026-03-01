"""Lexical mapping from BPE vocabulary to grammar terminals.

Maps model tokenizer vocabulary to abstract grammar terminals so the
Earley parser can work at the BPE token level. This is the approach
used by both LAVE and ETH (Mundler et al.) for token-level grammar checking.

For each BPE token, determines which grammar terminal(s) it could represent.
"""

from __future__ import annotations

import json
import re
from typing import Any


class LexicalMap:
    """Maps BPE token IDs to grammar terminal names.

    Each BPE token can map to zero or more grammar terminals.
    Tokens that don't map to any terminal are "unconstrained" and
    treated as wildcards during grammar checking.
    """

    def __init__(self):
        # token_id -> set of grammar terminal names
        self._id_to_terminals: dict[int, set[str]] = {}
        # terminal_name -> set of token_ids
        self._terminal_to_ids: dict[str, set[int]] = {}

    def add_mapping(self, token_id: int, terminal: str):
        self._id_to_terminals.setdefault(token_id, set()).add(terminal)
        self._terminal_to_ids.setdefault(terminal, set()).add(token_id)

    def terminals_for(self, token_id: int) -> set[str]:
        return self._id_to_terminals.get(token_id, set())

    def ids_for(self, terminal: str) -> set[int]:
        return self._terminal_to_ids.get(terminal, set())

    def has_mapping(self, token_id: int) -> bool:
        return token_id in self._id_to_terminals

    def all_terminals(self) -> set[str]:
        return set(self._terminal_to_ids.keys())

    def token_ids_to_terminal_seq(
        self, token_ids: list[int], mask_id: int,
    ) -> list[str | None]:
        """Convert a sequence of BPE token IDs to grammar terminals.

        Returns a list where each element is:
        - A terminal name (if the token maps to exactly one terminal)
        - "[MASK]" if the token is a mask
        - None if the token has no mapping or ambiguous mapping
        """
        result = []
        for tid in token_ids:
            if tid == mask_id:
                result.append("[MASK]")
            else:
                terminals = self.terminals_for(tid)
                if len(terminals) == 1:
                    result.append(next(iter(terminals)))
                elif len(terminals) == 0:
                    result.append(None)
                else:
                    # Ambiguous, return first (could be improved)
                    result.append(next(iter(terminals)))
        return result


def build_json_lexmap(tokenizer: Any) -> LexicalMap:
    """Build a lexical map for JSON grammar from a tokenizer's vocabulary.

    Categorizes each BPE token into JSON grammar terminals:
    - LBRACE, RBRACE, LBRACKET, RBRACKET, COLON, COMMA
    - STRING (quoted strings), NUMBER, TRUE, FALSE, NULL
    - WHITESPACE (spaces, newlines, tabs)
    """
    lmap = LexicalMap()
    vocab = tokenizer.get_vocab()

    for token_str, token_id in vocab.items():
        # Decode the token to get its actual string representation
        decoded = tokenizer.decode([token_id]).strip()

        if not decoded:
            continue

        # Exact structural tokens
        if decoded == "{":
            lmap.add_mapping(token_id, "LBRACE")
        elif decoded == "}":
            lmap.add_mapping(token_id, "RBRACE")
        elif decoded == "[":
            lmap.add_mapping(token_id, "LBRACKET")
        elif decoded == "]":
            lmap.add_mapping(token_id, "RBRACKET")
        elif decoded == ":":
            lmap.add_mapping(token_id, "COLON")
        elif decoded == ",":
            lmap.add_mapping(token_id, "COMMA")
        elif decoded == "true":
            lmap.add_mapping(token_id, "TRUE")
        elif decoded == "false":
            lmap.add_mapping(token_id, "FALSE")
        elif decoded == "null":
            lmap.add_mapping(token_id, "NULL")

        # Strings: tokens that start with " or are quoted
        if decoded.startswith('"') and decoded.endswith('"') and len(decoded) >= 2:
            lmap.add_mapping(token_id, "STRING")
        elif decoded == '"':
            lmap.add_mapping(token_id, "QUOTE")

        # Numbers
        if _is_number_token(decoded):
            lmap.add_mapping(token_id, "NUMBER")

        # Whitespace
        if decoded.strip() == "" and decoded:
            lmap.add_mapping(token_id, "WS")

        # Tokens that could be part of a string value (letters, words)
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', decoded):
            lmap.add_mapping(token_id, "WORD")

        # Newline
        if "\n" in decoded:
            lmap.add_mapping(token_id, "WS")

        # Markdown fence (```) - model often wraps JSON in these
        if decoded.startswith("```"):
            lmap.add_mapping(token_id, "FENCE")

    return lmap


def _is_number_token(s: str) -> bool:
    """Check if a string looks like a JSON number (or part of one)."""
    s = s.strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


# JSON grammar at the terminal level (for Earley parser)
JSON_TOKEN_GRAMMAR = """
start: Value
Value -> Object | Array | STRING | QuotedStr | NUMBER | TRUE | FALSE | NULL
Object -> LBRACE RBRACE | LBRACE Members RBRACE
Members -> Pair | Pair COMMA Members
Pair -> Key COLON Value
Key -> STRING | QuotedStr
QuotedStr -> QUOTE WORD QUOTE | QUOTE QUOTE
Array -> LBRACKET RBRACKET | LBRACKET Elements RBRACKET
Elements -> Value | Value COMMA Elements
"""

# Relaxed JSON grammar that allows whitespace and markdown fences
JSON_TOKEN_GRAMMAR_RELAXED = """
start: Doc
Doc -> Value | Fenced
Fenced -> FENCE WS Value WS FENCE | FENCE Value FENCE
Value -> Object | Array | STRING | QuotedStr | NUMBER | TRUE | FALSE | NULL
Object -> LBRACE RBRACE | LBRACE WS RBRACE | LBRACE Members RBRACE | LBRACE WS Members WS RBRACE
Members -> Pair | Pair COMMA Members | Pair COMMA WS Members | Pair WS COMMA WS Members
Pair -> Key COLON Value | Key COLON WS Value | Key WS COLON WS Value
Key -> STRING | QuotedStr
QuotedStr -> QUOTE WORD QUOTE | QUOTE QUOTE | QUOTE NUMBER QUOTE
Array -> LBRACKET RBRACKET | LBRACKET Elements RBRACKET | LBRACKET WS Elements WS RBRACKET
Elements -> Value | Value COMMA Elements | Value COMMA WS Elements
"""
