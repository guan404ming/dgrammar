"""Tests for the grammar definition helpers."""

from dgrammar.grammar import json_schema_to_cfg, cpp_grammar, smiles_grammar
from dgrammar._core import GrammarChecker


class TestJsonSchemaToCfg:
    def test_simple_object(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        cfg = json_schema_to_cfg(schema)
        assert "start:" in cfg
        assert '"name"' in cfg or "name" in cfg

    def test_generates_parseable_grammar(self):
        schema = {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
            },
        }
        cfg = json_schema_to_cfg(schema)
        # Should not raise
        GrammarChecker(cfg)


class TestBuiltinGrammars:
    def test_cpp_grammar_loads(self):
        cfg = cpp_grammar()
        GrammarChecker(cfg)

    def test_smiles_grammar_loads(self):
        cfg = smiles_grammar()
        GrammarChecker(cfg)
