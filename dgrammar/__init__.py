"""dGrammar: Grammar-constrained decoding for diffusion LLMs."""

__version__ = "0.1.0"

from dgrammar._core import GrammarChecker, MASK_TOKEN

__all__ = ["GrammarChecker", "MASK_TOKEN"]
