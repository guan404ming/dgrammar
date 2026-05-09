"""Dgrammar: Grammar-constrained decoding for diffusion LLMs."""

from dgrammar.checker import TokenChecker
from dgrammar.cfg_checker import CFGTokenChecker, build_cfg_checker
from dgrammar.generate import (
    TimingStats,
    autocomplete_ar,
    autocomplete_greedy,
    generate,
)

__version__ = "0.2.0"

__all__ = [
    "TokenChecker", "CFGTokenChecker", "build_cfg_checker",
    "TimingStats", "generate", "autocomplete_greedy", "autocomplete_ar",
]
