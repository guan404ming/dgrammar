"""Evaluation harness for Dgrammar experiments.

Computes syntactic@k, throughput, and other metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dgrammar._core import GrammarChecker
from dgrammar.decode import DecodeConfig, DecodeStats, decode


@dataclass
class EvalResult:
    """Result for a single benchmark problem."""
    problem_id: str
    prompt: str
    output: str
    is_syntactically_valid: bool
    stats: DecodeStats
    elapsed_seconds: float


@dataclass
class BenchmarkResult:
    """Aggregated results for a benchmark suite."""
    benchmark_name: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def syntactic_at_1(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.is_syntactically_valid) / len(self.results)

    @property
    def avg_steps(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.stats.total_steps for r in self.results) / len(self.results)

    @property
    def avg_throughput(self) -> float:
        """Average tokens per second."""
        total_tokens = sum(
            sum(r.stats.tokens_per_step) for r in self.results
        )
        total_time = sum(r.elapsed_seconds for r in self.results)
        if total_time == 0:
            return 0.0
        return total_tokens / total_time

    @property
    def degradation_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(
            1 for r in self.results if r.stats.degraded_to_fallback
        ) / len(self.results)

    def summary(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "n_problems": len(self.results),
            "syntactic@1": round(self.syntactic_at_1, 4),
            "avg_steps": round(self.avg_steps, 2),
            "avg_throughput_tps": round(self.avg_throughput, 2),
            "degradation_rate": round(self.degradation_rate, 4),
        }


def syntactic_at_k(results: list[list[EvalResult]], k: int) -> float:
    """Compute syntactic@k across multiple samples per problem.

    Args:
        results: List of problems, each with a list of k samples.
        k: Number of samples to consider.

    Returns:
        Fraction of problems where at least one of k samples is valid.
    """
    if not results:
        return 0.0
    count = 0
    for samples in results:
        top_k = samples[:k]
        if any(s.is_syntactically_valid for s in top_k):
            count += 1
    return count / len(results)


def run_benchmark(
    model: Any,
    tokenizer: Any,
    problems: list[dict[str, str]],
    grammar_checker: GrammarChecker,
    config: DecodeConfig | None = None,
    benchmark_name: str = "unnamed",
    validator: Any | None = None,
    device: str = "cuda",
) -> BenchmarkResult:
    """Run a full benchmark suite.

    Args:
        model: The dLLM model.
        tokenizer: The tokenizer.
        problems: List of dicts with "id" and "prompt" keys.
        grammar_checker: Grammar checker for the target language.
        config: Decode configuration.
        benchmark_name: Name for reporting.
        validator: Optional external validator (e.g., jsonschema, tree-sitter).
            Should accept a string and return bool.
        device: Torch device.

    Returns:
        BenchmarkResult with per-problem results.
    """
    bench = BenchmarkResult(benchmark_name=benchmark_name)

    for problem in problems:
        pid = problem["id"]
        prompt = problem["prompt"]

        start_time = time.time()
        output, stats = decode(
            model, tokenizer, prompt, grammar_checker,
            config=config, device=device,
        )
        elapsed = time.time() - start_time

        # Check syntactic validity
        tokens = output.split()
        is_valid = grammar_checker.check(tokens)

        # Use external validator if provided
        if validator is not None:
            try:
                is_valid = is_valid and validator(output)
            except Exception:
                is_valid = False

        result = EvalResult(
            problem_id=pid,
            prompt=prompt,
            output=output,
            is_syntactically_valid=is_valid,
            stats=stats,
            elapsed_seconds=elapsed,
        )
        bench.results.append(result)

    return bench


def save_results(bench: BenchmarkResult, path: str | Path) -> None:
    """Save benchmark results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": bench.summary(),
        "results": [
            {
                "problem_id": r.problem_id,
                "output": r.output,
                "is_valid": r.is_syntactically_valid,
                "elapsed_seconds": round(r.elapsed_seconds, 4),
                "total_steps": r.stats.total_steps,
                "total_violations": r.stats.total_violations,
                "degraded": r.stats.degraded_to_fallback,
            }
            for r in bench.results
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
