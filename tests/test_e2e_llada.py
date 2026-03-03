"""End-to-end test with LLaDA-8B-Instruct (cached model).

Tests the full Dgrammar pipeline with a real diffusion LLM:
1. Load LLaDA from HuggingFace cache
2. Run Dgrammar decode with JSON grammar
3. Verify output validity
4. Compare Dgrammar vs No-CD vs FS-CD (Mundler/ETH) vs LAVE baselines

Metrics align with LAVE paper (arxiv:2602.00612) and ETH paper (arxiv:2508.10111):
- syntactic@1: fraction of outputs that are syntactically valid
- total_steps: denoising steps to reach output
- throughput: tokens per second

Usage:
    pytest tests/test_e2e_llada.py -v -s
    # Or run directly:
    python tests/test_e2e_llada.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

# Skip if no GPU or model not cached
LLADA_CACHE_PATH = Path.home() / ".cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct"
HAS_GPU = torch.cuda.is_available()
HAS_MODEL = LLADA_CACHE_PATH.exists()

requires_gpu_and_model = pytest.mark.skipif(
    not (HAS_GPU and HAS_MODEL),
    reason="Requires GPU and cached LLaDA model",
)


# --- Test problems ---

JSON_PROBLEMS = [
    {
        "id": "json_simple_1",
        "prompt": "Extract the name and age from: John is 30 years old.\nOutput JSON:\n",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        },
    },
    {
        "id": "json_bool_1",
        "prompt": "Is the sky blue? Answer in JSON with a field 'answer' (boolean).\nOutput JSON:\n",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "boolean"}},
            "required": ["answer"],
        },
    },
]


@requires_gpu_and_model
class TestLLaDAE2E:
    """E2E tests with real LLaDA model."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        from dgrammar.models import load_llada
        model, tokenizer = load_llada()
        return model, tokenizer

    def test_forward_pass(self, model_and_tokenizer):
        """Verify LLaDA forward pass returns logits."""
        model, tokenizer = model_and_tokenizer
        assert tokenizer.mask_token_id == 126336

        prompt = "Hello"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        masks = torch.full((1, 8), tokenizer.mask_token_id, dtype=torch.long, device="cuda")
        x = torch.cat([input_ids, masks], dim=1)

        output = model(x)
        assert hasattr(output, "logits")
        assert output.logits.shape[0] == 1
        assert output.logits.shape[1] == x.shape[1]
        print(f"\nForward pass OK: logits shape = {output.logits.shape}")

    def test_dgrammar_json_generation(self, model_and_tokenizer):
        """Test Dgrammar decode on JSON generation."""
        from dgrammar.decode import DecodeConfig, decode
        from dgrammar.validators import StringLevelChecker, JsonValidator

        model, tokenizer = model_and_tokenizer
        checker = StringLevelChecker(JsonValidator())

        config = DecodeConfig(
            output_length=32,
            total_steps=32,
            max_unmask_per_step=4,
            confidence_threshold=0.3,
            max_remask_attempts=3,
            enable_fallback=True,
        )

        problem = JSON_PROBLEMS[0]
        start = time.time()
        output, stats = decode(
            model, tokenizer, problem["prompt"], checker, config, device="cuda",
        )
        elapsed = time.time() - start

        print(f"\n--- Dgrammar JSON Test ---")
        print(f"Output: {output}")
        print(f"Steps: {stats.total_steps}, Violations: {stats.total_violations}")
        print(f"Remasks: {stats.total_remasks}, Fallback: {stats.degraded_to_fallback}")
        print(f"Time: {elapsed:.2f}s")

        try:
            parsed = json.loads(output)
            print(f"Valid JSON: {parsed}")
        except json.JSONDecodeError:
            print(f"Not valid JSON")

    def test_no_cd_baseline(self, model_and_tokenizer):
        """Test unconstrained decoding (No-CD)."""
        from dgrammar.baselines import BaselineConfig, decode_no_cd

        model, tokenizer = model_and_tokenizer
        config = BaselineConfig(output_length=32, total_steps=32)

        problem = JSON_PROBLEMS[0]
        start = time.time()
        output, stats = decode_no_cd(model, tokenizer, problem["prompt"], config, device="cuda")
        elapsed = time.time() - start

        print(f"\n--- No-CD ---")
        print(f"Output: {output}")
        print(f"Steps: {stats.total_steps}, Time: {elapsed:.2f}s")
        try:
            json.loads(output)
            print(f"syntactic@1: PASS")
        except json.JSONDecodeError:
            print(f"syntactic@1: FAIL")

    def test_fscd_baseline(self, model_and_tokenizer):
        """Test FS-CD (Mundler/ETH) baseline."""
        from dgrammar.baselines import BaselineConfig, decode_fscd
        from dgrammar.validators import StringLevelChecker, JsonValidator

        model, tokenizer = model_and_tokenizer
        checker = StringLevelChecker(JsonValidator())
        config = BaselineConfig(output_length=32, total_steps=32, max_rejections=5)

        problem = JSON_PROBLEMS[0]
        start = time.time()
        output, stats = decode_fscd(
            model, tokenizer, problem["prompt"], checker, config, device="cuda",
        )
        elapsed = time.time() - start

        print(f"\n--- FS-CD (Mundler/ETH) ---")
        print(f"Output: {output}")
        print(f"Steps: {stats.total_steps}, Violations: {stats.total_violations}")
        print(f"Time: {elapsed:.2f}s")
        try:
            json.loads(output)
            print(f"syntactic@1: PASS")
        except json.JSONDecodeError:
            print(f"syntactic@1: FAIL")

    def test_lave_baseline(self, model_and_tokenizer):
        """Test LAVE baseline."""
        from dgrammar.baselines import BaselineConfig, decode_lave
        from dgrammar.validators import StringLevelChecker, JsonValidator

        model, tokenizer = model_and_tokenizer
        checker = StringLevelChecker(JsonValidator())
        config = BaselineConfig(output_length=32, total_steps=32, lookahead_k=3)

        problem = JSON_PROBLEMS[0]
        start = time.time()
        output, stats = decode_lave(
            model, tokenizer, problem["prompt"], checker, config, device="cuda",
        )
        elapsed = time.time() - start

        print(f"\n--- LAVE ---")
        print(f"Output: {output}")
        print(f"Steps: {stats.total_steps}, Violations: {stats.total_violations}")
        print(f"Time: {elapsed:.2f}s")
        try:
            json.loads(output)
            print(f"syntactic@1: PASS")
        except json.JSONDecodeError:
            print(f"syntactic@1: FAIL")


def run_comparison():
    """Run a full comparison. Execute directly: python tests/test_e2e_llada.py"""
    if not HAS_GPU:
        print("No GPU available. Skipping.")
        return
    if not HAS_MODEL:
        print(f"Model not cached. Skipping.")
        return

    from dgrammar.models import load_llada
    from dgrammar.decode import DecodeConfig, decode
    from dgrammar.baselines import BaselineConfig, decode_no_cd, decode_fscd, decode_lave
    from dgrammar.validators import StringLevelChecker, JsonValidator

    print("Loading LLaDA-8B-Instruct...")
    model, tokenizer = load_llada()
    checker = StringLevelChecker(JsonValidator())
    print(f"mask_token_id: {tokenizer.mask_token_id}\n")

    prompt = JSON_PROBLEMS[0]["prompt"]
    results = {}

    # No-CD
    print("=== No-CD (unconstrained) ===")
    bc = BaselineConfig(output_length=32, total_steps=32)
    t0 = time.time()
    out, stats = decode_no_cd(model, tokenizer, prompt, bc, device="cuda")
    t1 = time.time()
    valid = _is_json(out)
    results["No-CD"] = {"valid": valid, "time": t1 - t0, "steps": stats.total_steps}
    print(f"  Output: {out}")
    print(f"  Valid: {valid}, Time: {t1-t0:.2f}s, Steps: {stats.total_steps}")

    # FS-CD
    print("\n=== FS-CD (Mundler/ETH) ===")
    bc = BaselineConfig(output_length=32, total_steps=32, max_rejections=5)
    t0 = time.time()
    out, stats = decode_fscd(model, tokenizer, prompt, checker, bc, device="cuda")
    t1 = time.time()
    valid = _is_json(out)
    results["FS-CD"] = {"valid": valid, "time": t1 - t0, "steps": stats.total_steps, "violations": stats.total_violations}
    print(f"  Output: {out}")
    print(f"  Valid: {valid}, Time: {t1-t0:.2f}s, Steps: {stats.total_steps}")

    # LAVE
    print("\n=== LAVE ===")
    bc = BaselineConfig(output_length=32, total_steps=32, lookahead_k=3)
    t0 = time.time()
    out, stats = decode_lave(model, tokenizer, prompt, checker, bc, device="cuda")
    t1 = time.time()
    valid = _is_json(out)
    results["LAVE"] = {"valid": valid, "time": t1 - t0, "steps": stats.total_steps, "violations": stats.total_violations}
    print(f"  Output: {out}")
    print(f"  Valid: {valid}, Time: {t1-t0:.2f}s, Steps: {stats.total_steps}")

    # Dgrammar
    print("\n=== Dgrammar (ours) ===")
    dc = DecodeConfig(
        output_length=32, total_steps=32,
        max_unmask_per_step=4, confidence_threshold=0.3,
        max_remask_attempts=3, enable_fallback=True,
    )
    t0 = time.time()
    out, stats = decode(model, tokenizer, prompt, checker, dc, device="cuda")
    t1 = time.time()
    valid = _is_json(out)
    results["Dgrammar"] = {
        "valid": valid, "time": t1 - t0, "steps": stats.total_steps,
        "violations": stats.total_violations, "remasks": stats.total_remasks,
        "fallback": stats.degraded_to_fallback,
    }
    print(f"  Output: {out}")
    print(f"  Valid: {valid}, Time: {t1-t0:.2f}s, Steps: {stats.total_steps}")
    print(f"  Violations: {stats.total_violations}, Remasks: {stats.total_remasks}")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Method':<15} {'Valid':<8} {'Steps':<8} {'Time':<10}")
    print("-" * 41)
    for method, r in results.items():
        print(f"{method:<15} {str(r['valid']):<8} {r['steps']:<8} {r['time']:.2f}s")


def _is_json(text: str) -> bool:
    """Check if text is valid JSON, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


if __name__ == "__main__":
    run_comparison()
