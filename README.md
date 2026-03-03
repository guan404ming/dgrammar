# AdaGram: Adaptive Grammar-Constrained Decoding for Diffusion Language Models

Constrained decoding for discrete diffusion LLMs (LLaDA-8B-Instruct) using incremental grammar checking with adaptive batch unmasking and selective remasking.

## Benchmark Results

All experiments: JSON-Bench (272 instances), LLaDA-8B-Instruct, T=0.2, gen_length=256, seed=0.

### Main Comparison (T=128, A100)

| Method | Syntactic | Functional | Mean Time | Median | P95 | Max | Constraint % |
|---|---|---|---|---|---|---|---|
| NO-CD | 85.4% | 46.9% | 9.48s | - | - | - | 0% |
| IG-CD | 94.9% | 52.1% | 8.89s | 8.55s | 13.51s | 26.22s | 18.1% |
| LAVE | 98.9% | 55.0% | 7.60s | 4.45s | 23.59s | 140.22s | 8.5% |
| AdaGram v1 (rustformlang) | 95.6% | 51.8% | 8.43s | 8.36s | 11.58s | 18.94s | 12.4% |
| AdaGram v2 (llguidance) | 97.0% | - | 8.31s | 8.50s | 9.92s | 10.77s | 21.1% |
| AdaGram v2 + async | 97.0% | - | 8.02s | 8.22s | 9.81s | 10.56s | 12.1% |
| **AdaGram v2 + async + AC4** | **100%** | - | **8.15s** | **8.32s** | **9.63s** | **14.23s** | **11.9%** |

- NO-CD, IG-CD, LAVE functional numbers from LAVE paper (arXiv:2602.00612).
- IG-CD and AdaGram v1 timed results on 136/272 instances (first two chunks).
- AdaGram v2 variants on 270/272 instances (2 schemas unsupported by llguidance).
- LAVE on 272/272 instances (3 failed: jsonschema_15, 16, 17).

### Semi-Autoregressive vs Full Parallel (AdaGram v2 + async + AC4)

| Setting | Valid | Mean Time | Median | P95 | Max | Constraint % |
|---|---|---|---|---|---|---|
| block=32 (semi-AR, 8 blocks) | 100% | 8.15s | 8.32s | 9.63s | 14.23s | 11.9% |
| block=256 (full parallel, 1 block) | 100% | 8.71s | 9.47s | 10.60s | 13.91s | 26.0% |

Full parallel increases constraint overhead 2.2x (26% vs 12%) because compute_mask runs 8x more frequently relative to blocks. Semi-AR is faster overall.

### Latency Distribution

AdaGram is significantly more stable than LAVE:

| Metric | AdaGram (block=32) | LAVE |
|---|---|---|
| Mean | 8.15s | 7.60s |
| Median | 8.32s | 4.45s |
| P95 | 9.63s | 23.59s |
| Max | 14.23s | 140.22s |
| P95/Median ratio | 1.16x | 5.30x |
| Failures | 0/270 | 3/272 |

LAVE is bimodal: 76% of instances finish with near-zero constraint overhead, but 24% trigger AR fallback with up to 1666 retries. AdaGram's latency is tightly concentrated.

### Ablation: Feature Contributions

| Feature | Valid % | Mean Time | Constraint % |
|---|---|---|---|
| v2 baseline (llguidance) | 97.0% | 8.31s | 21.1% |
| + async overlap | 97.0% | 8.02s (-3.5%) | 12.1% |
| + autocomplete (AC4) | 100% (+3pp) | 8.15s | 11.9% |

- **Async overlap**: compute_mask (CPU) runs in parallel with model forward (GPU), hiding ~43% of constraint overhead.
- **Autocomplete (AC4)**: grammar-guided greedy completion with logit refresh every 8 steps. Brings valid rate from 97% to 100%.

### Per-Operation Timing (AdaGram v2 + async + AC4, block=32)

| Operation | Mean per call | Calls per instance | Total per instance |
|---|---|---|---|
| Model forward | 38.3ms | 120 | 4,596ms |
| Grammar check (try_consume) | 0.003ms | ~150 | 0.45ms |
| compute_mask (frontier) | 2.5ms | ~8 | 20ms |
| Mask wait (after async overlap) | 0.1ms | ~8 | 0.8ms |
| Token selection | 0.3ms | ~120 | 36ms |

Grammar checking via llguidance is ~9000x faster than v1's rustformlang DFA intersection (0.003ms vs 27ms per check).

## Method Overview

AdaGram operates as a post-hoc wrapper around the diffusion model's denoising loop:

1. **Adaptive batch unmasking**: unmask 1-8 tokens per inner step. Batch size doubles on success, resets to 1 on violation.
2. **Incremental prefix checking**: after placing tokens, extend the validated prefix left-to-right using `try_consume_tokens`. Returns the exact violator position.
3. **Selective remasking**: on violation, remask only the violator token. Try next-best tokens from the same logits. Validated prefix tokens are never disturbed.
4. **Frontier masking**: at the consume frontier (first unvalidated position), apply `compute_mask` to guarantee the placed token is grammar-valid.
5. **Async CPU/GPU overlap**: `compute_mask` runs on a CPU thread while the GPU forward pass executes.
6. **Autocomplete fallback (AC4)**: after diffusion steps complete, if masks remain, do grammar-guided greedy completion. Refresh logits every 8 steps to avoid staleness.

## Key Files

| File | Description |
|---|---|
| `dgrammar/generate_v2.py` | Core algorithm: adaptive batch + selective remasking + prefix checking |
| `dgrammar/checker.py` | TokenChecker wrapper for llguidance LLMatcher |
| `run_v2_async_timed.py` | Full pipeline: async overlap + AC4 + timing instrumentation |
| `dgrammar/generate_dgrammar.py` | v1 algorithm (rustformlang DFA intersection) |
| `run_lave_timed.py` | LAVE baseline runner with timing instrumentation |
| `modal_v2_async_timed.py` | Modal deployment for AdaGram (8x A100) |
| `modal_lave_bench.py` | Modal deployment for LAVE baseline |
| `eval_dgrammar.py` | ETH functional evaluation (JSON schema checker) |

## Setup

```bash
uv pip install -e .
# llguidance for token-level grammar checking
uv pip install llguidance>=1.6
# For v1 (rustformlang backend)
cd vendor/constrained-diffusion/rustformlang_bindings && maturin develop --release
```

## Running

```bash
# AdaGram v2 + async + AC4 on Modal (8x A100, semi-AR)
modal run modal_v2_async_timed.py --chunks 8 --block-ar 1

# Full parallel variant
modal run modal_v2_async_timed.py --chunks 8 --block-ar 0

# LAVE baseline on Modal
modal run modal_lave_bench.py --chunks 8
```
