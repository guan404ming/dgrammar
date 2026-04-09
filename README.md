# Dgrammar

Grammar-constrained decoding for diffusion LLMs (LLaDA-8B-Instruct) via incremental token-level grammar checking, adaptive batch unmasking, frontier-only masking, and async CPU/GPU overlap.

## Results

JSON-Bench medium_test (511 instances), B200, T=128, seed=0.

| Run | schema@1 | mean | p95 | max | timeout |
|---|---:|---:|---:|---:|---:|
| Vanilla LLaDA | 51.3% | 3.91s | 5.68s | 9.64s | 0 |
| LAVE | 76.1% | 30.56s | 120s | 120.71s | 65 (12.7%) |
| Dgrammar r100 | 76.5% | 8.45s | 25.22s | 49.76s | 0 |
| **Dgrammar r500** | **81.0%** | **8.18s** | **14.58s** | 50.65s | **0** |

Dgrammar beats LAVE by 4.9 points schema@1, 3.7x lower mean latency, 8.2x lower p95, and zero timeouts vs LAVE's 12.7%.

## Key Files

| File | Description |
|---|---|
| `dgrammar/generate.py` | Core algorithm: async overlap + adaptive batch + frontier masking + AC4 autocomplete |
| `dgrammar/checker.py` | TokenChecker wrapper for llguidance LLMatcher |
| `bench/modal_bench.py` | Unified Modal launcher |
| `bench/runner/run_{dgrammar,lave,igcd,vanilla}.py` | Method runners |
| `bench/jsb_dataset.py` | JSONSchemaBench adapter |

## Setup

```bash
uv pip install -e .
uv pip install llguidance>=1.6
```

## Running

```bash
cd bench

# Dgrammar main config (B200, r500)
modal run modal_bench.py --method dgrammar --total 586 --chunks 8 \
    --dataset jsb_medium_test --max-resamples 500

# Baselines
modal run modal_bench.py --method lave    --total 586 --chunks 8 --dataset jsb_medium_test
modal run modal_bench.py --method vanilla --total 586 --chunks 8 --dataset jsb_medium_test
modal run modal_bench.py --method igcd    --total 272 --chunks 4
```

Each invocation drops its chunks into a fresh timestamped folder:
`results/{method}/{YYYYMMDD_HHMMSS}[_{tag}]/{dataset}_s{seed}_t{steps}[_off{K}].jsonl`
