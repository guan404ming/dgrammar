# Probe Experiments

Probing classifiers on diffusion LLM hidden states to predict functional correctness.

## Scripts

| Script | Description |
|---|---|
| `modal_midstep_probe.py` | Mid-step probing (generation + feature extraction + probe training). Supports `--dataset` and `--model` flags, multi-GPU via `--chunks`. |
| `modal_early_exit_sim.py` | Early exit simulation. Uses per-step probe confidence to decide when to stop. CPU-only. |
| `modal_adaptive_compute_sim.py` | Adaptive compute simulation. Uses step-0 probe to classify easy/hard, allocates fewer steps to easy instances. CPU-only. |
| `modal_seed_rerank.py` | Seed reranking experiment (negative result). |

## Data (Modal volume `probe-results`)

All results organized as `{dataset}_{model}/`.

```
jsonschema_llada/
  chunk_off{0,34,...,238}.npz        # 8 chunks, 272 instances total
  midstep_probe_results.json
  early_exit_results.json
  adaptive_compute_results.json

jsonschema_dream/
  chunk_off{0,34,...,238}.npz        # 8 chunks, 272 instances total
  midstep_probe_results.json
  early_exit_results.json
  adaptive_compute_results.json

gsm8k_llada/
  chunk_off{0,165,...,1155}.npz      # 8 chunks, 1319 instances total
  midstep_probe_results.json
  early_exit_results.json
  adaptive_compute_results.json

gsm8k_dream/
  chunk_off{0,165,...,1155}.npz      # 8 chunks, 1319 instances total
  midstep_probe_results.json
  early_exit_results.json
  adaptive_compute_results.json

seed_rerank/
  phase1_off*.npz, phase1_off*_ids.json
  phase3_off*.json
  probe_params.npz, instance_ids.json
```

### Chunk npz format

Each `chunk_off{offset}.npz` contains:
- `labels`: (n_instances,) int array, 1=functional, 0=not
- `feat_s{step}_r{region}`: (n_instances, n_layers, hidden_dim) float32
  - Steps: 0, 1, 4, 16, 32, 64, 127
  - Regions: 0-3 (gen_length / 4 each, mean-pooled)

## Models

| Key | Model | Mask ID | Layers |
|---|---|---|---|
| `llada` | GSAI-ML/LLaDA-8B-Instruct | 126336 | 33 |
| `dream` | Dream-org/Dream-v0-Instruct-7B | 151666 | 29 |

## Datasets

| Key | Source | N instances | Gen length | Correctness check |
|---|---|---|---|---|
| `jsonschema` | eth-sri/json-mode-eval-extended | 272 | 256 | JSON parse + reference match |
| `gsm8k` | openai/gsm8k (test) | 1319 | 512 | Numeric answer match (####) |

## Key Results

### Baseline functional rates (seed=0, 128 steps)

| | jsonschema | gsm8k |
|---|---|---|
| LLaDA | 48.5% (132/272) | 66.3% (875/1319) |
| Dream | 46.0% (125/272) | 61.5% (811/1319) |

### Mid-step probe AUC (best layer, final step)

| | jsonschema | gsm8k |
|---|---|---|
| LLaDA | 0.809 (layer 24) | 0.786 (layer 24) |
| Dream | 0.828 (layer 5) | 0.818 (layer 19) |

### Early exit (threshold=0.80)

| | jsonschema | gsm8k |
|---|---|---|
| LLaDA | 98.4% saved, 73.5% acc | 61.3% saved, 73.5% acc |
| Dream | 96.7% saved, 72.4% acc | 36.0% saved, 73.5% acc |

### Adaptive compute (conf=0.75, easy=32 steps)

| | jsonschema | gsm8k |
|---|---|---|
| LLaDA | 30.1% saved, 72.5% easy precision | 33.5% saved, 80.0% easy precision |
| Dream | 23.2% saved, 65.5% easy precision | 22.3% saved, 76.3% easy precision |

Step-0 probe classifies instances as easy (P(func) >= threshold) or hard.
Easy instances get fewer denoising steps. "Easy precision" = fraction of
probe-predicted-easy instances that are actually functional.

### Seed reranking (negative result)

Train probe on seed=0 step-1 hidden states (layer 23, mean-pooled) to predict
functional@1. Score 5 seeds at step 1, pick best, run full 128-step denoising.

| | Baseline | Rerank | Probe train acc |
|---|---|---|---|
| LLaDA | 48.5% (132/272) | 48.5% (+0.0%) | 85.7% |
| Dream | 46.0% (125/272) | 46.0% (+0.0%) | 84.6% |

Probe learns instance difficulty, not seed quality.

## Usage

```bash
cd probe

# Mid-step probe (8x A100)
../.venv/bin/modal run modal_midstep_probe.py --dataset jsonschema --model llada --chunks 8

# Early exit simulation (CPU)
../.venv/bin/modal run modal_early_exit_sim.py --dataset gsm8k --model dream --chunks 8

# Adaptive compute simulation (CPU)
../.venv/bin/modal run modal_adaptive_compute_sim.py --dataset gsm8k --model llada --chunks 8
```
