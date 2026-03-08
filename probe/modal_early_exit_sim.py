"""Early exit simulation using mid-step probe confidence.

Loads chunk features from Modal volume, trains per-step probes via 5-fold CV,
simulates early stopping when probe confidence exceeds a threshold.

Reports compute savings vs accuracy tradeoff.

Usage:
  cd probe
  ../.venv/bin/modal run modal_early_exit_sim.py --dataset jsonschema --model llada
  ../.venv/bin/modal run modal_early_exit_sim.py --dataset gsm8k --model dream
"""

import modal

app = modal.App("probe-early-exit")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "scikit-learn")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)

STEPS = 128
CHECKPOINT_STEPS = sorted([0, 1, 4, 16, 32, 64, STEPS - 1])
N_REGIONS = 4


@app.function(
    image=image,
    timeout=1800,
    volumes={"/results": RESULTS_VOL},
)
def run_early_exit_sim(dataset_key: str, model_key: str, n_chunks: int, total: int):
    import json
    import os

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    RESULTS_VOL.reload()

    gen_lengths = {"jsonschema": 256, "gsm8k": 512}
    gen_length = gen_lengths[dataset_key]
    region_size = gen_length // N_REGIONS
    chunk_size = (total + n_chunks - 1) // n_chunks
    in_dir = f"/results/{dataset_key}_{model_key}"

    # Load all chunks
    all_labels = []
    all_feats = {}

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_labels.append(data["labels"])
        for s in CHECKPOINT_STEPS:
            for r in range(N_REGIONS):
                key = (s, r)
                if key not in all_feats:
                    all_feats[key] = []
                all_feats[key].append(data[f"feat_s{s}_r{r}"])
        print(f"  Loaded {path}: {len(data['labels'])} samples")

    labels = np.concatenate(all_labels)
    features = {}
    for s in CHECKPOINT_STEPS:
        features[s] = {}
        for r in range(N_REGIONS):
            features[s][r] = np.concatenate(all_feats[(s, r)])

    n_samples = len(labels)
    n_func = int(labels.sum())
    n_layers = features[CHECKPOINT_STEPS[0]][0].shape[1]
    print(f"\nLoaded: {n_samples} samples, {n_func} functional "
          f"({100*n_func/n_samples:.1f}%), {n_layers} layers")

    # Find best layer from final step (all regions pooled)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_step = CHECKPOINT_STEPS[-1]

    best_auc = -1
    best_layer = 0
    for layer_idx in range(n_layers):
        X = np.mean([features[final_step][r][:, layer_idx, :]
                      for r in range(N_REGIONS)], axis=0)
        aucs = []
        for train_idx, test_idx in skf.split(X, labels):
            clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=min(64, X.shape[1])),
                LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
            )
            clf.fit(X[train_idx], labels[train_idx])
            prob = clf.predict_proba(X[test_idx])[:, 1]
            try:
                aucs.append(roc_auc_score(labels[test_idx], prob))
            except ValueError:
                aucs.append(0.5)
        mean_auc = np.mean(aucs)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_layer = layer_idx

    print(f"Best layer: {best_layer} (AUC={best_auc:.3f} at final step)")

    # Train per-step probes with CV, collect per-instance probabilities
    # Use leave-one-fold-out: for each instance, the probability comes from
    # a model that didn't see it during training.
    step_probs = {}  # step -> (n_samples,) array of P(functional)
    step_aucs = {}

    for s in CHECKPOINT_STEPS:
        X = np.mean([features[s][r][:, best_layer, :]
                      for r in range(N_REGIONS)], axis=0)
        probs = np.zeros(n_samples)
        aucs = []

        for train_idx, test_idx in skf.split(X, labels):
            clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=min(64, X.shape[1])),
                LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
            )
            clf.fit(X[train_idx], labels[train_idx])
            prob = clf.predict_proba(X[test_idx])[:, 1]
            probs[test_idx] = prob
            try:
                aucs.append(roc_auc_score(labels[test_idx], prob))
            except ValueError:
                aucs.append(0.5)

        step_probs[s] = probs
        step_aucs[s] = np.mean(aucs)
        print(f"  Step {s:>3}: AUC={step_aucs[s]:.3f}")

    # Early exit simulation
    # Strategy: at each checkpoint step (in order), if probe confidence
    # exceeds threshold, stop and use the probe's prediction.
    # "Confidence" = max(P(func), 1-P(func)), i.e. how sure the probe is.
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # Compute forward passes at each checkpoint
    # step 0 = 1 forward pass, step 1 = 2, step 4 = 5, etc.
    # For block-based (LLaDA): total STEPS forward passes for full generation
    # For simplicity: compute savings as fraction of steps completed
    step_cost = {s: (s + 1) / STEPS for s in CHECKPOINT_STEPS}

    print(f"\n{'='*80}")
    print(f"Early Exit Simulation (layer={best_layer})")
    print(f"{'='*80}")
    print(f"Baseline: {n_func}/{n_samples} functional "
          f"({100*n_func/n_samples:.1f}%), 100% compute")
    print()

    # Also compute: what if we just predict at the final step (no early exit)?
    final_preds = (step_probs[final_step] >= 0.5).astype(int)
    final_acc = (final_preds == labels).mean()
    print(f"Final-step probe accuracy: {100*final_acc:.1f}%")
    print()

    header = f"{'Threshold':>10} | {'Exited Early':>12} | {'Avg Compute':>11} | {'Saved':>6} | {'Accuracy':>8} | {'Func Kept':>9}"
    print(header)
    print("-" * len(header))

    results_table = []

    for thresh in thresholds:
        exit_step = np.full(n_samples, -1, dtype=int)  # -1 = no early exit
        predictions = np.full(n_samples, -1, dtype=int)

        for idx in range(n_samples):
            for s in CHECKPOINT_STEPS:
                p = step_probs[s][idx]
                conf = max(p, 1 - p)
                if conf >= thresh:
                    exit_step[idx] = s
                    predictions[idx] = 1 if p >= 0.5 else 0
                    break

            # If no early exit, use final step prediction
            if exit_step[idx] == -1:
                exit_step[idx] = final_step
                predictions[idx] = 1 if step_probs[final_step][idx] >= 0.5 else 0

        # Compute metrics
        n_early = (exit_step < final_step).sum()
        avg_cost = np.mean([(s + 1) / STEPS for s in exit_step])
        compute_saved = 1 - avg_cost
        accuracy = (predictions == labels).mean()

        # "Functional kept": among truly functional instances, how many
        # were correctly predicted as functional?
        func_mask = labels == 1
        func_kept = (predictions[func_mask] == 1).mean() if func_mask.sum() > 0 else 0

        row = {
            "threshold": thresh,
            "n_early_exit": int(n_early),
            "pct_early_exit": round(100 * n_early / n_samples, 1),
            "avg_compute_pct": round(100 * avg_cost, 1),
            "compute_saved_pct": round(100 * compute_saved, 1),
            "accuracy": round(100 * accuracy, 1),
            "func_recall": round(100 * func_kept, 1),
        }
        results_table.append(row)

        print(f"{thresh:>10.2f} | {n_early:>5} ({100*n_early/n_samples:>5.1f}%) | "
              f"{100*avg_cost:>9.1f}% | {100*compute_saved:>5.1f}% | "
              f"{100*accuracy:>6.1f}%  | {100*func_kept:>7.1f}%")

    # Per-step exit distribution for a mid-range threshold
    print(f"\n--- Exit step distribution (threshold=0.75) ---")
    thresh = 0.75
    exit_counts = {s: 0 for s in CHECKPOINT_STEPS}
    for idx in range(n_samples):
        for s in CHECKPOINT_STEPS:
            p = step_probs[s][idx]
            conf = max(p, 1 - p)
            if conf >= thresh:
                exit_counts[s] += 1
                break
        else:
            exit_counts[final_step] += 1

    for s in CHECKPOINT_STEPS:
        bar = "#" * (exit_counts[s] * 40 // max(1, max(exit_counts.values())))
        print(f"  Step {s:>3}: {exit_counts[s]:>5} ({100*exit_counts[s]/n_samples:>5.1f}%) {bar}")

    # Save results
    results = {
        "dataset": dataset_key,
        "model": model_key,
        "n_samples": n_samples,
        "n_functional": n_func,
        "best_layer": best_layer,
        "best_layer_auc": round(best_auc, 4),
        "step_aucs": {str(s): round(a, 4) for s, a in step_aucs.items()},
        "final_step_probe_accuracy": round(float(final_acc), 4),
        "early_exit_results": results_table,
    }

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/early_exit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()

    print(f"\nResults saved to {out_path}")
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(
    dataset: str = "jsonschema",
    model: str = "llada",
    chunks: int = 8,
    total: int = 0,
):
    totals = {"jsonschema": 272, "gsm8k": 1319}
    if total <= 0:
        total = totals[dataset]
    print(f"Early exit simulation: dataset={dataset}, model={model}, total={total}")
    result = run_early_exit_sim.remote(dataset, model, chunks, total)
    print("\n" + result)
