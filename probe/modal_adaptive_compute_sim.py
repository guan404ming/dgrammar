"""Adaptive compute simulation using step-0 probe.

Uses step-0 probe to classify instances as easy/hard, then simulates
allocating fewer steps to easy instances. Reports compute savings vs
accuracy tradeoff.

Key difference from early_exit: here we still generate output, just with
fewer denoising steps for easy instances. The probe decides the step budget,
not the final answer.

Usage:
  cd probe
  ../.venv/bin/modal run modal_adaptive_compute_sim.py --dataset jsonschema --model llada
  ../.venv/bin/modal run modal_adaptive_compute_sim.py --dataset gsm8k --model dream
"""

import modal

app = modal.App("probe-adaptive-compute")

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
def run_adaptive_sim(dataset_key: str, model_key: str, n_chunks: int, total: int):
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

    # Find best layer from final step
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

    print(f"Best layer: {best_layer} (AUC={best_auc:.3f})")

    # Get step-0 probe probabilities via CV
    X_step0 = np.mean([features[0][r][:, best_layer, :]
                        for r in range(N_REGIONS)], axis=0)
    step0_probs = np.zeros(n_samples)
    step0_aucs = []

    for train_idx, test_idx in skf.split(X_step0, labels):
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(64, X_step0.shape[1])),
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        )
        clf.fit(X_step0[train_idx], labels[train_idx])
        prob = clf.predict_proba(X_step0[test_idx])[:, 1]
        step0_probs[test_idx] = prob
        try:
            step0_aucs.append(roc_auc_score(labels[test_idx], prob))
        except ValueError:
            step0_aucs.append(0.5)

    print(f"Step-0 probe AUC: {np.mean(step0_aucs):.3f}")

    # Also get per-step probe probabilities for later analysis
    step_probs = {}
    for s in CHECKPOINT_STEPS:
        X = np.mean([features[s][r][:, best_layer, :]
                      for r in range(N_REGIONS)], axis=0)
        probs = np.zeros(n_samples)
        for train_idx, test_idx in skf.split(X, labels):
            clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=min(64, X.shape[1])),
                LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
            )
            clf.fit(X[train_idx], labels[train_idx])
            probs[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        step_probs[s] = probs

    # Adaptive compute simulation
    # Strategy: use step-0 probe P(functional) to split easy/hard.
    # Easy instances (P > threshold): allocate fewer steps.
    # Hard instances: allocate full 128 steps.
    # Assumption: easy instances (high P(func)) maintain correctness
    # with fewer steps, since the model is already confident.

    print(f"\n{'='*80}")
    print(f"Adaptive Compute Simulation (layer={best_layer})")
    print(f"{'='*80}")
    print(f"Baseline: {n_func}/{n_samples} functional "
          f"({100*n_func/n_samples:.1f}%), 128 steps for all")
    print()

    # Easy step budgets to try
    easy_budgets = [16, 32, 64, 96]
    # Confidence thresholds for "easy"
    conf_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    results_table = []

    header = (f"{'Conf':>5} | {'Easy Budget':>11} | {'N Easy':>8} | "
              f"{'N Hard':>8} | {'Avg Steps':>9} | {'Saved':>6} | "
              f"{'Easy Acc':>8} | {'Hard Acc':>8} | {'Overall':>8}")
    print(header)
    print("-" * len(header))

    for conf_thresh in conf_thresholds:
        # Classify: easy = probe confident it will be correct
        easy_mask = step0_probs >= conf_thresh
        hard_mask = ~easy_mask
        n_easy = easy_mask.sum()
        n_hard = hard_mask.sum()

        for easy_steps in easy_budgets:
            # Compute average steps
            avg_steps = (n_easy * easy_steps + n_hard * STEPS) / n_samples
            compute_saved = 1 - avg_steps / STEPS

            # For easy instances: check what fraction are actually correct
            # (with full 128 steps). If correct with 128, likely correct
            # with fewer steps too.
            easy_acc = labels[easy_mask].mean() if n_easy > 0 else 0
            hard_acc = labels[hard_mask].mean() if n_hard > 0 else 0
            overall_acc = labels.mean()  # same as baseline (we still generate)

            # More realistic: estimate accuracy loss from fewer steps.
            # Use probe confidence at the easy_step checkpoint as proxy.
            # Find nearest checkpoint <= easy_steps
            nearest_ckpt = max(s for s in CHECKPOINT_STEPS if s <= easy_steps)
            easy_probs_at_ckpt = step_probs[nearest_ckpt][easy_mask]
            easy_preds_at_ckpt = (easy_probs_at_ckpt >= 0.5).astype(int)
            easy_acc_at_ckpt = (
                (easy_preds_at_ckpt == labels[easy_mask]).mean()
                if n_easy > 0 else 0
            )

            row = {
                "conf_threshold": conf_thresh,
                "easy_steps": easy_steps,
                "n_easy": int(n_easy),
                "n_hard": int(n_hard),
                "avg_steps": round(avg_steps, 1),
                "compute_saved_pct": round(100 * compute_saved, 1),
                "easy_actual_func_rate": round(100 * easy_acc, 1),
                "hard_actual_func_rate": round(100 * hard_acc, 1),
                "overall_func_rate": round(100 * overall_acc, 1),
                "easy_probe_acc_at_ckpt": round(100 * easy_acc_at_ckpt, 1),
            }
            results_table.append(row)

            print(f"{conf_thresh:>5.2f} | {easy_steps:>7} steps | "
                  f"{n_easy:>5} ({100*n_easy/n_samples:>4.0f}%) | "
                  f"{n_hard:>5} ({100*n_hard/n_samples:>4.0f}%) | "
                  f"{avg_steps:>7.1f}  | {100*compute_saved:>5.1f}% | "
                  f"{100*easy_acc:>6.1f}%  | {100*hard_acc:>6.1f}%  | "
                  f"{100*overall_acc:>6.1f}%")

        print()

    # Key insight: among instances the probe thinks are easy,
    # what fraction are actually correct?
    print("--- Probe reliability: P(actually correct | probe says easy) ---")
    for conf_thresh in conf_thresholds:
        easy_mask = step0_probs >= conf_thresh
        if easy_mask.sum() > 0:
            precision = labels[easy_mask].mean()
            print(f"  Conf >= {conf_thresh:.2f}: {easy_mask.sum():>5} instances, "
                  f"precision={100*precision:.1f}%")

    # Save results
    results = {
        "dataset": dataset_key,
        "model": model_key,
        "n_samples": n_samples,
        "n_functional": n_func,
        "best_layer": best_layer,
        "best_layer_auc": round(best_auc, 4),
        "step0_auc": round(float(np.mean(step0_aucs)), 4),
        "adaptive_compute_results": results_table,
    }

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/adaptive_compute_results.json"
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
    print(f"Adaptive compute simulation: dataset={dataset}, model={model}, total={total}")
    result = run_adaptive_sim.remote(dataset, model, chunks, total)
    print("\n" + result)
