"""Train per-layer linear probes on extracted hidden states.

For each of 33 layers, trains logistic regression on mean-pooled activations
to predict functional correctness. Produces the core figure: layer vs accuracy.

Usage:
    python probe/train_probe.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    data_dir = Path("probe/data")
    meta = json.load(open(data_dir / "meta.json"))

    # Load first sample to get dimensions
    sample = np.load(data_dir / meta[0]["npz"])
    n_layers = sample["hidden_states"].shape[0]
    hidden_dim = sample["hidden_states"].shape[2]

    n_pos = sum(1 for m in meta if m["functional"])
    n_neg = len(meta) - n_pos
    print(f"Samples: {len(meta)} ({n_pos} functional, {n_neg} not)")
    print(f"Layers: {n_layers}, hidden_dim: {hidden_dim}")

    # Build per-layer feature matrices (mean pooling over gen tokens)
    print("Loading hidden states...")
    features = [[] for _ in range(n_layers)]
    labels = []

    for idx, m in enumerate(meta):
        data = np.load(data_dir / m["npz"])
        hs = data["hidden_states"]  # [n_layers, gen_len, hidden_dim]
        labels.append(int(m["functional"]))

        for layer in range(n_layers):
            pooled = hs[layer].mean(axis=0).astype(np.float32)
            features[layer].append(pooled)

        if (idx + 1) % 50 == 0 or idx == len(meta) - 1:
            print(f"  loaded {idx + 1}/{len(meta)}")

    labels = np.array(labels)
    for layer in range(n_layers):
        features[layer] = np.stack(features[layer])

    # Train per-layer probes with 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'Layer':>6} {'Acc':>8} {'AUC':>8}")
    print("-" * 24)

    results = []
    for layer in range(n_layers):
        X = features[layer]
        y = labels

        accs, aucs = [], []
        for train_idx, test_idx in skf.split(X, y):
            clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=64),
                LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
            )
            clf.fit(X[train_idx], y[train_idx])

            pred = clf.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))

            prob = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], prob))

        mean_acc = np.mean(accs)
        mean_auc = np.mean(aucs)
        results.append({"layer": layer, "acc": mean_acc, "auc": mean_auc})
        print(f"{layer:>6} {mean_acc:>8.3f} {mean_auc:>8.3f}")

    best = max(results, key=lambda r: r["auc"])
    print(f"\nBest layer: {best['layer']} (AUC={best['auc']:.3f}, Acc={best['acc']:.3f})")

    # Control probe: shuffle labels to check for overfitting
    print("\n--- Control probe (shuffled labels) ---")
    rng = np.random.RandomState(0)
    shuffled_labels = labels.copy()
    rng.shuffle(shuffled_labels)

    print(f"{'Layer':>6} {'Acc':>8} {'AUC':>8}")
    print("-" * 24)

    control_results = []
    for layer in range(n_layers):
        X = features[layer]
        y = shuffled_labels

        accs, aucs = [], []
        for train_idx, test_idx in skf.split(X, y):
            clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=64),
                LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
            )
            clf.fit(X[train_idx], y[train_idx])

            pred = clf.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))

            prob = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], prob))

        mean_acc = np.mean(accs)
        mean_auc = np.mean(aucs)
        control_results.append({"layer": layer, "acc": mean_acc, "auc": mean_auc})
        print(f"{layer:>6} {mean_acc:>8.3f} {mean_auc:>8.3f}")

    # Save results
    with open(data_dir / "probe_results.json", "w") as f:
        json.dump({"real": results, "control": control_results}, f, indent=2)

    # Plot: layer vs AUC, real vs control
    layers_x = [r["layer"] for r in results]
    real_aucs = [r["auc"] for r in results]
    ctrl_aucs = [r["auc"] for r in control_results]
    real_accs = [r["acc"] for r in results]
    ctrl_accs = [r["acc"] for r in control_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC plot
    ax1.plot(layers_x, real_aucs, "o-", label="Real labels", color="#2196F3")
    ax1.plot(layers_x, ctrl_aucs, "x--", label="Shuffled labels (control)", color="#999")
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("AUC")
    ax1.set_title("Probing AUC: Real vs Control")
    ax1.legend()
    ax1.set_ylim(0.3, 1.0)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(layers_x, real_accs, "o-", label="Real labels", color="#FF5722")
    ax2.plot(layers_x, ctrl_accs, "x--", label="Shuffled labels (control)", color="#999")
    ax2.axhline(y=max(n_pos, n_neg) / len(meta), color="gray", linestyle=":",
                alpha=0.5, label=f"Majority baseline ({max(n_pos, n_neg)/len(meta):.1%})")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Probing Accuracy: Real vs Control")
    ax2.legend()
    ax2.set_ylim(0.3, 1.0)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(data_dir / "probe_curve.png", dpi=150)
    print(f"\nFigure saved to {data_dir / 'probe_curve.png'}")

    # Positional analysis: which token positions carry the signal?
    positional_probe(data_dir, meta, labels, skf)


def positional_probe(data_dir, meta, labels, skf):
    """Probe using hidden states from specific token position windows."""
    print("\n\n=== Positional Probing ===")

    # Load raw hidden states (need per-position, not pooled)
    # Use a subset of layers to keep it fast
    probe_layers = [0, 8, 16, 23, 32]
    gen_len = 256
    window = 16  # pool over windows of 16 tokens
    n_windows = gen_len // window

    print(f"Layers: {probe_layers}, window={window}, n_windows={n_windows}")
    print("Loading per-position hidden states...")

    # features[layer_idx][window_idx] = list of pooled vectors
    features = {l: [[] for _ in range(n_windows)] for l in probe_layers}

    for idx, m in enumerate(meta):
        data = np.load(data_dir / m["npz"])
        hs = data["hidden_states"]  # [33, 256, 4096]

        for l in probe_layers:
            for w in range(n_windows):
                start = w * window
                end = start + window
                pooled = hs[l, start:end].mean(axis=0).astype(np.float32)
                features[l][w].append(pooled)

        if (idx + 1) % 50 == 0 or idx == len(meta) - 1:
            print(f"  loaded {idx + 1}/{len(meta)}")

    for l in probe_layers:
        for w in range(n_windows):
            features[l][w] = np.stack(features[l][w])

    # Train probes: layer x position window
    print(f"\nTraining probes (layer x position)...")
    heatmap = np.zeros((len(probe_layers), n_windows))

    for li, l in enumerate(probe_layers):
        row = []
        for w in range(n_windows):
            X = features[l][w]
            y = labels

            aucs = []
            for train_idx, test_idx in skf.split(X, y):
                clf = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=min(64, X.shape[1])),
                    LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
                )
                clf.fit(X[train_idx], y[train_idx])
                prob = clf.predict_proba(X[test_idx])[:, 1]
                aucs.append(roc_auc_score(y[test_idx], prob))

            heatmap[li, w] = np.mean(aucs)
            row.append(f"{np.mean(aucs):.3f}")

        print(f"  Layer {l:>2}: {' '.join(row)}")

    # Save positional results
    pos_results = {
        "layers": probe_layers,
        "window": window,
        "heatmap": heatmap.tolist(),
    }
    with open(data_dir / "positional_probe_results.json", "w") as f:
        json.dump(pos_results, f, indent=2)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlBu_r", vmin=0.45, vmax=0.85)
    ax.set_yticks(range(len(probe_layers)))
    ax.set_yticklabels([f"Layer {l}" for l in probe_layers])
    ax.set_xlabel(f"Token position (window={window})")
    ax.set_xticks(range(n_windows))
    ax.set_xticklabels([f"{w*window}-{(w+1)*window-1}" for w in range(n_windows)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_title("Probing AUC by Layer and Token Position")
    plt.colorbar(im, ax=ax, label="AUC")

    fig.tight_layout()
    fig.savefig(data_dir / "positional_probe.png", dpi=150)
    print(f"\nHeatmap saved to {data_dir / 'positional_probe.png'}")

    # Also plot per-position curves for each layer
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = ["#9E9E9E", "#4CAF50", "#2196F3", "#FF5722", "#9C27B0"]
    for li, l in enumerate(probe_layers):
        positions = [w * window + window // 2 for w in range(n_windows)]
        ax2.plot(positions, heatmap[li], "o-", label=f"Layer {l}", color=colors[li])
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Token position")
    ax2.set_ylabel("AUC")
    ax2.set_title("Probing AUC by Position (per layer)")
    ax2.legend()
    ax2.set_ylim(0.4, 0.9)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(data_dir / "positional_probe_curves.png", dpi=150)
    print(f"Curves saved to {data_dir / 'positional_probe_curves.png'}")


if __name__ == "__main__":
    main()
