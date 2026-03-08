"""Seed reranking via step-1 probe on Modal 8x A100.

Phase 1: 8x A100 extract step-1 hidden states + full-gen functional labels (seed=0)
Phase 2: Train probe on merged data (single GPU)
Phase 3: 8x A100 score 5 seeds at step 1, pick best, run full denoising
"""

import modal

app = modal.App("probe-seed-rerank")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0",
        "transformers==4.52.2",
        "accelerate>=0.30",
        "numpy",
        "frozendict",
        "jsonschema",
        "datasets==2.21.0",
        "setuptools<75",
        "maturin",
        "llguidance>=1.6",
        "huggingface_hub",
        "scikit-learn",
    )
    .add_local_dir("../vendor/constrained-diffusion", "/root/constrained-diffusion", copy=True)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/constrained-diffusion/rustformlang_bindings && "
        "maturin build --release && "
        "pip install $(ls target/wheels/*.whl | head -1) && "
        "cd /root/constrained-diffusion && pip install -e .",
    )
    .add_local_dir("../dgrammar", "/root/dgrammar")
    .add_local_file("../pyproject.toml", "/root/pyproject.toml")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)

# ---- Shared generation helpers (loaded inside remote functions) ----

STEPS = 128
GEN_LENGTH = 256
BLOCK_LENGTH = 32
MASK_ID = 126336
TEMPERATURE = 0.2
PROBE_LAYER = 23


def _setup():
    """Common imports and model loading."""
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/constrained-diffusion")

    from constrained_diffusion.eval.dllm.dataset import load_dataset
    from constrained_diffusion.eval.dllm.model import load_model

    dataset = load_dataset("jsonschema")
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")
    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")
    return dataset, eval_model, tokenizer, model


def _run_n_steps(model, prompt_ids, seed, n_steps_to_run, capture_step):
    """Run n diffusion steps, capture hidden states at capture_step."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens

    torch.manual_seed(seed)
    x = torch.full((1, prompt_ids.shape[1] + GEN_LENGTH), MASK_ID,
                    dtype=torch.long, device=model.device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

    gen_start = prompt_ids.shape[1]
    num_blocks = GEN_LENGTH // BLOCK_LENGTH
    steps_per_block = STEPS // num_blocks

    captured_hs = None
    global_step = 0

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * BLOCK_LENGTH
        block_end = gen_start + (num_block + 1) * BLOCK_LENGTH

        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for si in range(steps_per_block):
            if global_step >= n_steps_to_run:
                return x, captured_hs

            need_hs = (global_step == capture_step)
            out = model(x, output_hidden_states=need_hs)
            logits = out.logits

            if need_hs and hasattr(out, 'hidden_states') and out.hidden_states:
                h = out.hidden_states[PROBE_LAYER][0, gen_start:gen_start + GEN_LENGTH]
                captured_hs = h.float().mean(dim=0).cpu().numpy()

            logits_with_noise = add_gumbel_noise(logits, temperature=TEMPERATURE)

            n_transfer = num_transfer_tokens[0, si].item()
            if n_transfer > 0:
                mask_index = x == MASK_ID
                x0 = torch.argmax(logits_with_noise, dim=-1)

                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                x0_p[:, :block_start] = -np.inf
                x0_p[:, block_end:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                n_unmask = min(n_transfer, mask_index[0, block_start:block_end].sum().item())
                if n_unmask > 0:
                    _, indices = torch.topk(confidence[0], k=n_unmask)
                    x[0, indices] = x0[0, indices]

            global_step += 1

    return x, captured_hs


def _run_full(model, prompt_ids, seed):
    """Run full 128-step generation."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens

    torch.manual_seed(seed)
    x = torch.full((1, prompt_ids.shape[1] + GEN_LENGTH), MASK_ID,
                    dtype=torch.long, device=model.device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

    gen_start = prompt_ids.shape[1]
    num_blocks = GEN_LENGTH // BLOCK_LENGTH
    steps_per_block = STEPS // num_blocks

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * BLOCK_LENGTH
        block_end = gen_start + (num_block + 1) * BLOCK_LENGTH

        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for si in range(steps_per_block):
            out = model(x)
            logits = out.logits
            logits_with_noise = add_gumbel_noise(logits, temperature=TEMPERATURE)

            n_transfer = num_transfer_tokens[0, si].item()
            if n_transfer == 0:
                continue

            mask_index = x == MASK_ID
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )
            x0_p[:, :block_start] = -np.inf
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            n_unmask = min(n_transfer, mask_index[0, block_start:block_end].sum().item())
            if n_unmask == 0:
                continue

            _, indices = torch.topk(confidence[0], k=n_unmask)
            x[0, indices] = x0[0, indices]

    return x


def _check_func(x, gen_start, instance, tokenizer, suffix_str, start_line):
    from eval.dllm.jsonmode.checker import check_instance
    code = tokenizer.batch_decode(x[:, gen_start:], skip_special_tokens=True)[0]
    extracted = instance.extract_result(suffix_str + start_line + code)
    try:
        result = check_instance(
            {"instance_id": instance.instance_id(), "extracted": extracted},
            timeout=40,
        )
        return result.get("passed_tests", False), extracted
    except Exception:
        return False, extracted


# ==== Phase 1: Extract step-1 features + functional labels (8x A100) ====

@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def phase1_chunk(offset: int, limit: int):
    """Extract step-1 hidden states and full-gen functional labels for a chunk."""
    import json
    import os
    import time

    import numpy as np
    import torch

    dataset, eval_model, tokenizer, model = _setup()
    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    chunk = all_instances[offset:offset + limit]

    print(f"Phase 1 chunk: offset={offset}, limit={limit}, actual={len(chunk)}")
    t_start = time.monotonic()

    features = []
    labels = []
    instance_ids = []

    for idx, instance in enumerate(chunk):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            continue

        prompt_ids, prompt_len, suffix_str, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )
        gen_start = prompt_ids.shape[1]

        # Step 1 hidden states (2 steps to capture step 1)
        with torch.no_grad():
            _, hs_feat = _run_n_steps(model, prompt_ids, seed=0,
                                       n_steps_to_run=2, capture_step=1)
        features.append(hs_feat)

        # Full generation for functional label
        with torch.no_grad():
            x_full = _run_full(model, prompt_ids, seed=0)
        functional, _ = _check_func(x_full, gen_start, instance, tokenizer,
                                     suffix_str, start_line)
        labels.append(int(functional))
        instance_ids.append(instance.instance_id())

        if (idx + 1) % 10 == 0 or idx == len(chunk) - 1:
            n_func = sum(labels)
            elapsed = time.monotonic() - t_start
            print(f"  [{idx+1}/{len(chunk)}] functional={n_func}/{len(labels)}, "
                  f"time={elapsed:.0f}s")

    # Save chunk results to volume
    os.makedirs("/results/seed_rerank", exist_ok=True)
    out_path = f"/results/seed_rerank/phase1_off{offset}.npz"
    np.savez(out_path,
             features=np.stack(features),
             labels=np.array(labels))

    # Save instance IDs separately (strings)
    with open(f"/results/seed_rerank/phase1_off{offset}_ids.json", "w") as f:
        json.dump(instance_ids, f)

    RESULTS_VOL.commit()
    total_time = time.monotonic() - t_start
    n_func = sum(labels)
    print(f"Phase 1 chunk done: {len(labels)} instances, {n_func} functional, "
          f"{total_time:.0f}s")
    return {"offset": offset, "n": len(labels), "n_functional": n_func,
            "time": total_time}


# ==== Phase 2: Train probe on merged data ====

@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={"/results": RESULTS_VOL},
)
def train_probe(n_chunks: int, chunk_size: int):
    """Merge phase 1 data and train probe. Save probe params to volume."""
    import json
    import os

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    RESULTS_VOL.reload()

    all_features = []
    all_labels = []
    all_ids = []

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"/results/seed_rerank/phase1_off{offset}.npz"
        ids_path = f"/results/seed_rerank/phase1_off{offset}_ids.json"
        if not os.path.exists(path):
            print(f"Warning: missing {path}")
            continue
        data = np.load(path)
        all_features.append(data["features"])
        all_labels.append(data["labels"])
        with open(ids_path) as f:
            all_ids.extend(json.load(f))

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)

    print(f"Merged: {len(labels)} instances, {labels.sum()} functional "
          f"({labels.mean():.1%})")

    # Train probe
    probe = make_pipeline(
        StandardScaler(),
        PCA(n_components=64),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    )
    probe.fit(features, labels)
    train_score = probe.score(features, labels)
    print(f"Probe train accuracy: {train_score:.3f}")

    # Save probe components for phase 3 (avoid pickle, save raw arrays)
    scaler = probe[0]
    pca = probe[1]
    lr = probe[2]

    np.savez("/results/seed_rerank/probe_params.npz",
             scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_,
             pca_components=pca.components_,
             pca_mean=pca.mean_,
             lr_coef=lr.coef_,
             lr_intercept=lr.intercept_)

    # Also save instance ordering for phase 3
    with open("/results/seed_rerank/instance_ids.json", "w") as f:
        json.dump(all_ids, f)

    RESULTS_VOL.commit()
    print("Probe params saved.")
    return {"n": len(labels), "n_functional": int(labels.sum()),
            "baseline_rate": float(labels.mean()),
            "train_acc": float(train_score)}


# ==== Phase 3: Rerank seeds (8x A100) ====

@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def phase3_chunk(offset: int, limit: int, n_seeds: int = 5):
    """Score 5 seeds at step 1 with probe, pick best, run full gen."""
    import json
    import os
    import time

    import numpy as np
    import torch

    RESULTS_VOL.reload()

    # Load probe params and reconstruct scoring function
    params = np.load("/results/seed_rerank/probe_params.npz")
    scaler_mean = params["scaler_mean"]
    scaler_scale = params["scaler_scale"]
    pca_components = params["pca_components"]
    pca_mean = params["pca_mean"]
    lr_coef = params["lr_coef"]
    lr_intercept = params["lr_intercept"]

    def probe_score(feat):
        """Score a single feature vector. Returns P(functional)."""
        x = (feat - scaler_mean) / scaler_scale
        x = (x - pca_mean) @ pca_components.T
        logit = x @ lr_coef.T + lr_intercept
        return 1.0 / (1.0 + np.exp(-logit[0]))

    dataset, eval_model, tokenizer, model = _setup()
    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    chunk = all_instances[offset:offset + limit]

    print(f"Phase 3 chunk: offset={offset}, limit={limit}, actual={len(chunk)}, "
          f"n_seeds={n_seeds}")
    t_start = time.monotonic()

    results = []

    for idx, instance in enumerate(chunk):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            continue

        prompt_ids, prompt_len, suffix_str, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )
        gen_start = prompt_ids.shape[1]

        # Score all seeds at step 1 (1 forward pass each)
        seed_scores = []
        with torch.no_grad():
            for seed in range(n_seeds):
                _, hs_feat = _run_n_steps(model, prompt_ids, seed=seed,
                                           n_steps_to_run=2, capture_step=1)
                score = probe_score(hs_feat)
                seed_scores.append(float(score))

        best_seed = int(np.argmax(seed_scores))

        # Full generation for best seed
        with torch.no_grad():
            x_best = _run_full(model, prompt_ids, seed=best_seed)
        func_best, _ = _check_func(x_best, gen_start, instance, tokenizer,
                                    suffix_str, start_line)

        # Full generation for all other seeds (oracle + random baselines)
        seed_functional = []
        for seed in range(n_seeds):
            if seed == best_seed:
                seed_functional.append(func_best)
            else:
                with torch.no_grad():
                    x_s = _run_full(model, prompt_ids, seed=seed)
                func_s, _ = _check_func(x_s, gen_start, instance, tokenizer,
                                         suffix_str, start_line)
                seed_functional.append(func_s)

        results.append({
            "instance_id": instance.instance_id(),
            "best_seed": best_seed,
            "best_score": seed_scores[best_seed],
            "seed_scores": seed_scores,
            "seed_functional": seed_functional,
            "rerank_functional": func_best,
        })

        if (idx + 1) % 10 == 0 or idx == len(chunk) - 1:
            n_rerank = sum(r["rerank_functional"] for r in results)
            elapsed = time.monotonic() - t_start
            print(f"  [{idx+1}/{len(chunk)}] rerank_func={n_rerank}/{len(results)}, "
                  f"time={elapsed:.0f}s")

    # Save chunk results
    os.makedirs("/results/seed_rerank", exist_ok=True)
    out_path = f"/results/seed_rerank/phase3_off{offset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    RESULTS_VOL.commit()
    total_time = time.monotonic() - t_start
    n_rerank = sum(r["rerank_functional"] for r in results)
    print(f"Phase 3 chunk done: {len(results)} instances, "
          f"rerank_func={n_rerank}, time={total_time:.0f}s")
    return {"offset": offset, "n": len(results), "rerank_functional": n_rerank,
            "time": total_time}


@app.local_entrypoint()
def main(total: int = 272, n_seeds: int = 5, chunks: int = 8):
    import json
    import numpy as np

    chunk_size = (total + chunks - 1) // chunks
    print(f"Seed reranking: {total} instances, {n_seeds} seeds, {chunks}x A100")
    print(f"Chunk size: {chunk_size}")

    # Phase 1: Extract training data
    print(f"\n=== Phase 1: Extract step-1 features + labels ({chunks}x A100) ===")
    handles = []
    for i in range(chunks):
        offset = i * chunk_size
        limit = min(chunk_size, total - offset)
        if limit <= 0:
            break
        handles.append(phase1_chunk.spawn(offset, limit))

    phase1_results = []
    for i, handle in enumerate(handles):
        result = handle.get()
        phase1_results.append(result)
        print(f"  Chunk {i}: {result}")

    total_instances = sum(r["n"] for r in phase1_results)
    total_func = sum(r["n_functional"] for r in phase1_results)
    print(f"Phase 1 total: {total_instances} instances, "
          f"{total_func} functional ({total_func/total_instances:.1%})")

    # Phase 2: Train probe
    print(f"\n=== Phase 2: Train probe ===")
    probe_result = train_probe.remote(len(handles), chunk_size)
    print(f"  {probe_result}")

    # Phase 3: Rerank seeds
    print(f"\n=== Phase 3: Rerank with {n_seeds} seeds ({chunks}x A100) ===")
    handles = []
    for i in range(chunks):
        offset = i * chunk_size
        limit = min(chunk_size, total - offset)
        if limit <= 0:
            break
        handles.append(phase3_chunk.spawn(offset, limit, n_seeds))

    phase3_results = []
    for i, handle in enumerate(handles):
        result = handle.get()
        phase3_results.append(result)
        print(f"  Chunk {i}: {result}")

    # Aggregate
    n_instances = sum(r["n"] for r in phase3_results)
    rerank_func = sum(r["rerank_functional"] for r in phase3_results)
    rerank_rate = rerank_func / n_instances
    baseline_rate = probe_result["baseline_rate"]

    print(f"\n{'='*60}")
    print(f"=== RESULTS ===")
    print(f"{'='*60}")
    print(f"Instances: {n_instances}")
    print(f"Seeds: {n_seeds}")
    print(f"Baseline (seed=0): {baseline_rate:.1%} ({total_func}/{total_instances})")
    print(f"Probe reranking:   {rerank_rate:.1%} ({rerank_func}/{n_instances})")
    print(f"Improvement:       {rerank_rate - baseline_rate:+.1%}")
    print(f"Probe train acc:   {probe_result['train_acc']:.3f}")
