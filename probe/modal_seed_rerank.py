"""Seed reranking via step-1 probe on Modal 8x A100.

Phase 1: Extract step-1 hidden states + full-gen functional labels (seed=0)
Phase 2: Train probe on merged data (CPU)
Phase 3: Score N seeds at step 1, pick best, run full denoising

Usage:
  cd probe
  ../.venv/bin/modal run modal_seed_rerank.py --dataset jsonschema --model llada --chunks 8
  ../.venv/bin/modal run modal_seed_rerank.py --dataset gsm8k --model dream --chunks 8
"""

import modal
import re

app = modal.App("probe-seed-rerank")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0",
        "transformers==4.52.2",
        "accelerate>=0.30",
        "numpy",
        "datasets==2.21.0",
        "huggingface_hub",
        "scikit-learn",
    )
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)

MODEL_CFGS = {
    "llada": {"name": "GSAI-ML/LLaDA-8B-Instruct", "mask_id": 126336, "temperature": 0.2},
    "dream": {"name": "Dream-org/Dream-v0-Instruct-7B", "mask_id": 151666, "temperature": 0.0},
}

DATASET_CFGS = {
    "jsonschema": {"gen_length": 256, "total": 272},
    "gsm8k": {"gen_length": 512, "total": 1319},
}

STEPS = 128
BLOCK_LENGTH = 32
PROBE_STEP = 1  # capture hidden states at step 1


# ---- Dataset helpers (shared with modal_midstep_probe.py) ----

def load_instances(dataset_key, offset, limit):
    from datasets import load_dataset
    if dataset_key == "jsonschema":
        ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
        all_insts = sorted(list(ds), key=lambda x: x["instance_id"])
    elif dataset_key == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        all_insts = list(ds)
    return all_insts[offset:offset + limit]


def build_system_prompt(dataset_key, instance):
    if dataset_key == "jsonschema":
        return (
            "You are a helpful assistant that answers in JSON. "
            "Here's the JSON schema you must adhere to:\n"
            f"<schema>\n{instance['schema']}\n</schema>\n"
        )
    return (
        "Solve the math problem step by step. "
        "End your answer with #### followed by the final numeric answer."
    )


def build_user_prompt(dataset_key, instance):
    if dataset_key == "jsonschema":
        return instance["input"]
    return instance["question"]


def build_prompt_llada(tokenizer, model, dataset_key, instance):
    import torch
    messages = [
        {"role": "system", "content": build_system_prompt(dataset_key, instance)},
        {"role": "user", "content": build_user_prompt(dataset_key, instance)},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if dataset_key == "jsonschema":
        text += "```json\n"
    ids = tokenizer(text)["input_ids"]
    return torch.tensor(ids, device=model.device).unsqueeze(0), None


def build_prompt_dream(tokenizer, model, dataset_key, instance):
    messages = [
        {"role": "system", "content": build_system_prompt(dataset_key, instance)},
        {"role": "user", "content": build_user_prompt(dataset_key, instance)},
    ]
    if dataset_key == "jsonschema":
        messages.append({"role": "assistant", "content": "```json\n"})
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            continue_final_message=True,
        )
    else:
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        )
    return (inputs.input_ids.to(device=model.device),
            inputs.attention_mask.to(device=model.device))


def check_functional(dataset_key, instance, output_text):
    import json as json_mod
    if dataset_key == "jsonschema":
        extracted = output_text
        prefix = "```json\n"
        if prefix in output_text:
            extracted = output_text.split(prefix, 1)[-1]
        end = extracted.find("```")
        if end != -1:
            extracted = extracted[:end]
        extracted = extracted.strip().strip("`") + "\n"
        try:
            ref = json_mod.loads(instance["output"])
            ref_str = json_mod.dumps(ref, indent=4)
            gen = json_mod.loads(extracted)
            gen_str = json_mod.dumps(gen, indent=4)
            return ref_str == gen_str
        except (json_mod.JSONDecodeError, ValueError):
            return False
    else:
        pred = _extract_gsm8k_answer(output_text)
        gold = _extract_gsm8k_gold(instance["answer"])
        if pred is None:
            return False
        try:
            return float(pred) == float(gold)
        except ValueError:
            return pred.strip() == gold.strip()


def _extract_gsm8k_answer(text):
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _extract_gsm8k_gold(answer_text):
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", answer_text)
    if m:
        return m.group(1).replace(",", "")
    raise ValueError(f"No answer found in: {answer_text}")


# ---- Generation helpers ----

def _run_llada(model_obj, prompt_ids, seed, gen_length, mask_id, temperature,
               n_steps=STEPS, capture_step=None, probe_layer=None):
    """Run LLaDA generation. Optionally capture hidden states at one step."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id,
                    dtype=torch.long, device=model_obj.device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

    gen_start = prompt_ids.shape[1]
    num_blocks = gen_length // BLOCK_LENGTH
    steps_per_block = STEPS // num_blocks
    captured_hs = None
    global_step = 0

    def add_gumbel_noise(logits, temp):
        if temp == 0:
            return logits
        noise = torch.rand_like(logits, dtype=torch.float64)
        noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        return logits.to(torch.float64) + noise * temp

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * BLOCK_LENGTH
        block_end = gen_start + (num_block + 1) * BLOCK_LENGTH
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        mask_num = block_mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps_per_block
        remainder = mask_num % steps_per_block
        num_transfer_tokens = torch.where(
            torch.arange(steps_per_block, device=x.device).unsqueeze(0) < remainder,
            base + 1, base,
        )

        for si in range(steps_per_block):
            if global_step >= n_steps:
                return x, captured_hs

            need_hs = (capture_step is not None and global_step == capture_step)
            out = model_obj(x, output_hidden_states=need_hs)
            logits = out.logits

            if need_hs and hasattr(out, 'hidden_states') and out.hidden_states:
                h = out.hidden_states[probe_layer][0, gen_start:gen_start + gen_length]
                captured_hs = h.float().mean(dim=0).cpu().numpy()

            logits_with_noise = add_gumbel_noise(logits, temperature)
            n_transfer = num_transfer_tokens[0, si].item()
            if n_transfer > 0:
                mask_index = x == mask_id
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


def _run_dream(model_obj, prompt_ids, attn_mask, seed, gen_length, mask_id,
               n_steps=STEPS, capture_step=None, probe_layer=None):
    """Run Dream generation. Optionally capture hidden states at one step."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    EPS = 1e-3
    x = F.pad(prompt_ids, (0, gen_length), value=mask_id)
    gen_start = prompt_ids.shape[1]

    if attn_mask is not None and torch.any(attn_mask == 0.0):
        attn_mask = F.pad(attn_mask, (0, x.shape[1] - attn_mask.shape[1]), value=1.0)
        tok_idx = attn_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attn_mask == 0, 1)
        attn_mask = torch.logical_and(
            attn_mask.unsqueeze(1).unsqueeze(-2),
            attn_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attn_mask = "full"

    timesteps = torch.linspace(1, EPS, STEPS + 1, device=x.device)
    captured_hs = None

    with torch.no_grad():
        for i in range(n_steps):
            mask_index = x == mask_id
            need_hs = (capture_step is not None and i == capture_step)

            out = model_obj(x, attn_mask, tok_idx, output_hidden_states=need_hs)
            logits = out.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            if need_hs and hasattr(out, 'hidden_states') and out.hidden_states:
                h = out.hidden_states[probe_layer][0, gen_start:gen_start + gen_length]
                captured_hs = h.float().mean(dim=0).cpu().numpy()

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            confidence, x0 = mask_logits.max(dim=-1)
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = (
                int(num_mask_token * (1 - s / t))
                if i < STEPS - 1
                else int(num_mask_token)
            )

            full_confidence = torch.full_like(
                x, -torch.inf, device=model_obj.device, dtype=logits.dtype,
            )
            full_confidence[mask_index] = confidence
            if number_transfer_tokens > 0:
                _, transfer_index = torch.topk(
                    full_confidence, number_transfer_tokens,
                )
                x_ = torch.zeros_like(x) + mask_id
                x_[mask_index] = x0.clone()
                row_indices = (
                    torch.arange(x.size(0), device=model_obj.device)
                    .unsqueeze(1).expand_as(transfer_index)
                )
                x[row_indices, transfer_index] = x_[row_indices, transfer_index]

    return x, captured_hs


# ==== Phase 1: Extract step-1 features + functional labels ====

@app.function(image=image, gpu="A100", timeout=7200, volumes={"/results": RESULTS_VOL})
def phase1_chunk(dataset_key: str, model_key: str, offset: int, limit: int,
                 probe_layer: int):
    import os
    import time

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel

    cfg = MODEL_CFGS[model_key]
    dcfg = DATASET_CFGS[dataset_key]
    MODEL_NAME = cfg["name"]
    MASK_ID = cfg["mask_id"]
    TEMPERATURE = cfg["temperature"]
    GEN_LENGTH = dcfg["gen_length"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    instances = load_instances(dataset_key, offset, limit)
    build_prompt = (
        build_prompt_llada if model_key == "llada" else build_prompt_dream
    )

    print(f"Phase 1: offset={offset}, limit={limit}, "
          f"dataset={dataset_key}, model={model_key}")
    t_start = time.monotonic()

    features = []
    labels = []

    for idx, inst in enumerate(instances):
        prompt_ids, attn_mask = build_prompt(tokenizer, model, dataset_key, inst)
        gen_start = prompt_ids.shape[1]

        with torch.no_grad():
            if model_key == "llada":
                _, hs = _run_llada(
                    model, prompt_ids, seed=0, gen_length=GEN_LENGTH,
                    mask_id=MASK_ID, temperature=TEMPERATURE,
                    n_steps=PROBE_STEP + 1, capture_step=PROBE_STEP,
                    probe_layer=probe_layer,
                )
                x_full, _ = _run_llada(
                    model, prompt_ids, seed=0, gen_length=GEN_LENGTH,
                    mask_id=MASK_ID, temperature=TEMPERATURE,
                )
            else:
                _, hs = _run_dream(
                    model, prompt_ids, attn_mask, seed=0, gen_length=GEN_LENGTH,
                    mask_id=MASK_ID,
                    n_steps=PROBE_STEP + 1, capture_step=PROBE_STEP,
                    probe_layer=probe_layer,
                )
                x_full, _ = _run_dream(
                    model, prompt_ids, attn_mask, seed=0, gen_length=GEN_LENGTH,
                    mask_id=MASK_ID,
                )

        features.append(hs)

        output_text = tokenizer.batch_decode(
            x_full[:, gen_start:], skip_special_tokens=True,
        )[0]
        functional = check_functional(dataset_key, inst, output_text)
        labels.append(int(functional))

        if (idx + 1) % 10 == 0 or idx == len(instances) - 1:
            n_func = sum(labels)
            print(f"  [{idx+1}/{len(instances)}] functional={n_func}/{len(labels)} "
                  f"({100*n_func/len(labels):.1f}%)")

    out_dir = f"/results/{dataset_key}_{model_key}/seed_rerank"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/phase1_off{offset}.npz",
             features=np.stack(features), labels=np.array(labels))
    RESULTS_VOL.commit()

    total_time = time.monotonic() - t_start
    n_func = sum(labels)
    summary = f"Phase1 off={offset}: {len(labels)} samples, {n_func} func, {total_time:.0f}s"
    print(summary)
    return {"offset": offset, "n": len(labels), "n_functional": n_func}


# ==== Phase 2: Train probe ====

@app.function(image=image, timeout=600, volumes={"/results": RESULTS_VOL})
def train_probe(dataset_key: str, model_key: str, n_chunks: int, chunk_size: int):
    import os

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    RESULTS_VOL.reload()

    in_dir = f"/results/{dataset_key}_{model_key}/seed_rerank"
    all_features = []
    all_labels = []

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/phase1_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        data = np.load(path)
        all_features.append(data["features"])
        all_labels.append(data["labels"])

    features = np.concatenate(all_features)
    labels_arr = np.concatenate(all_labels)

    print(f"Merged: {len(labels_arr)} instances, {labels_arr.sum()} functional "
          f"({labels_arr.mean():.1%})")

    probe = make_pipeline(
        StandardScaler(),
        PCA(n_components=min(64, features.shape[1])),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    )
    probe.fit(features, labels_arr)
    train_score = probe.score(features, labels_arr)
    print(f"Probe train accuracy: {train_score:.3f}")

    scaler = probe[0]
    pca = probe[1]
    lr = probe[2]

    np.savez(f"{in_dir}/probe_params.npz",
             scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_,
             pca_components=pca.components_,
             pca_mean=pca.mean_,
             lr_coef=lr.coef_,
             lr_intercept=lr.intercept_)
    RESULTS_VOL.commit()

    print("Probe params saved.")
    return {"n": len(labels_arr), "n_functional": int(labels_arr.sum()),
            "baseline_rate": float(labels_arr.mean()),
            "train_acc": float(train_score)}


# ==== Phase 3: Rerank seeds ====

@app.function(image=image, gpu="A100", timeout=7200, volumes={"/results": RESULTS_VOL})
def phase3_chunk(dataset_key: str, model_key: str, offset: int, limit: int,
                 n_seeds: int, probe_layer: int):
    import json
    import os
    import time

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel

    RESULTS_VOL.reload()

    cfg = MODEL_CFGS[model_key]
    dcfg = DATASET_CFGS[dataset_key]
    MODEL_NAME = cfg["name"]
    MASK_ID = cfg["mask_id"]
    TEMPERATURE = cfg["temperature"]
    GEN_LENGTH = dcfg["gen_length"]

    # Load probe
    in_dir = f"/results/{dataset_key}_{model_key}/seed_rerank"
    params = np.load(f"{in_dir}/probe_params.npz")
    scaler_mean = params["scaler_mean"]
    scaler_scale = params["scaler_scale"]
    pca_components = params["pca_components"]
    pca_mean = params["pca_mean"]
    lr_coef = params["lr_coef"]
    lr_intercept = params["lr_intercept"]

    def probe_score(feat):
        x = (feat - scaler_mean) / scaler_scale
        x = (x - pca_mean) @ pca_components.T
        logit = x @ lr_coef.T + lr_intercept
        return 1.0 / (1.0 + np.exp(-logit[0]))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    instances = load_instances(dataset_key, offset, limit)
    build_prompt = (
        build_prompt_llada if model_key == "llada" else build_prompt_dream
    )

    print(f"Phase 3: offset={offset}, limit={limit}, n_seeds={n_seeds}")
    t_start = time.monotonic()

    results = []

    for idx, inst in enumerate(instances):
        prompt_ids, attn_mask = build_prompt(tokenizer, model, dataset_key, inst)
        gen_start = prompt_ids.shape[1]

        # Score all seeds at step 1
        seed_scores = []
        with torch.no_grad():
            for seed in range(n_seeds):
                if model_key == "llada":
                    _, hs = _run_llada(
                        model, prompt_ids, seed=seed, gen_length=GEN_LENGTH,
                        mask_id=MASK_ID, temperature=TEMPERATURE,
                        n_steps=PROBE_STEP + 1, capture_step=PROBE_STEP,
                        probe_layer=probe_layer,
                    )
                else:
                    _, hs = _run_dream(
                        model, prompt_ids, attn_mask, seed=seed,
                        gen_length=GEN_LENGTH, mask_id=MASK_ID,
                        n_steps=PROBE_STEP + 1, capture_step=PROBE_STEP,
                        probe_layer=probe_layer,
                    )
                seed_scores.append(float(probe_score(hs)))

        best_seed = int(np.argmax(seed_scores))

        # Full generation for all seeds
        seed_functional = []
        for seed in range(n_seeds):
            with torch.no_grad():
                if model_key == "llada":
                    x_s, _ = _run_llada(
                        model, prompt_ids, seed=seed, gen_length=GEN_LENGTH,
                        mask_id=MASK_ID, temperature=TEMPERATURE,
                    )
                else:
                    x_s, _ = _run_dream(
                        model, prompt_ids, attn_mask, seed=seed,
                        gen_length=GEN_LENGTH, mask_id=MASK_ID,
                    )
            output_text = tokenizer.batch_decode(
                x_s[:, gen_start:], skip_special_tokens=True,
            )[0]
            seed_functional.append(check_functional(dataset_key, inst, output_text))

        results.append({
            "index": offset + idx,
            "best_seed": best_seed,
            "best_score": seed_scores[best_seed],
            "seed_scores": seed_scores,
            "seed_functional": seed_functional,
            "rerank_functional": seed_functional[best_seed],
        })

        if (idx + 1) % 5 == 0 or idx == len(instances) - 1:
            n_rerank = sum(r["rerank_functional"] for r in results)
            print(f"  [{idx+1}/{len(instances)}] rerank={n_rerank}/{len(results)}")

    os.makedirs(in_dir, exist_ok=True)
    with open(f"{in_dir}/phase3_off{offset}.json", "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()

    total_time = time.monotonic() - t_start
    n_rerank = sum(r["rerank_functional"] for r in results)
    summary = (f"Phase3 off={offset}: {len(results)} samples, "
               f"rerank={n_rerank}, {total_time:.0f}s")
    print(summary)
    return {"offset": offset, "n": len(results), "rerank_functional": n_rerank}


@app.local_entrypoint()
def main(
    dataset: str = "jsonschema",
    model: str = "llada",
    chunks: int = 8,
    total: int = 0,
    n_seeds: int = 5,
    probe_layer: int = 23,
):
    if model not in MODEL_CFGS:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(MODEL_CFGS.keys())}")
    if dataset not in DATASET_CFGS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASET_CFGS.keys())}")

    if total <= 0:
        total = DATASET_CFGS[dataset]["total"]

    chunk_size = (total + chunks - 1) // chunks
    print(f"Seed reranking: dataset={dataset}, model={model}, "
          f"{chunks}x A100, total={total}, n_seeds={n_seeds}, "
          f"probe_layer={probe_layer}")

    # Phase 1
    print(f"\n=== Phase 1: Extract step-1 features + labels ({chunks}x A100) ===")
    handles = []
    for i in range(chunks):
        off = i * chunk_size
        lim = min(chunk_size, total - off)
        if lim <= 0:
            break
        handles.append(phase1_chunk.spawn(dataset, model, off, lim, probe_layer))

    p1_results = []
    for i, handle in enumerate(handles):
        result = handle.get()
        p1_results.append(result)
        print(f"  Chunk {i}: {result}")

    total_inst = sum(r["n"] for r in p1_results)
    total_func = sum(r["n_functional"] for r in p1_results)
    print(f"Phase 1: {total_inst} instances, {total_func} functional "
          f"({100*total_func/total_inst:.1f}%)")

    # Phase 2
    print("\n=== Phase 2: Train probe ===")
    probe_result = train_probe.remote(dataset, model, len(handles), chunk_size)
    print(f"  {probe_result}")

    # Phase 3
    print(f"\n=== Phase 3: Rerank {n_seeds} seeds ({chunks}x A100) ===")
    handles = []
    for i in range(chunks):
        off = i * chunk_size
        lim = min(chunk_size, total - off)
        if lim <= 0:
            break
        handles.append(phase3_chunk.spawn(dataset, model, off, lim, n_seeds,
                                          probe_layer))

    p3_results = []
    for i, handle in enumerate(handles):
        result = handle.get()
        p3_results.append(result)
        print(f"  Chunk {i}: {result}")

    # Aggregate
    n_inst = sum(r["n"] for r in p3_results)
    rerank_func = sum(r["rerank_functional"] for r in p3_results)
    rerank_rate = rerank_func / n_inst
    baseline_rate = probe_result["baseline_rate"]

    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset} + {model}")
    print(f"{'='*60}")
    print(f"Instances:       {n_inst}")
    print(f"Seeds:           {n_seeds}")
    print(f"Baseline (s=0):  {baseline_rate:.1%} ({total_func}/{total_inst})")
    print(f"Probe rerank:    {rerank_rate:.1%} ({rerank_func}/{n_inst})")
    print(f"Improvement:     {rerank_rate - baseline_rate:+.1%}")
    print(f"Probe train acc: {probe_result['train_acc']:.3f}")
