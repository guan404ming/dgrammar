"""Mid-step probing on Modal A100 (multi-GPU).

Runs LLaDA or Dream diffusion generation, captures hidden states at 7
checkpoints, mean-pools into 4 position regions, trains per-layer probes
to predict functional correctness.

Supported datasets: jsonschema (272 instances), gsm8k (1319 instances).
Supported models: llada (LLaDA-8B-Instruct), dream (Dream-v0-Instruct-7B).

Usage:
  cd probe
  ../.venv/bin/modal run modal_midstep_probe.py --dataset jsonschema --model llada --chunks 8
  ../.venv/bin/modal run modal_midstep_probe.py --dataset gsm8k --model dream --chunks 8
"""

import modal
import re

app = modal.App("probe-midstep")

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
CHECKPOINT_STEPS = {0, 1, 4, 16, 32, 64, STEPS - 1}
N_REGIONS = 4


# ---- Dataset helpers ----

def load_instances(dataset_key, offset, limit):
    """Load dataset instances as list of dicts."""
    from datasets import load_dataset
    if dataset_key == "jsonschema":
        ds = load_dataset("eth-sri/json-mode-eval-extended", split="test")
        all_insts = sorted(list(ds), key=lambda x: x["instance_id"])
    elif dataset_key == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        all_insts = list(ds)
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    return all_insts[offset:offset + limit]


def build_system_prompt(dataset_key, instance):
    if dataset_key == "jsonschema":
        schema = instance["schema"]
        return (
            "You are a helpful assistant that answers in JSON. "
            "Here's the JSON schema you must adhere to:\n"
            f"<schema>\n{schema}\n</schema>\n"
        )
    else:
        return (
            "Solve the math problem step by step. "
            "End your answer with #### followed by the final numeric answer."
        )


def build_user_prompt(dataset_key, instance):
    if dataset_key == "jsonschema":
        return instance["input"]
    else:
        return instance["question"]


def build_prompt_llada(tokenizer, model, dataset_key, instance):
    """Build LLaDA prompt. Returns (input_ids, None)."""
    import torch
    sys_prompt = build_system_prompt(dataset_key, instance)
    user_prompt = build_user_prompt(dataset_key, instance)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if dataset_key == "jsonschema":
        text += "```json\n"
    ids = tokenizer(text)["input_ids"]
    return torch.tensor(ids, device=model.device).unsqueeze(0), None


def build_prompt_dream(tokenizer, model, dataset_key, instance):
    """Build Dream prompt. Returns (input_ids, attention_mask)."""
    sys_prompt = build_system_prompt(dataset_key, instance)
    user_prompt = build_user_prompt(dataset_key, instance)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
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
    """Check if the generated output is functionally correct."""
    import json as json_mod

    if dataset_key == "jsonschema":
        # Extract code block content
        prefix = "```json\n"
        extracted = output_text
        if prefix in output_text:
            extracted = output_text.split(prefix, 1)[-1]
        end = extracted.find("```")
        if end != -1:
            extracted = extracted[:end]
        extracted = extracted.strip().strip("`") + "\n"

        # Compare parsed JSON with reference
        try:
            ref = json_mod.loads(instance["output"])
            ref_str = json_mod.dumps(ref, indent=4)
            gen = json_mod.loads(extracted)
            gen_str = json_mod.dumps(gen, indent=4)
            return ref_str == gen_str
        except (json_mod.JSONDecodeError, ValueError):
            return False

    else:  # gsm8k
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


# ---- Hidden states ----

def capture_hidden_states(hs_tuple, gen_start, gen_length, n_regions, region_size):
    import numpy as np
    n_layers = len(hs_tuple)
    feats = np.zeros((n_layers, n_regions, hs_tuple[0].shape[-1]), dtype=np.float32)
    for li in range(n_layers):
        h = hs_tuple[li][0, gen_start:gen_start + gen_length]
        for r in range(n_regions):
            rs = r * region_size
            re_ = rs + region_size
            feats[li, r] = h[rs:re_].float().mean(dim=0).cpu().detach().numpy()
    return feats


# ---- Generation ----

@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(dataset_key: str, model_key: str, offset: int, limit: int):
    """Generate on a subset, save features + labels to volume."""
    import os
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    cfg = MODEL_CFGS[model_key]
    dcfg = DATASET_CFGS[dataset_key]
    MODEL_NAME = cfg["name"]
    MASK_ID = cfg["mask_id"]
    TEMPERATURE = cfg["temperature"]
    GEN_LENGTH = dcfg["gen_length"]
    region_size = GEN_LENGTH // N_REGIONS

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    instances = load_instances(dataset_key, offset, limit)

    # LLaDA helpers
    def add_gumbel_noise(logits, temperature=1.0):
        if temperature == 0:
            return logits
        noise = torch.rand_like(logits, dtype=torch.float64)
        noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        return logits.to(torch.float64) + noise * temperature

    def get_num_transfer_tokens(mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        return torch.where(
            torch.arange(steps, device=mask_index.device).unsqueeze(0) < remainder,
            base + 1, base,
        )

    def generate_llada(x, gen_start):
        num_blocks = GEN_LENGTH // BLOCK_LENGTH
        steps_per_block = STEPS // num_blocks
        global_step = 0
        step_features = {}

        for num_block in range(num_blocks):
            block_start = gen_start + num_block * BLOCK_LENGTH
            block_end = gen_start + (num_block + 1) * BLOCK_LENGTH
            block_mask_index = (x[:, block_start:block_end] == MASK_ID)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for si in range(steps_per_block):
                need_hs = global_step in CHECKPOINT_STEPS
                out = model(x, output_hidden_states=need_hs)
                logits = out.logits

                if need_hs and hasattr(out, 'hidden_states') and out.hidden_states:
                    step_features[global_step] = capture_hidden_states(
                        out.hidden_states, gen_start, GEN_LENGTH, N_REGIONS, region_size,
                    )

                logits_with_noise = add_gumbel_noise(logits, temperature=TEMPERATURE)
                n_transfer = num_transfer_tokens[0, si].item()
                if n_transfer == 0:
                    global_step += 1
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
                if n_unmask > 0:
                    _, indices = torch.topk(confidence[0], k=n_unmask)
                    x[0, indices] = x0[0, indices]
                global_step += 1

        return x, step_features

    def generate_dream(x, gen_start, attention_mask):
        EPS = 1e-3
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(
                attention_mask, (0, x.shape[1] - attention_mask.shape[1]), value=1.0
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, EPS, STEPS + 1, device=x.device)
        step_features = {}

        with torch.no_grad():
            for i in range(STEPS):
                mask_index = x == MASK_ID
                need_hs = i in CHECKPOINT_STEPS
                out = model(x, attention_mask, tok_idx, output_hidden_states=need_hs)
                logits = out.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                if need_hs and hasattr(out, 'hidden_states') and out.hidden_states:
                    step_features[i] = capture_hidden_states(
                        out.hidden_states, gen_start, GEN_LENGTH, N_REGIONS, region_size,
                    )

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
                    x, -torch.inf, device=model.device, dtype=logits.dtype,
                )
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    _, transfer_index = torch.topk(
                        full_confidence, number_transfer_tokens,
                    )
                    x_ = torch.zeros_like(x) + MASK_ID
                    x_[mask_index] = x0.clone()
                    row_indices = (
                        torch.arange(x.size(0), device=model.device)
                        .unsqueeze(1).expand_as(transfer_index)
                    )
                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]

        return x, step_features

    # Main loop
    features = {}
    for s in CHECKPOINT_STEPS:
        features[s] = {r: [] for r in range(N_REGIONS)}
    labels = []

    print(f"Chunk offset={offset}, limit={limit}, "
          f"dataset={dataset_key}, model={model_key}")
    t_start = time.monotonic()

    for i, inst in enumerate(instances):
        if model_key == "llada":
            prompt_ids, _ = build_prompt_llada(tokenizer, model, dataset_key, inst)
        else:
            prompt_ids, attn_mask = build_prompt_dream(tokenizer, model, dataset_key, inst)
        gen_start = prompt_ids.shape[1]

        torch.manual_seed(0)
        t0 = time.monotonic()

        x = torch.full((1, gen_start + GEN_LENGTH), MASK_ID,
                        dtype=torch.long, device=model.device)
        x[:, :gen_start] = prompt_ids.clone()

        if model_key == "llada":
            x, step_features = generate_llada(x, gen_start)
        else:
            x, step_features = generate_dream(x, gen_start, attn_mask)

        # Fill missing captures
        for s in CHECKPOINT_STEPS:
            if s not in step_features:
                if model_key == "llada":
                    out = model(x, output_hidden_states=True)
                else:
                    with torch.no_grad():
                        out = model(x, "full", None, output_hidden_states=True)
                step_features[s] = capture_hidden_states(
                    out.hidden_states, gen_start, GEN_LENGTH, N_REGIONS, region_size,
                )
                break

        # Check correctness
        output_text = tokenizer.batch_decode(
            x[:, gen_start:], skip_special_tokens=True,
        )[0]
        functional = check_functional(dataset_key, inst, output_text)
        labels.append(int(functional))

        for s in CHECKPOINT_STEPS:
            if s in step_features:
                for r in range(N_REGIONS):
                    features[s][r].append(step_features[s][:, r, :])

        elapsed = time.monotonic() - t0
        if (i + 1) % 10 == 0 or i == len(instances) - 1:
            n_func = sum(labels)
            print(f"  [{i+1}/{len(instances)}] functional={n_func}/{len(labels)} "
                  f"({100*n_func/len(labels):.1f}%), time={elapsed:.1f}s")

    total_time = time.monotonic() - t_start

    # Stack and save
    for s in CHECKPOINT_STEPS:
        for r in range(N_REGIONS):
            features[s][r] = np.stack(features[s][r])

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        f"{out_dir}/chunk_off{offset}.npz",
        labels=np.array(labels),
        **{
            f"feat_s{s}_r{r}": features[s][r]
            for s in CHECKPOINT_STEPS
            for r in range(N_REGIONS)
        },
    )

    n_func = sum(labels)
    summary = (f"Chunk off={offset}: {len(labels)} samples, "
               f"{n_func} functional ({100*n_func/len(labels):.1f}%), "
               f"{total_time:.0f}s")
    print(summary)
    RESULTS_VOL.commit()
    return summary


@app.function(
    image=image,
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
)
def run_train_probes(dataset_key: str, model_key: str, n_chunks: int, total: int):
    """Load all chunk data, merge, train probes, save results."""
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

    dcfg = DATASET_CFGS[dataset_key]
    GEN_LENGTH = dcfg["gen_length"]
    region_size = GEN_LENGTH // N_REGIONS
    chunk_size = (total + n_chunks - 1) // n_chunks
    in_dir = f"/results/{dataset_key}_{model_key}"

    all_labels = []
    all_feats = {}

    for i in range(n_chunks):
        offset = i * chunk_size
        path = f"{in_dir}/chunk_off{offset}.npz"
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping")
            continue
        data = np.load(path)
        all_labels.append(data["labels"])
        for s in sorted(CHECKPOINT_STEPS):
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

    n_func = int(labels.sum())
    n_samples = len(labels)
    n_layers = features[sorted(CHECKPOINT_STEPS)[0]][0].shape[1]
    print(f"\nMerged: {n_samples} samples, {n_func} functional "
          f"({100*n_func/n_samples:.1f}%), {n_layers} layers")

    if n_func < 10 or n_func > n_samples - 10:
        print(f"WARNING: severe class imbalance ({n_func}/{n_samples})")

    # Train probes
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sorted_steps = sorted(CHECKPOINT_STEPS)

    print("\n=== Step x Layer probing (all regions pooled) ===")
    step_layer_auc = {}
    for s in sorted_steps:
        step_layer_auc[s] = []
        for layer_idx in range(n_layers):
            X = np.mean([features[s][r][:, layer_idx, :] for r in range(N_REGIONS)], axis=0)
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
            step_layer_auc[s].append(np.mean(aucs))

        print(f"  Step {s:>3}: best_layer={np.argmax(step_layer_auc[s])}, "
              f"best_auc={max(step_layer_auc[s]):.3f}, "
              f"mean_auc={np.mean(step_layer_auc[s]):.3f}")

    final_step = sorted_steps[-1]
    best_layer = int(np.argmax(step_layer_auc[final_step]))
    print(f"\n=== Step x Region probing (layer {best_layer}) ===")
    step_region_auc = {}
    for s in sorted_steps:
        step_region_auc[s] = []
        for r in range(N_REGIONS):
            X = features[s][r][:, best_layer, :]
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
            step_region_auc[s].append(np.mean(aucs))
        print(f"  Step {s:>3}: regions={[f'{a:.3f}' for a in step_region_auc[s]]}")

    # Save results
    results = {
        "dataset": dataset_key,
        "model": MODEL_CFGS[model_key]["name"],
        "n_samples": n_samples,
        "n_functional": n_func,
        "functional_rate": round(n_func / n_samples, 4),
        "gen_length": GEN_LENGTH,
        "checkpoint_steps": sorted_steps,
        "n_layers": n_layers,
        "n_regions": N_REGIONS,
        "region_size": region_size,
        "step_layer_auc": {str(s): aucs for s, aucs in step_layer_auc.items()},
        "step_region_auc": {str(s): aucs for s, aucs in step_region_auc.items()},
        "best_layer": best_layer,
    }

    out_dir = f"/results/{dataset_key}_{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/midstep_probe_results.json"
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
    if model not in MODEL_CFGS:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(MODEL_CFGS.keys())}")
    if dataset not in DATASET_CFGS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASET_CFGS.keys())}")

    if total <= 0:
        total = DATASET_CFGS[dataset]["total"]

    chunk_size = (total + chunks - 1) // chunks
    print(f"Running mid-step probe: dataset={dataset}, model={model}, "
          f"{chunks}x A100, total={total}, chunk_size={chunk_size}")

    handles = []
    for i in range(chunks):
        offset = i * chunk_size
        limit = min(chunk_size, total - offset)
        if limit <= 0:
            break
        print(f"  Chunk {i}: offset={offset}, limit={limit}")
        handles.append(run_chunk.spawn(dataset, model, offset, limit))

    for i, handle in enumerate(handles):
        result = handle.get()
        print(f"  Done: {result}")

    print("\nAll chunks done. Training probes...")
    result = run_train_probes.remote(dataset, model, chunks, total)
    print("\n" + result)
