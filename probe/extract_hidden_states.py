"""Extract hidden states from pure LLaDA inference for probing experiments.

Runs unconstrained LLaDA generation (no grammar constraint), extracts
per-layer hidden states at the final step, then checks functional
correctness (function@1) post-hoc for labels.

Usage:
    python probe/extract_hidden_states.py              # 1 instance, 1 seed
    python probe/extract_hidden_states.py 272 5        # 272 instances, 5 seeds
    python probe/extract_hidden_states.py 272 5 42     # with offset 42
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model
from dgrammar.generate import add_gumbel_noise, get_num_transfer_tokens


# ---- Pure LLaDA generation (no grammar) ----

@torch.no_grad()
def generate_unconstrained(
    model, prompt, steps=128, gen_length=256,
    block_length=32, temperature=0.2,
    mask_id=126336, eos_id=126081, eot_id=126348,
    capture_steps=None,
):
    """Pure LLaDA diffusion generation without any grammar constraint.

    Args:
        capture_steps: set of global step indices at which to capture hidden states.
            If None, only captures at the final step.

    Returns (x, step_hidden_states) where step_hidden_states is a dict
    mapping global_step -> tuple of hidden state tensors.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    gen_start = prompt.shape[1]
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    total_steps = num_blocks * steps_per_block

    if capture_steps is None:
        capture_steps = {total_steps - 1}

    step_hidden_states = {}
    global_step = 0

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end = gen_start + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            need_hs = global_step in capture_steps
            out = model(x, output_hidden_states=need_hs)
            logits = out.logits

            if need_hs and hasattr(out, 'hidden_states') and out.hidden_states:
                step_hidden_states[global_step] = out.hidden_states

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            n_transfer = num_transfer_tokens[0, i].item()
            if n_transfer == 0:
                global_step += 1
                continue

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
            if n_unmask == 0:
                global_step += 1
                continue

            _, indices = torch.topk(confidence[0], k=n_unmask)
            x[0, indices] = x0[0, indices]
            global_step += 1

    # Fill any missing captures
    for s in capture_steps:
        if s not in step_hidden_states:
            out = model(x, output_hidden_states=True)
            step_hidden_states[s] = out.hidden_states
            break

    return x, step_hidden_states


# ---- Post-hoc functional correctness check ----

def check_functional(extracted, instance):
    """Check functional correctness (function@1) post-hoc.

    Compares generated JSON against the reference solution from the dataset.
    Uses ETH's checker for consistency with benchmark evaluation.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "constrained-diffusion"))
    from eval.dllm.jsonmode.checker import check_instance

    output = {
        "instance_id": instance.instance_id(),
        "extracted": extracted,
    }
    try:
        result = check_instance(output, timeout=40)
        return result.get("passed_tests", False), result.get("syntax_ok", False)
    except Exception:
        return False, False


# ---- Main ----

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    dataset = load_dataset("jsonschema")
    eval_model = load_model("GSAI-ML/LLaDA-8B-Instruct")

    tokenizer = eval_model.tokenizer("cuda")
    model = eval_model.model("cuda")

    all_instances = sorted(dataset, key=lambda x: x.instance_id())
    instances = all_instances[offset:offset + limit]

    output_dir = Path("probe/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pure LLaDA hidden state extraction: {len(instances)} instances, "
          f"{n_seeds} seeds, offset={offset}")

    n_functional = 0
    n_syntax = 0
    n_total = 0
    all_meta = []

    for i, instance in enumerate(instances):
        schema_str = instance.data.get("schema", "")
        if not schema_str:
            continue

        prompt_ids, prompt_len, suffix_str, start_line, prompt_raw = (
            eval_model.prepare_prompt(instance, tokenizer, model, trace=False)
        )
        gen_start = prompt_ids.shape[1]

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            t0 = time.monotonic()

            # Capture at early (step 0), 25%, 50%, 75%, and final step
            total_steps = 128  # steps
            capture_steps = {0, total_steps // 4, total_steps // 2,
                             3 * total_steps // 4, total_steps - 1}

            x, step_hidden_states = generate_unconstrained(
                model, prompt_ids, steps=128, gen_length=256,
                block_length=32, temperature=0.2,
                capture_steps=capture_steps,
            )

            elapsed = time.monotonic() - t0

            # Decode generated tokens to text
            code = tokenizer.batch_decode(
                x[:, gen_start:], skip_special_tokens=True
            )[0]
            extracted = instance.extract_result(suffix_str + start_line + code)

            # Post-hoc functional correctness check
            functional, syntax_ok = check_functional(extracted, instance)

            n_total += 1
            if functional:
                n_functional += 1
            if syntax_ok:
                n_syntax += 1

            gen_len = x.shape[1] - gen_start
            gen_ids = x[0, gen_start:].tolist()

            # Save per-step hidden states
            fname = f"{instance.instance_id()}_s{seed}"
            npz_data = {"token_ids": np.array(gen_ids, dtype=np.int32)}

            for step_idx, hs_tuple in step_hidden_states.items():
                n_layers = len(hs_tuple)
                layers = []
                for layer_idx in range(n_layers):
                    h = hs_tuple[layer_idx][0, gen_start:gen_start + gen_len]
                    layers.append(h.cpu().to(torch.float16).numpy())
                npz_data[f"step_{step_idx}"] = np.stack(layers, axis=0)

            np.savez_compressed(output_dir / f"{fname}.npz", **npz_data)

            file_size = (output_dir / f"{fname}.npz").stat().st_size / 1024 / 1024
            print(
                f"  [{i+1}/{len(instances)}] {instance.instance_id()} s{seed}: "
                f"functional={functional}, syntax={syntax_ok}, time={elapsed:.1f}s, "
                f"steps={sorted(step_hidden_states.keys())}, size={file_size:.1f}MB"
            )

            all_meta.append({
                "instance_id": instance.instance_id(),
                "seed": seed,
                "functional": functional,
                "syntax_ok": syntax_ok,
                "extracted": extracted,
                "n_layers": n_layers,
                "hidden_dim": npz_data[f"step_{sorted(step_hidden_states.keys())[0]}"].shape[-1],
                "gen_len": gen_len,
                "capture_steps": sorted(step_hidden_states.keys()),
                "npz": f"{fname}.npz",
            })

    with open(output_dir / "meta.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nDone. functional={n_functional}/{n_total} ({n_functional/n_total*100:.1f}%), "
          f"syntax={n_syntax}/{n_total} ({n_syntax/n_total*100:.1f}%)")
    print(f"Metadata saved to {output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
