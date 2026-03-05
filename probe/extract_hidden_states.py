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
):
    """Pure LLaDA diffusion generation without any grammar constraint.

    Returns (x, hidden_states) where hidden_states is from the final forward pass.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    gen_start = prompt.shape[1]
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = gen_start + num_block * block_length
        block_end = gen_start + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            is_last_step = (num_block == num_blocks - 1 and i == steps_per_block - 1)
            out = model(x, output_hidden_states=is_last_step)
            logits = out.logits

            if is_last_step and hasattr(out, 'hidden_states') and out.hidden_states:
                hidden_states = out.hidden_states

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            n_transfer = num_transfer_tokens[0, i].item()
            if n_transfer == 0:
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
                continue

            _, indices = torch.topk(confidence[0], k=n_unmask)
            x[0, indices] = x0[0, indices]

    # If we didn't capture hidden states (e.g. complete early), do one more pass
    if 'hidden_states' not in dir():
        out = model(x, output_hidden_states=True)
        hidden_states = out.hidden_states

    return x, hidden_states


# ---- Post-hoc functional correctness check ----

def check_functional(extracted, instance):
    """Check functional correctness (function@1) post-hoc.

    Compares generated JSON against the reference solution from the dataset.
    Uses ETH's checker for consistency with benchmark evaluation.
    """
    from constrained_diffusion.eval.dllm.jsonmode.checker import check_instance

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

            x, hidden_states = generate_unconstrained(
                model, prompt_ids, steps=128, gen_length=256,
                block_length=32, temperature=0.2,
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

            # Extract generation region hidden states
            gen_len = x.shape[1] - gen_start
            n_layers = len(hidden_states)

            layers = []
            for layer_idx in range(n_layers):
                h = hidden_states[layer_idx][0, gen_start:gen_start + gen_len]
                layers.append(h.cpu().to(torch.float16).numpy())

            all_hidden = np.stack(layers, axis=0)  # [n_layers, gen_len, hidden_dim]

            gen_ids = x[0, gen_start:].tolist()

            print(
                f"  [{i+1}/{len(instances)}] {instance.instance_id()} s{seed}: "
                f"functional={functional}, syntax={syntax_ok}, time={elapsed:.1f}s, "
                f"shape={all_hidden.shape}, size={all_hidden.nbytes/1024/1024:.1f}MB"
            )

            # Save hidden states + labels
            fname = f"{instance.instance_id()}_s{seed}"
            np.savez_compressed(
                output_dir / f"{fname}.npz",
                hidden_states=all_hidden,
                token_ids=np.array(gen_ids, dtype=np.int32),
            )

            meta = {
                "instance_id": instance.instance_id(),
                "seed": seed,
                "functional": functional,
                "syntax_ok": syntax_ok,
                "extracted": extracted,
                "n_layers": n_layers,
                "hidden_dim": all_hidden.shape[-1],
                "gen_len": gen_len,
            }
            with open(output_dir / f"{fname}.json", "w") as f:
                json.dump(meta, f)

    print(f"\nDone. functional={n_functional}/{n_total} ({n_functional/n_total*100:.1f}%), "
          f"syntax={n_syntax}/{n_total} ({n_syntax/n_total*100:.1f}%)")


if __name__ == "__main__":
    main()
