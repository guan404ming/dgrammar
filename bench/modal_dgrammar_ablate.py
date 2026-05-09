"""Dgrammar-only Modal launcher for ablation runs.

This avoids building unrelated baseline images when running Dgrammar variants.

Usage:
    modal run modal_dgrammar_ablate.py --total 148 --chunks 2 \
        --dataset jsb_medium_test --max-resamples 500
"""

import modal

app = modal.App("dgrammar-ablate")

_BASE_PIP = [
    "torch>=2.0",
    "transformers==4.52.2",
    "accelerate>=0.30",
    "numpy",
    "frozendict",
    "jsonschema",
    "datasets==2.21.0",
    "setuptools<75",
    "maturin",
    "huggingface_hub",
    "llguidance>=1.6",
]

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(*_BASE_PIP)
    .run_commands(
        "git clone --depth 1 https://github.com/eth-sri/constrained-diffusion.git "
        "/root/constrained-diffusion && "
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/constrained-diffusion/rustformlang_bindings && "
        "rm -rf target/wheels && "
        "maturin build --release && "
        "pip install target/wheels/*.whl && "
        "cd /root/constrained-diffusion && pip install -e .",
    )
    .add_local_dir("../dgrammar", "/root/dgrammar")
    .add_local_file("runner/run_dgrammar.py", "/root/run_dgrammar.py")
    .add_local_file("jsb_dataset.py", "/root/jsb_dataset.py")
    .add_local_file("../pyproject.toml", "/root/pyproject.toml")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)
_COMMON_FN_KW = dict(timeout=7200, volumes={"/results": RESULTS_VOL})


def _chunk_fname(run_id: str, tag: str, dataset: str, seed: int, steps: int, offset: int) -> str:
    folder = f"dgrammar/{run_id}_{tag}" if tag else f"dgrammar/{run_id}"
    ds_safe = dataset.replace("/", "_")
    sfx = f"_off{offset}" if offset > 0 else ""
    return f"{folder}/{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"


@app.function(image=_image, gpu="B200", **_COMMON_FN_KW)
def run_dgrammar(seed: int, limit: int, offset: int, steps: int,
                 dataset: str, block_ar: int, max_resamples: int,
                 max_batch: int, async_mask: int, ac_enabled: int,
                 model_name: str, run_id: str, ac_mode: str = "greedy"):
    import os
    import shutil
    import subprocess

    parts = []
    model_short = model_name.split("/")[-1].lower().replace("-instruct", "")
    if "llada" not in model_short:
        parts.append(model_short)
    if not block_ar:
        parts.append("fullpar")
    if max_resamples != 100:
        parts.append(f"r{max_resamples}")
    if max_batch != 8:
        parts.append(f"b{max_batch}")
    if not async_mask:
        parts.append("noasync")
    if not ac_enabled:
        parts.append("noac")
    if ac_mode == "ar":
        parts.append("acar")
    tag = "_".join(parts) or "r100"
    fname = _chunk_fname(run_id, tag, dataset, seed, steps, offset)

    local_file = f"/root/results/{fname}"
    out_file = f"/results/{fname}"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if os.path.exists(out_file):
        os.remove(out_file)

    result = subprocess.run(
        [
            "python", "/root/run_dgrammar.py",
            str(seed), str(limit), dataset, str(steps), str(offset),
            str(block_ar), str(max_resamples), fname,
            str(max_batch), str(async_mask), str(ac_enabled),
            model_name, ac_mode,
        ],
        capture_output=True,
        text=True,
        cwd="/root",
        env={
            "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/root",
            "PYTHONPATH": "/root:/root/constrained-diffusion",
        },
    )
    print(result.stdout[-5000:] if result.stdout else "")
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    if result.returncode != 0:
        raise RuntimeError(f"run_dgrammar.py failed with code {result.returncode}")

    shutil.copy2(local_file, out_file)
    return result.stdout[-5000:] if result.stdout else ""


@app.local_entrypoint()
def main(
    seed: int = 0,
    total: int = 148,
    steps: int = 128,
    chunks: int = 2,
    dataset: str = "jsb_medium_test",
    block_ar: int = 1,
    max_resamples: int = 500,
    max_batch: int = 8,
    async_mask: int = 1,
    ac_enabled: int = 1,
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    ac_mode: str = "greedy",
):
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_size = (total + chunks - 1) // chunks
    print(f"Running dgrammar ablation ({run_id}): model={model_name}, dataset={dataset}, "
          f"seed={seed}, T={steps}, total={total}, chunk_size={chunk_size}, "
          f"max_resamples={max_resamples}, max_batch={max_batch}, "
          f"async={async_mask}, ac={ac_enabled}")

    handles = []
    for i in range(chunks):
        offset = i * chunk_size
        limit = min(chunk_size, total - offset)
        if limit <= 0:
            break
        print(f"  Chunk {i}: offset={offset}, limit={limit}")
        handles.append(
            run_dgrammar.spawn(
                seed, limit, offset, steps, dataset, block_ar, max_resamples,
                max_batch, async_mask, ac_enabled, model_name, run_id, ac_mode
            )
        )

    for i, handle in enumerate(handles):
        result = handle.get()
        print(f"\n{'=' * 60}\n=== Chunk {i} ===\n{'=' * 60}")
        print(result)
