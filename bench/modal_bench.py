"""Unified Modal launcher for dgrammar + baselines (lave, igcd, vanilla).

Usage:
    cd bench && ../.venv/bin/modal run modal_bench.py \
        --method dgrammar --seed 0 --total 586 --steps 128 --chunks 8 \
        --dataset jsb_medium_test --max-resamples 500

    ../.venv/bin/modal run modal_bench.py --method lave --total 586 --chunks 8 \
        --dataset jsb_medium_test --instance-timeout 120

    ../.venv/bin/modal run modal_bench.py --method igcd --total 272 --chunks 4
    ../.venv/bin/modal run modal_bench.py --method vanilla --total 511 --chunks 8 \
        --dataset jsb_medium_test
"""

import modal

app = modal.App("dgrammar-bench")

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
]

_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(*_BASE_PIP)
)

# constrained-diffusion vendor: used by dgrammar / igcd / vanilla
_cd_image = (
    _base_image
    .pip_install("llguidance>=1.6")
    .add_local_dir("../vendor/constrained-diffusion", "/root/constrained-diffusion", copy=True)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/constrained-diffusion/rustformlang_bindings && "
        "rm -rf target/wheels && "
        "maturin build --release && "
        "pip install target/wheels/*.whl && "
        "cd /root/constrained-diffusion && pip install -e .",
    )
)

_dgrammar_image = (
    _cd_image
    .add_local_dir("../dgrammar", "/root/dgrammar")
    .add_local_file("runner/run_dgrammar.py", "/root/run_dgrammar.py")
    .add_local_file("jsb_dataset.py", "/root/jsb_dataset.py")
    .add_local_file("../pyproject.toml", "/root/pyproject.toml")
)

_igcd_image = _cd_image.add_local_file("runner/run_igcd.py", "/root/run_igcd.py")

_vanilla_image = (
    _cd_image
    .add_local_file("runner/run_vanilla.py", "/root/run_vanilla.py")
    .add_local_file("jsb_dataset.py", "/root/jsb_dataset.py")
    .add_local_file("../pyproject.toml", "/root/pyproject.toml")
)

# CD4dLLM vendor: used by LAVE baseline
_lave_image = (
    _base_image
    .pip_install("llguidance>=1.6", "stopit")
    .add_local_dir("../vendor/CD4dLLM", "/root/CD4dLLM", copy=True)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/CD4dLLM/rustformlang_bindings && "
        "maturin build --release && "
        "pip install target/wheels/*cp312*.whl && "
        "cd /root/CD4dLLM && pip install -e .",
    )
    .add_local_file("runner/run_lave.py", "/root/run_lave.py")
    .add_local_file("jsb_dataset.py", "/root/jsb_dataset.py")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)

_COMMON_FN_KW = dict(timeout=7200, volumes={"/results": RESULTS_VOL})


def _run_and_save(script: str, args: list, pythonpath: str, out_fname: str):
    """Invoke a runner script, stream its output, copy its jsonl to shared volume."""
    import os
    import shutil
    import subprocess

    local_file = f"/root/results/{out_fname}"
    out_file = f"/results/{out_fname}"

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if os.path.exists(out_file):
        os.remove(out_file)

    result = subprocess.run(
        ["python", f"/root/{script}", *args],
        capture_output=True,
        text=True,
        cwd="/root",
        env={
            "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/root",
            "PYTHONPATH": pythonpath,
        },
    )
    print(result.stdout[-5000:] if result.stdout else "")
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    try:
        shutil.copy2(local_file, out_file)
        print(f"Saved to {out_file}")
    except FileNotFoundError:
        print(f"Result file not found: {local_file}")

    return result.stdout[-5000:] if result.stdout else result.stderr[-2000:]


def _chunk_fname(method: str, run_id: str, tag: str, dataset: str,
                 seed: int, steps: int, offset: int) -> str:
    """results/{method}/{run_id}[_{tag}]/{dataset}_s{seed}_t{steps}[_off{K}].jsonl"""
    folder = f"{method}/{run_id}_{tag}" if tag else f"{method}/{run_id}"
    ds_safe = dataset.replace("/", "_")
    sfx = f"_off{offset}" if offset > 0 else ""
    return f"{folder}/{ds_safe}_s{seed}_t{steps}{sfx}.jsonl"


@app.function(image=_dgrammar_image, gpu="B200", **_COMMON_FN_KW)
def run_dgrammar(seed: int, limit: int, offset: int, steps: int,
                 dataset: str, block_ar: int, max_resamples: int, run_id: str):
    parts = []
    if not block_ar:
        parts.append("fullpar")
    if max_resamples != 100:
        parts.append(f"r{max_resamples}")
    tag = "_".join(parts)
    fname = _chunk_fname("dgrammar", run_id, tag, dataset, seed, steps, offset)
    return _run_and_save(
        "run_dgrammar.py",
        [str(seed), str(limit), dataset, str(steps), str(offset),
         str(block_ar), str(max_resamples), fname],
        "/root:/root/constrained-diffusion",
        fname,
    )


@app.function(image=_lave_image, gpu="B200", **_COMMON_FN_KW)
def run_lave(seed: int, limit: int, offset: int, steps: int,
             dataset: str, instance_timeout: int, run_id: str):
    fname = _chunk_fname("lave", run_id, "", dataset, seed, steps, offset)
    return _run_and_save(
        "run_lave.py",
        [str(seed), str(limit), dataset, str(steps), str(offset),
         str(instance_timeout), fname],
        "/root:/root/CD4dLLM",
        fname,
    )


@app.function(image=_igcd_image, gpu="A100", **_COMMON_FN_KW)
def run_igcd(seed: int, limit: int, offset: int, steps: int, run_id: str):
    fname = _chunk_fname("igcd", run_id, "", "jsonschema", seed, steps, offset)
    return _run_and_save(
        "run_igcd.py",
        [str(seed), str(limit), "jsonschema", str(steps), str(offset), fname],
        "/root:/root/constrained-diffusion",
        fname,
    )


@app.function(image=_vanilla_image, gpu="B200", **_COMMON_FN_KW)
def run_vanilla(seed: int, limit: int, offset: int, steps: int,
                dataset: str, run_id: str):
    fname = _chunk_fname("vanilla", run_id, "", dataset, seed, steps, offset)
    return _run_and_save(
        "run_vanilla.py",
        [str(seed), str(limit), dataset, str(steps), str(offset), fname],
        "/root:/root/constrained-diffusion",
        fname,
    )


@app.local_entrypoint()
def main(
    method: str = "dgrammar",
    seed: int = 0,
    total: int = 272,
    steps: int = 128,
    chunks: int = 2,
    dataset: str = "jsonschema",
    block_ar: int = 1,
    max_resamples: int = 100,
    instance_timeout: int = 120,
):
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_size = (total + chunks - 1) // chunks
    print(f"Running {method} ({run_id}) on {chunks} chunk(s): {dataset}, "
          f"seed={seed}, T={steps}, total={total}, chunk_size={chunk_size}")

    handles = []
    for i in range(chunks):
        offset = i * chunk_size
        limit = min(chunk_size, total - offset)
        if limit <= 0:
            break
        print(f"  Chunk {i}: offset={offset}, limit={limit}")
        if method == "dgrammar":
            h = run_dgrammar.spawn(seed, limit, offset, steps, dataset,
                                    block_ar, max_resamples, run_id)
        elif method == "lave":
            h = run_lave.spawn(seed, limit, offset, steps, dataset,
                                instance_timeout, run_id)
        elif method == "igcd":
            h = run_igcd.spawn(seed, limit, offset, steps, run_id)
        elif method == "vanilla":
            h = run_vanilla.spawn(seed, limit, offset, steps, dataset, run_id)
        else:
            raise ValueError(f"Unknown method: {method} (choose dgrammar/lave/igcd/vanilla)")
        handles.append(h)

    for i, handle in enumerate(handles):
        result = handle.get()
        print(f"\n{'='*60}\n=== Chunk {i} ===\n{'='*60}")
        print(result)
