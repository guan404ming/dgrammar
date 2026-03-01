"""Benchmark dGrammar with tensor parallelism (multi-GPU) on Modal.

Runs a single instance with different GPU counts to measure forward pass speedup.
"""

import modal

app = modal.App("dgrammar-tp-bench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0",
        "transformers==4.52.2",
        "numpy",
        "frozendict",
        "jsonschema",
        "datasets==2.21.0",
        "setuptools<75",
        "maturin",
    )
    .add_local_dir("vendor/constrained-diffusion", "/root/constrained-diffusion", copy=True)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/constrained-diffusion/rustformlang_bindings && "
        "maturin build --release && "
        "pip install target/wheels/*.whl && "
        "cd /root/constrained-diffusion && pip install -e .",
    )
    .add_local_dir("dgrammar", "/root/dgrammar")
    .add_local_file("run_dgrammar_eval.py", "/root/run_dgrammar_eval.py")
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1),
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
)
def run_1gpu(seed: int, limit: int, dataset: str, steps: int):
    return _run(seed, limit, dataset, steps, "1gpu")


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=2),
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
)
def run_2gpu(seed: int, limit: int, dataset: str, steps: int):
    return _run(seed, limit, dataset, steps, "2gpu")


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=4),
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
)
def run_4gpu(seed: int, limit: int, dataset: str, steps: int):
    return _run(seed, limit, dataset, steps, "4gpu")


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=8),
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
)
def run_8gpu(seed: int, limit: int, dataset: str, steps: int):
    return _run(seed, limit, dataset, steps, "8gpu")


def _run(seed, limit, dataset, steps, tag):
    import subprocess
    import shutil
    import torch

    n_gpus = torch.cuda.device_count()
    print(f"[{tag}] GPUs available: {n_gpus}")

    ds_safe = dataset.replace("/", "_")
    local_file = f"/root/results/dgrammar_{ds_safe}_s{seed}_t{steps}_{tag}.jsonl"
    out_file = f"/results/dgrammar_{ds_safe}_s{seed}_t{steps}_{tag}.jsonl"

    result = subprocess.run(
        [
            "python", "/root/run_dgrammar_eval.py",
            str(seed), str(limit), dataset, str(steps), "0", tag,
        ],
        capture_output=True,
        text=True,
        cwd="/root",
        env={
            "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/root",
            "PYTHONPATH": "/root:/root/constrained-diffusion",
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(n_gpus)),
        },
    )
    print(result.stdout[-3000:] if result.stdout else "")
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    try:
        shutil.copy2(local_file, out_file)
        print(f"Saved to {out_file}")
    except FileNotFoundError:
        print(f"Result file not found: {local_file}")

    return result.stdout[-3000:] if result.stdout else result.stderr[-2000:]


@app.local_entrypoint()
def main(
    seed: int = 0,
    limit: int = 5,
    dataset: str = "jsonschema",
    steps: int = 128,
):
    print(f"Tensor parallelism benchmark: {limit} instances, T={steps}")

    # Run all GPU configs in parallel
    handles = {
        "1gpu": run_1gpu.spawn(seed, limit, dataset, steps),
        "2gpu": run_2gpu.spawn(seed, limit, dataset, steps),
        "4gpu": run_4gpu.spawn(seed, limit, dataset, steps),
        "8gpu": run_8gpu.spawn(seed, limit, dataset, steps),
    }

    for tag, handle in handles.items():
        result = handle.get()
        print(f"\n=== {tag} ===")
        print(result)
