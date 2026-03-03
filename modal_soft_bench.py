"""Run soft constrained decoding POC on Modal with A100 GPU."""

import modal

app = modal.App("dgrammar-soft-bench")

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
    .add_local_file("run_soft_eval.py", "/root/run_soft_eval.py")
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
)
def run_soft(seed: int, limit: int, offset: int, dataset: str, steps: int, soft_bias: float):
    import subprocess
    import shutil

    ds_safe = dataset.replace("/", "_")
    bias_tag = f"sb{soft_bias}"
    suffix = f"_off{offset}" if offset > 0 else ""
    local_file = f"/root/results/soft_{bias_tag}_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl"
    out_file = f"/results/soft_{bias_tag}_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl"

    args = [
        "python", "/root/run_soft_eval.py",
        str(seed), str(limit), dataset, str(steps), str(offset), str(soft_bias),
    ]

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd="/root",
        env={
            "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/root",
            "PYTHONPATH": "/root:/root/constrained-diffusion",
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
    limit: int = 10,
    dataset: str = "jsonschema",
    steps: int = 128,
):
    # Test multiple soft bias values in parallel
    bias_values = [3.0, 5.0, 10.0, 20.0]
    print(f"Running soft bias POC: {len(bias_values)} configs x {limit} instances")

    handles = []
    for bias in bias_values:
        print(f"  soft_bias={bias}")
        handles.append(run_soft.spawn(seed, limit, 0, dataset, steps, bias))

    for bias, handle in zip(bias_values, handles):
        result = handle.get()
        print(f"\n=== soft_bias={bias} ===")
        print(result)
