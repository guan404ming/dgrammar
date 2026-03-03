"""Run soft constrained decoding on the 11 hard instances."""

import modal

app = modal.App("dgrammar-soft-hard")

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
    .add_local_file("run_soft_hard.py", "/root/run_soft_hard.py")
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/results": RESULTS_VOL},
)
def run_soft(soft_bias: float):
    import subprocess
    import shutil

    tag = f"hard_sb{soft_bias}"
    local_file = f"/root/results/soft_{tag}.jsonl"
    out_file = f"/results/soft_{tag}.jsonl"

    args = [
        "python", "/root/run_soft_hard.py", str(soft_bias),
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
def main():
    bias_values = [0.0, 1.0, 3.0, 5.0, 10.0]
    print(f"Running soft bias on 11 hard instances: {bias_values}")

    handles = []
    for bias in bias_values:
        print(f"  soft_bias={bias}")
        handles.append(run_soft.spawn(bias))

    for bias, handle in zip(bias_values, handles):
        result = handle.get()
        print(f"\n=== soft_bias={bias} ===")
        print(result)
