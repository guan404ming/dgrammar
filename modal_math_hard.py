"""Run mathematical approaches on 11 hard instances on Modal."""

import modal

app = modal.App("dgrammar-math-hard")

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
    .add_local_file("run_math_hard.py", "/root/run_math_hard.py")
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
)

RESULTS_VOL = modal.Volume.from_name("dgrammar-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_experiment(mode: str, k: int = 5):
    import subprocess
    import shutil

    if mode == "reject":
        tag = f"reject_k{k}"
    else:
        tag = "kl_proj"

    local_file = f"/root/results/math_{tag}.jsonl"
    out_file = f"/results/math_{tag}.jsonl"

    args = ["python", "/root/run_math_hard.py", mode]
    if mode == "reject":
        args.append(str(k))

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
    print(result.stdout[-5000:] if result.stdout else "")
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    try:
        shutil.copy2(local_file, out_file)
        print(f"Saved to {out_file}")
    except FileNotFoundError:
        print(f"Result file not found: {local_file}")

    return result.stdout[-5000:] if result.stdout else result.stderr[-2000:]


@app.local_entrypoint()
def main():
    print("Running 3 experiments in parallel on A100:")
    print("  1. Rejection sampling K=5")
    print("  2. Rejection sampling K=10")
    print("  3. KL projection")

    handles = [
        run_experiment.spawn("reject", 5),
        run_experiment.spawn("reject", 10),
        run_experiment.spawn("kl"),
    ]
    labels = ["Reject K=5", "Reject K=10", "KL Projection"]

    for label, handle in zip(labels, handles):
        result = handle.get()
        print(f"\n{'='*60}")
        print(f"=== {label} ===")
        print(f"{'='*60}")
        print(result)
