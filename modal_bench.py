"""Run dGrammar benchmarks on Modal with A100 GPUs."""

import modal

app = modal.App("dgrammar-bench")

# Build image with all dependencies
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
        # Install Rust + build rustformlang + install constrained-diffusion
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
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(seed: int, limit: int, offset: int, dataset: str, steps: int, tag: str = ""):
    import subprocess
    import shutil

    ds_safe = dataset.replace("/", "_")
    # Build the same filename that run_dgrammar_eval.py will produce
    suffix = f"_off{offset}" if offset > 0 else ""
    if tag:
        suffix += f"_{tag}"
    local_file = f"/root/results/dgrammar_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl"
    out_file = f"/results/dgrammar_{ds_safe}_s{seed}_t{steps}{suffix}.jsonl"

    args = [
        "python", "/root/run_dgrammar_eval.py",
        str(seed), str(limit), dataset, str(steps), str(offset),
    ]
    if tag:
        args.append(tag)

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
        print("STDERR:", result.stderr[-1000:])

    try:
        shutil.copy2(local_file, out_file)
        print(f"Saved to {out_file}")
    except FileNotFoundError:
        print(f"Result file not found: {local_file}")

    return result.stdout[-3000:] if result.stdout else result.stderr[-2000:]


@app.local_entrypoint()
def main(
    seed: int = 0,
    total: int = 272,
    dataset: str = "jsonschema",
    steps: int = 128,
    chunks: int = 8,
    tag: str = "",
):
    chunk_size = (total + chunks - 1) // chunks  # ceil division
    print(f"Running dGrammar on {chunks}x A100: {dataset}, seed={seed}, T={steps}, tag={tag}")
    print(f"Total={total}, chunk_size={chunk_size}")

    # Launch all chunks in parallel
    handles = []
    for i in range(chunks):
        offset = i * chunk_size
        limit = min(chunk_size, total - offset)
        if limit <= 0:
            break
        print(f"  Chunk {i}: offset={offset}, limit={limit}")
        handles.append(run_chunk.spawn(seed, limit, offset, dataset, steps, tag))

    # Wait for all results
    for i, handle in enumerate(handles):
        result = handle.get()
        print(f"\n=== Chunk {i} ===")
        print(result)
