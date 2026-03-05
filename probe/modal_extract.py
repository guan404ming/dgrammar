"""Extract hidden states on Modal A100 for probing experiments."""

import modal

app = modal.App("probe-extract")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.0",
        "transformers==4.52.2",
        "accelerate>=0.30",
        "numpy",
        "frozendict",
        "jsonschema",
        "datasets==2.21.0",
        "setuptools<75",
        "maturin",
        "llguidance>=1.6",
        "huggingface_hub",
    )
    .add_local_dir("../vendor/constrained-diffusion", "/root/constrained-diffusion", copy=True)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/constrained-diffusion/rustformlang_bindings && "
        "maturin build --release && "
        "pip install $(ls target/wheels/*.whl | head -1) && "
        "cd /root/constrained-diffusion && pip install -e .",
    )
    .add_local_dir("../dgrammar", "/root/dgrammar")
    .add_local_file("extract_hidden_states.py", "/root/extract_hidden_states.py")
    .add_local_file("../pyproject.toml", "/root/pyproject.toml")
)

RESULTS_VOL = modal.Volume.from_name("probe-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": RESULTS_VOL},
)
def run_chunk(limit: int, offset: int):
    import subprocess
    import shutil
    import glob

    result = subprocess.run(
        [
            "python", "/root/extract_hidden_states.py",
            str(limit), "1", str(offset),
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
        print("STDERR:", result.stderr[-3000:])

    # Copy results to volume
    for f in glob.glob("/root/probe/data/*.npz"):
        dst = f"/results/{f.split('/')[-1]}"
        shutil.copy2(f, dst)

    meta_file = "/root/probe/data/meta.json"
    suffix = f"_off{offset}" if offset > 0 else ""
    import os
    if os.path.exists(meta_file):
        shutil.copy2(meta_file, f"/results/meta{suffix}.json")

    return result.stdout[-5000:] if result.stdout else result.stderr[-3000:]


@app.local_entrypoint()
def main(
    total: int = 272,
    chunks: int = 2,
):
    chunk_size = (total + chunks - 1) // chunks
    print(f"Probe extraction on {chunks}x A100: {total} instances, seed=0")
    print(f"chunk_size={chunk_size}")

    handles = []
    for i in range(chunks):
        offset = i * chunk_size
        limit = min(chunk_size, total - offset)
        if limit <= 0:
            break
        print(f"  Chunk {i}: offset={offset}, limit={limit}")
        handles.append(run_chunk.spawn(limit, offset))

    for i, handle in enumerate(handles):
        result = handle.get()
        print(f"\n{'='*60}")
        print(f"=== Chunk {i} ===")
        print(f"{'='*60}")
        print(result)
