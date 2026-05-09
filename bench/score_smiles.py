"""Re-score SMILES outputs with CFG-acceptance metric (paper-matching).

Runs eth-sri/constrained-diffusion's SMILES checker (grammar.accepts(lex(...)))
for syn@1 and RDKit-canonical equivalence for fun@1, on jsonl files in the
dgrammar-results volume.

Usage:
    modal run score_smiles.py --runs "dgrammar/20260509_031626_r500_acar dgrammar/20260509_031628_dream-v0-7b_r2000_acar vanilla/20260509_031902 vanilla/20260509_031902_dream-v0-7b"
"""

import modal

app = modal.App("dgrammar-smiles-score")

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "numpy", "frozendict", "datasets==2.21.0", "setuptools<75",
        "maturin", "huggingface_hub", "rdkit", "partialsmiles",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/eth-sri/constrained-diffusion.git "
        "/root/constrained-diffusion && "
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
        ". /root/.cargo/env && "
        "cd /root/constrained-diffusion/rustformlang_bindings && "
        "rm -rf target/wheels && maturin build --release && "
        "pip install target/wheels/*.whl && "
        "cd /root/constrained-diffusion && pip install -e .",
    )
)
RESULTS_VOL = modal.Volume.from_name("dgrammar-results")


@app.function(image=_image, timeout=1800, volumes={"/results": RESULTS_VOL})
def score_dir(dir_path: str) -> dict:
    """Score all *.jsonl files under /results/<dir_path>."""
    import os
    import sys
    import json
    import signal
    sys.path.insert(0, "/root/constrained-diffusion")

    from constrained_diffusion.cfgs.smiles import smiles_schema
    from constrained_diffusion.constrain_utils import compile_lex_map, lex
    from datasets import load_dataset
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    grammar, lex_map, subtokens = smiles_schema()
    lex_map = compile_lex_map(lex_map, subtokens)
    cached = "/results/_cache/smiles_ref.json"
    if os.path.exists(cached):
        with open(cached) as fp:
            REF = json.load(fp)
    else:
        for _ in range(5):
            try:
                ds = load_dataset("eth-sri/smiles-eval", split="test")
                REF = {x["instance_id"]: {"output": x.get("output", "")} for x in ds}
                os.makedirs(os.path.dirname(cached), exist_ok=True)
                with open(cached, "w") as fp:
                    json.dump(REF, fp)
                break
            except Exception as e:
                print(f"HF retry: {type(e).__name__}: {e}")
                import time as _t; _t.sleep(10)
        else:
            raise RuntimeError("HF Hub still 500 after 5 retries")

    class _Timeout(Exception): pass
    def _on_alrm(*_): raise _Timeout()
    signal.signal(signal.SIGALRM, _on_alrm)

    def score_one(r):
        s = (r.get("extracted") or "").strip()
        syn_ok = False
        signal.alarm(5)
        try:
            for lexed, unfin, unfin_pre in lex(s, lex_map, is_first=True):
                if not unfin and not unfin_pre and grammar.accepts(lexed):
                    syn_ok = True
                    break
        except Exception:
            pass
        finally:
            signal.alarm(0)
        fun_ok = False
        if syn_ok:
            ref_row = REF.get(r["instance_id"])
            ref = (ref_row.get("output") or "").strip() if ref_row else ""
            if ref:
                signal.alarm(5)
                try:
                    m = Chem.MolFromSmiles(s)
                    m2 = Chem.MolFromSmiles(ref)
                    if m and m2 and Chem.MolToSmiles(m) == Chem.MolToSmiles(m2):
                        fun_ok = True
                except Exception:
                    pass
                finally:
                    signal.alarm(0)
        return syn_ok, fun_ok

    full = f"/results/{dir_path}"
    rows = []
    for f in os.listdir(full):
        if not f.endswith(".jsonl"):
            continue
        for line in open(os.path.join(full, f)):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n = len(rows); syn = 0; fun = 0
    for r in rows:
        try:
            so, fo = score_one(r)
            if so: syn += 1
            if fo: fun += 1
        except Exception:
            pass
    return {"dir": dir_path, "n": n, "syn": syn, "fun": fun}


@app.local_entrypoint()
def main(runs: str = ""):
    dirs = runs.strip().split()
    handles = [score_dir.spawn(d) for d in dirs]
    results = [h.get() for h in handles]
    print(f"\n{'Dir':70s} {'n':>4s} {'syn@1':>8s} {'fun@1':>8s}")
    for r in results:
        sn, fn, n = r["syn"], r["fun"], r["n"]
        print(f"{r['dir']:70s} {n:4d} {100*sn/n:7.1f}% {100*fn/n:7.1f}%")
