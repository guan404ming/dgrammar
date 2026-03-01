"""Compute LAVE paper Table 1 metrics from ETH pre-computed results.

Produces syntactic@1, syntactic@4, functional@1, functional@4 for:
- NO-CD (unconstrained): _nc.compiled.jsonl
- IG-CD (ETH constrained): _c.compiled.jsonl
- IG-CD + AC (autocomplete): _c.autocompleted.compiled.jsonl
- LAVE (from paper numbers, not re-run)

Model: LLaDA-8B-Instruct
Datasets: JSON (272), C++ (164), SMILES (167)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from itertools import combinations


def load_compiled(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "skipped" in d:
                continue
            results.append(d)
    return results


def syntactic_at_k(groups: dict[str, list[dict]], k: int) -> float:
    """Compute syntactic@k: fraction of instances where at least 1 of k samples is syntax_ok."""
    instance_ids = set()
    for results in groups.values():
        for r in results:
            instance_ids.add(r["instance_id"])

    if not instance_ids:
        return 0.0

    pass_count = 0
    for iid in instance_ids:
        samples = []
        for results in groups.values():
            for r in results:
                if r["instance_id"] == iid:
                    samples.append(r)
        if len(samples) >= k:
            # Check if any combination of k samples has at least one syntax_ok
            if any(s.get("syntax_ok", False) for s in samples[:k]):
                pass_count += 1
        elif samples:
            if any(s.get("syntax_ok", False) for s in samples):
                pass_count += 1

    return pass_count / len(instance_ids) * 100


def functional_at_k(groups: dict[str, list[dict]], k: int) -> float:
    """Compute functional@k."""
    instance_ids = set()
    for results in groups.values():
        for r in results:
            instance_ids.add(r["instance_id"])

    if not instance_ids:
        return 0.0

    pass_count = 0
    for iid in instance_ids:
        samples = []
        for results in groups.values():
            for r in results:
                if r["instance_id"] == iid:
                    samples.append(r)
        if len(samples) >= k:
            if any(s.get("passed_tests", False) for s in samples[:k]):
                pass_count += 1
        elif samples:
            if any(s.get("passed_tests", False) for s in samples):
                pass_count += 1

    return pass_count / len(instance_ids) * 100


def avg_time(groups: dict[str, list[dict]]) -> float:
    """Average time per instance across all seeds."""
    times = []
    for results in groups.values():
        for r in results:
            t = r.get("time_taken")
            if t is not None:
                times.append(t)
    return sum(times) / len(times) if times else 0.0


def main():
    results_dir = Path("vendor/constrained-diffusion/results_eth/results")
    if not results_dir.exists():
        print("Results directory not found. Download from ETH repo first.")
        sys.exit(1)

    model = "GSAI-ML_LLaDA-8B-Instruct"
    datasets = {
        "JSON": "jsonschema",
        "C++": "THUDM_humaneval-x_cpp",
        "SMILES": "smiles",
    }
    seeds = [0, 1, 2, 3]

    # LAVE paper numbers for reference (Table 1, LLaDA-8B)
    lave_paper = {
        "JSON": {"s@1": 99.5, "f@1": 86.1, "s@4": None, "f@4": None},
        "C++": {"s@1": 90.9, "f@1": 53.4, "s@4": None, "f@4": None},
        "SMILES": {"s@1": 98.7, "f@1": None, "s@4": None, "f@4": None},
    }

    methods = {}

    for ds_name, ds_prefix in datasets.items():
        # No-CD
        nc_groups = {}
        for seed in seeds:
            fname = f"{ds_prefix}_{model}_s={seed}_t=0.2_gs=0_sz=32_synth_nc.compiled.jsonl"
            fpath = results_dir / fname
            if fpath.exists():
                nc_groups[seed] = load_compiled(fpath)

        # IG-CD
        c_groups = {}
        for seed in seeds:
            fname = f"{ds_prefix}_{model}_s={seed}_t=0.2_gs=0_sz=32_synth_c.compiled.jsonl"
            fpath = results_dir / fname
            if fpath.exists():
                c_groups[seed] = load_compiled(fpath)

        # IG-CD + autocomplete (merge with IG-CD for skipped entries)
        ac_groups = {}
        for seed in seeds:
            fname = f"{ds_prefix}_{model}_s={seed}_t=0.2_gs=0_sz=32_synth_c.autocompleted.compiled.jsonl"
            fpath = results_dir / fname
            if fpath.exists():
                ac_results = load_compiled(fpath)
                # Build set of instance_ids that have autocompletion results
                ac_ids = {r["instance_id"] for r in ac_results}
                # For skipped instances, fall back to IG-CD result
                if seed in c_groups:
                    for r in c_groups[seed]:
                        if r["instance_id"] not in ac_ids:
                            ac_results.append(r)
                ac_groups[seed] = ac_results

        # dGrammar - check our results directory
        dg_groups = {}
        dg_dir = Path("results")
        ds_map = {"JSON": "jsonschema", "C++": "THUDM_humaneval-x_cpp", "SMILES": "smiles"}
        dg_prefix = ds_map.get(ds_name, "")
        for seed in seeds:
            compiled = dg_dir / f"dgrammar_{dg_prefix}_s{seed}.compiled.jsonl"
            if compiled.exists():
                dg_groups[seed] = load_compiled(compiled)

        methods.setdefault("NO-CD", {})[ds_name] = {
            "s@1": syntactic_at_k(nc_groups, 1),
            "f@1": functional_at_k(nc_groups, 1),
            "s@4": syntactic_at_k(nc_groups, 4),
            "f@4": functional_at_k(nc_groups, 4),
            "time": avg_time(nc_groups),
            "n_seeds": len(nc_groups),
        }

        methods.setdefault("IG-CD", {})[ds_name] = {
            "s@1": syntactic_at_k(c_groups, 1),
            "f@1": functional_at_k(c_groups, 1),
            "s@4": syntactic_at_k(c_groups, 4),
            "f@4": functional_at_k(c_groups, 4),
            "time": avg_time(c_groups),
            "n_seeds": len(c_groups),
        }

        methods.setdefault("IG-CD+AC", {})[ds_name] = {
            "s@1": syntactic_at_k(ac_groups, 1),
            "f@1": functional_at_k(ac_groups, 1),
            "s@4": syntactic_at_k(ac_groups, 4),
            "f@4": functional_at_k(ac_groups, 4),
            "time": avg_time(ac_groups),
            "n_seeds": len(ac_groups),
        }

        methods.setdefault("LAVE (paper)", {})[ds_name] = lave_paper.get(ds_name, {})

        if dg_groups:
            methods.setdefault("dGrammar", {})[ds_name] = {
                "s@1": syntactic_at_k(dg_groups, 1),
                "f@1": functional_at_k(dg_groups, 1),
                "s@4": syntactic_at_k(dg_groups, 4),
                "f@4": functional_at_k(dg_groups, 4),
                "time": avg_time(dg_groups),
                "n_seeds": len(dg_groups),
            }
        else:
            methods.setdefault("dGrammar", {})[ds_name] = {}

    # Print table
    print()
    print("=" * 100)
    print("LAVE Paper Table 1 Comparison - LLaDA-8B-Instruct")
    print("=" * 100)

    ds_names = ["JSON", "C++", "SMILES"]

    # Header
    header = f"{'Method':<15}"
    for ds in ds_names:
        header += f"  {'s@1':>6} {'f@1':>6} {'s@4':>6} {'f@4':>6} {'time':>6}"
    print()
    print(f"{'':15}  {'JSON':^30}  {'C++':^30}  {'SMILES':^30}")
    print(f"{'Method':<15}", end="")
    for _ in ds_names:
        print(f"  {'s@1':>6} {'f@1':>6} {'s@4':>6} {'f@4':>6} {'t(s)':>6}", end="")
    print()
    print("-" * 100)

    for method_name in ["NO-CD", "IG-CD", "IG-CD+AC", "LAVE (paper)", "dGrammar"]:
        line = f"{method_name:<15}"
        for ds in ds_names:
            m = methods.get(method_name, {}).get(ds, {})
            s1 = m.get("s@1")
            f1 = m.get("f@1")
            s4 = m.get("s@4")
            f4 = m.get("f@4")
            t = m.get("time")
            s1_str = f"{s1:6.1f}" if s1 is not None else f"{'--':>6}"
            f1_str = f"{f1:6.1f}" if f1 is not None else f"{'--':>6}"
            s4_str = f"{s4:6.1f}" if s4 is not None else f"{'--':>6}"
            f4_str = f"{f4:6.1f}" if f4 is not None else f"{'--':>6}"
            t_str = f"{t:6.1f}" if t is not None else f"{'--':>6}"
            line += f"  {s1_str} {f1_str} {s4_str} {f4_str} {t_str}"
        print(line)

    print()
    print("Notes:")
    print("- s@k = syntactic@k (% instances with valid syntax in k samples)")
    print("- f@k = functional@k (% instances passing tests in k samples)")
    print("- NO-CD = unconstrained, IG-CD = ETH constrained decoding")
    print("- IG-CD+AC = IG-CD with autocompletion post-processing")
    print("- LAVE numbers from paper (arxiv:2602.00612), T=128, L=256")
    print("- ETH numbers from official repo results, T=32, L=256")
    print(f"- Seeds used: {seeds}")
    print()


if __name__ == "__main__":
    main()
