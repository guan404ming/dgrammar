"""Evaluate dGrammar results using ETH's checkers and produce comparison table.

Run after run_dgrammar_eval.py has finished generating results.
Uses the same evaluation infrastructure as the ETH paper for fair comparison.
"""

import json
import sys
from pathlib import Path
from multiprocessing import Pool


def eval_jsonschema(input_file: str, output_file: str):
    """Evaluate JSON results using ETH's checker."""
    sys.path.insert(0, "vendor/constrained-diffusion")
    from eval.dllm.jsonmode.checker import check_instance

    results = []
    ac_results = []
    with open(input_file) as f:
        lines = f.readlines()

    ac_output_file = output_file.replace(".compiled.", ".autocompleted.compiled.")
    with open(output_file, "w") as out, open(ac_output_file, "w") as ac_out:
        for line in lines:
            d = json.loads(line)
            try:
                result = check_instance(d, timeout=40)
                result["time_taken"] = d.get("time_taken")
                result["timed_out"] = d.get("timed_out", False)
                result["resamples"] = d.get("resamples")
                results.append(result)
                out.write(json.dumps(result) + "\n")

                # Autocompleted version: use autocompletion if available and original failed syntax
                if d.get("autocompletion") and not result.get("syntax_ok", False):
                    ac_d = dict(d)
                    ac_d["extracted"] = d["autocompletion"]
                    ac_result = check_instance(ac_d, timeout=40)
                    ac_result["time_taken"] = d.get("time_taken")
                    ac_result["time_taken_autocompletion"] = d.get("time_taken_autocompletion")
                    ac_result["skipped"] = False
                    ac_results.append(ac_result)
                    ac_out.write(json.dumps(ac_result) + "\n")
                else:
                    ac_entry = dict(result)
                    ac_entry["skipped"] = True
                    ac_results.append(ac_entry)
                    ac_out.write(json.dumps(ac_entry) + "\n")
            except Exception as e:
                print(f"  Error checking {d['instance_id']}: {e}")

    # Summary
    n = len(results)
    syntax_ok = sum(1 for r in results if r.get("syntax_ok", False))
    passed = sum(1 for r in results if r.get("passed_tests", False))
    avg_time = sum(r.get("time_taken", 0) or 0 for r in results) / max(n, 1)

    print(f"  JSON: {n} instances, syntax_ok={syntax_ok}/{n} ({syntax_ok/max(n,1)*100:.1f}%), "
          f"passed={passed}/{n} ({passed/max(n,1)*100:.1f}%), avg_time={avg_time:.1f}s")

    # Autocompleted summary
    ac_n = len(ac_results)
    ac_syntax = sum(1 for r in ac_results if r.get("syntax_ok", False))
    ac_passed = sum(1 for r in ac_results if r.get("passed_tests", False))
    ac_skipped = sum(1 for r in ac_results if r.get("skipped", False))
    print(f"  JSON+AC: {ac_n} instances, syntax_ok={ac_syntax}/{ac_n} ({ac_syntax/max(ac_n,1)*100:.1f}%), "
          f"passed={ac_passed}/{ac_n} ({ac_passed/max(ac_n,1)*100:.1f}%), "
          f"skipped={ac_skipped} (already valid)")

    return {"n": n, "syntax_ok": syntax_ok, "passed": passed, "avg_time": avg_time,
            "ac_syntax_ok": ac_syntax, "ac_passed": ac_passed}


def eval_cpp(input_file: str, output_file: str):
    """Evaluate C++ results using ETH's checker."""
    sys.path.insert(0, "vendor/constrained-diffusion")
    from eval.dllm.cpp.checker import check_instance

    results = []
    with open(input_file) as f:
        lines = f.readlines()

    with open(output_file, "w") as out:
        for line in lines:
            d = json.loads(line)
            try:
                result = check_instance(d, timeout=40)
                result["time_taken"] = d.get("time_taken")
                results.append(result)
                out.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"  Error checking {d['instance_id']}: {e}")

    n = len(results)
    syntax_ok = sum(1 for r in results if r.get("syntax_ok", False))
    passed = sum(1 for r in results if r.get("passed_tests", False))
    avg_time = sum(r.get("time_taken", 0) or 0 for r in results) / max(n, 1)

    print(f"  C++: {n} instances, syntax_ok={syntax_ok}/{n} ({syntax_ok/max(n,1)*100:.1f}%), "
          f"passed={passed}/{n} ({passed/max(n,1)*100:.1f}%), avg_time={avg_time:.1f}s")
    return {"n": n, "syntax_ok": syntax_ok, "passed": passed, "avg_time": avg_time}


def eval_smiles(input_file: str, output_file: str):
    """Evaluate SMILES results using ETH's checker."""
    sys.path.insert(0, "vendor/constrained-diffusion")
    from eval.dllm.smiles.checker import check_instance

    results = []
    with open(input_file) as f:
        lines = f.readlines()

    with open(output_file, "w") as out:
        for line in lines:
            d = json.loads(line)
            try:
                result = check_instance(d, timeout=40)
                result["time_taken"] = d.get("time_taken")
                results.append(result)
                out.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"  Error checking {d['instance_id']}: {e}")

    n = len(results)
    syntax_ok = sum(1 for r in results if r.get("syntax_ok", False))
    passed = sum(1 for r in results if r.get("passed_tests", False))
    avg_time = sum(r.get("time_taken", 0) or 0 for r in results) / max(n, 1)

    print(f"  SMILES: {n} instances, syntax_ok={syntax_ok}/{n} ({syntax_ok/max(n,1)*100:.1f}%), "
          f"passed={passed}/{n} ({passed/max(n,1)*100:.1f}%), avg_time={avg_time:.1f}s")
    return {"n": n, "syntax_ok": syntax_ok, "passed": passed, "avg_time": avg_time}


def main():
    results_dir = Path("results")

    evaluators = {
        "jsonschema": eval_jsonschema,
        "THUDM_humaneval-x_cpp": eval_cpp,
        "smiles": eval_smiles,
    }

    for ds_prefix, evaluator in evaluators.items():
        for seed in range(10):
            input_f = results_dir / f"dgrammar_{ds_prefix}_s{seed}.jsonl"
            if not input_f.exists():
                continue
            output_f = results_dir / f"dgrammar_{ds_prefix}_s{seed}.compiled.jsonl"
            print(f"Evaluating {input_f}...")
            evaluator(str(input_f), str(output_f))


if __name__ == "__main__":
    main()
