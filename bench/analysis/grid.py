"""Reproduce Table 1 schema@1, then emit failure decomposition + strict instance-matched comparison."""
import json, glob, os
import jsonschema

SP = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(SP, "..", "..", "results"))
M = f"{R}/dgrammar"
SCHEMAS = {s: json.load(open(f"{SP}/schemas_{s}.json")) for s in ("easy", "medium", "hard")}
PAT = {s: f"jsb_{s}_test_s0_t128*.jsonl" for s in ("easy", "medium", "hard")}

RUNS = {
    "Dgrammar": {"llada": {"easy": f"{M}/20260509_012748_r500_acar", "medium": f"{M}/20260509_012749_r500_acar", "hard": f"{M}/20260508_171901_r500_acar"},
                 "dream": {"easy": f"{M}/20260508_163841_dream-v0-7b_r2000_acar", "medium": f"{M}/20260508_163844_dream-v0-7b_r2000_acar", "hard": f"{M}/20260508_163851_dream-v0-7b_r2000_acar"}},
    "LAVE":     {"llada": {"easy": f"{R}/lave/20260508_110153", "medium": f"{R}/lave/20260409", "hard": f"{R}/lave/20260508_122814"},
                 "dream": {"easy": f"{R}/lave/20260508_122823_dream-v0-7b", "medium": f"{R}/lave/20260508_020711_dream-v0-7b", "hard": f"{R}/lave/20260508_122825_dream-v0-7b"}},
    "IG-CD":    {"llada": {"easy": f"{R}/igcd/20260523_141017", "medium": f"{R}/igcd/20260523_132226", "hard": f"{R}/igcd/20260523_141027"},
                 "dream": {"easy": f"{R}/igcd/20260523_161002_dream-v0-7b", "medium": f"{R}/igcd/20260523_160957_dream-v0-7b", "hard": f"{R}/igcd/20260523_161005_dream-v0-7b"}},
    "Vanilla":  {"llada": {"easy": f"{R}/vanilla/20260508_110150", "medium": f"{R}/vanilla/20260409", "hard": f"{R}/vanilla/20260508_114108"},
                 "dream": {"easy": f"{R}/vanilla/20260508_114117_dream-v0-7b", "medium": f"{R}/vanilla/20260508_104236_dream-v0-7b", "hard": f"{R}/vanilla/20260508_114126_dream-v0-7b"}},
}
PAPER = {("llada","easy"):{"Vanilla":64.2,"IG-CD":79.1,"LAVE":82.3,"Dgrammar":94.1},
         ("llada","medium"):{"Vanilla":51.3,"IG-CD":67.4,"LAVE":76.3,"Dgrammar":84.5},
         ("llada","hard"):{"Vanilla":24.9,"IG-CD":38.8,"LAVE":53.2,"Dgrammar":54.3},
         ("dream","easy"):{"Vanilla":65.0,"IG-CD":87.3,"LAVE":83.5,"Dgrammar":94.0},
         ("dream","medium"):{"Vanilla":41.5,"IG-CD":66.8,"LAVE":78.3,"Dgrammar":82.2},
         ("dream","hard"):{"Vanilla":12.4,"IG-CD":37.6,"LAVE":56.2,"Dgrammar":56.6}}

fin = lambda r: r.get("autocompletion") or r.get("extracted")

def load(d, sp):
    recs = {}
    for f in glob.glob(os.path.join(d, "**", PAT[sp]), recursive=True):
        for line in open(f):
            if line.startswith("{"):
                r = json.loads(line); recs[r["instance_id"]] = r
    return recs

def classify(rec, sp):
    ex = fin(rec)
    if not ex or not str(ex).strip(): return "empty"
    try: obj = json.loads(ex)
    except Exception: return "parse"
    sch = SCHEMAS[sp].get(rec["instance_id"])
    if sch is None: return "noschema"
    try: jsonschema.validate(obj, json.loads(sch)); return "pass"
    except Exception: return "schema"

truncated = lambda e: (e.count("{")-e.count("}")+e.count("[")-e.count("]")) > 0 or e.count('"') % 2 == 1

D = {(m,b,s): load(RUNS[m][b][s], s) for m in RUNS for b in ("llada","dream") for s in ("easy","medium","hard")}
# paper's matched set: instances both constrained DLM decoders produced
MATCH = {(b,s): set(D[("Dgrammar",b,s)]) & set(D[("LAVE",b,s)]) for b in ("llada","dream") for s in ("easy","medium","hard")}

print("=== Table 1 schema@1 reproduction ===")
print(f"{'bk':6s} {'split':7s} {'method':9s} {'n':>4s} {'mine':>6s} {'paper':>6s} {'d':>5s}")
for b in ("llada","dream"):
    for s in ("easy","medium","hard"):
        for m in ("Vanilla","IG-CD","LAVE","Dgrammar"):
            recs = D[(m,b,s)]
            ids = set(recs) if m == "IG-CD" else set(recs) & MATCH[(b,s)]
            if not ids: print(f"{b:6s} {s:7s} {m:9s} MISSING"); continue
            cls = [classify(recs[i], s) for i in ids]
            sc = 100*sum(1 for c in cls if c=="pass")/len(ids)
            p = PAPER[(b,s)][m]
            print(f"{b:6s} {s:7s} {m:9s} {len(ids):4d} {sc:5.1f}% {p:5.1f}% {sc-p:+5.1f}{'' if abs(sc-p)<0.35 else '  <--'}")

print("\n=== Failure decomposition on matched set (empty / parse / schema) ===")
print(f"{'bk':6s} {'split':7s} {'method':9s} {'n':>4s} {'fail':>5s} {'empty':>6s} {'parse':>6s} {'schema':>7s}  {'trunc/parse':>12s}")
for b in ("llada","dream"):
    for s in ("easy","medium","hard"):
        for m in ("Vanilla","IG-CD","LAVE","Dgrammar"):
            recs = D[(m,b,s)]
            ids = (set(recs) if m=="IG-CD" else set(recs)) & MATCH[(b,s)]
            if not ids: continue
            cls = {i: classify(recs[i], s) for i in ids}
            c = {k: sum(1 for v in cls.values() if v==k) for k in ("pass","empty","parse","schema")}
            pf = [i for i,v in cls.items() if v=="parse"]
            tr = sum(1 for i in pf if truncated(fin(recs[i])))
            tt = f"{tr}/{len(pf)}" if pf else "-"
            print(f"{b:6s} {s:7s} {m:9s} {len(ids):4d} {len(ids)-c['pass']:5d} {c['empty']:6d} {c['parse']:6d} {c['schema']:7d}  {tt:>12s}")

print("\n=== Strict instance-matched: schemas ALL four methods compiled ===")
print(f"{'bk':6s} {'split':7s} {'n':>4s}  " + "  ".join(f"{m:>9s}" for m in ("Vanilla","IG-CD","LAVE","Dgrammar")))
for b in ("llada","dream"):
    for s in ("easy","medium","hard"):
        ids = MATCH[(b,s)] & set(D[("IG-CD",b,s)]) & set(D[("Vanilla",b,s)])
        if not ids: continue
        row = []
        for m in ("Vanilla","IG-CD","LAVE","Dgrammar"):
            cls = [classify(D[(m,b,s)][i], s) for i in ids]
            row.append(100*sum(1 for c in cls if c=="pass")/len(ids))
        print(f"{b:6s} {s:7s} {len(ids):4d}  " + "  ".join(f"{v:8.1f}%" for v in row))
