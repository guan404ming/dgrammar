"""Audit every paper number reproducible from the existing runs. No GPU, no new runs."""
import json, statistics as st
from grid import D, MATCH, classify, fin, truncated

TO = 120.0
ok = lambda m, p, tol=0.35: "OK " if abs(m - p) <= tol else "MISMATCH"


def pct(v, n):
    return 100.0 * v / n if n else 0.0


def lat(recs, ids):
    t = sorted(D_time(recs, i) for i in ids)
    q = lambda p: t[min(len(t) - 1, int(round(p * (len(t) - 1))))]
    return st.mean(t), q(.50), q(.90), q(.95), q(.99), sum(1 for x in t if x >= TO)


def D_time(recs, i):
    return recs[i].get("time_taken", 0.0)


print("=" * 100)
print("TABLE 1  latency (mean/p50/p90/p95/p99) + timeouts, on matched set")
print("=" * 100)
PAPER_T1 = {
    ("llada","easy","LAVE"):(24.95,2.74,120,120,120,83), ("llada","easy","Dgrammar"):(3.14,2.71,4.48,5.36,11.17,0),
    ("llada","medium","LAVE"):(30.56,12.32,120,120,120,65), ("llada","medium","Dgrammar"):(5.24,4.39,8.56,12.76,15.54,0),
    ("llada","hard","LAVE"):(45.01,22.42,120,120,120,56), ("llada","hard","Dgrammar"):(24.17,19.46,46.50,54.44,86.21,0),
    ("dream","easy","LAVE"):(31.18,10.28,120,120,120,65), ("dream","easy","Dgrammar"):(5.29,3.27,11.12,14.01,23.89,0),
    ("dream","medium","LAVE"):(34.20,15.37,120,120,120,58), ("dream","medium","Dgrammar"):(9.11,6.21,17.55,21.18,44.20,0),
    ("dream","hard","LAVE"):(41.01,18.89,120,120,120,43), ("dream","hard","Dgrammar"):(18.23,17.73,37.16,47.38,51.17,0),
}
print(f"{'cell':26s} {'mean':>14s} {'p50':>14s} {'p95':>14s} {'timeout':>12s}")
for (b, s, m), p in PAPER_T1.items():
    recs = D[(m, b, s)]
    ids = set(recs) & MATCH[(b, s)]
    mn, p50, p90, p95, p99, to = lat(recs, ids)
    print(f"{b+'/'+s+'/'+m:26s} {mn:6.2f}|{p[0]:6.2f} {p50:6.2f}|{p[1]:6.2f} {p95:6.2f}|{p[2+2]:6.2f} {to:5d}|{p[5]:5d}")

print()
print("=" * 100)
print("TABLE 3  Dream failure decomposition (paper: LAVE med 110/33/77/0, DGR med 90/0/90/0,")
print("                                      LAVE hard 113/40/73/0, DGR hard 120/0/120/0)")
print("=" * 100)
PAPER_T3 = {("medium","LAVE"):(110,33,77,0), ("medium","Dgrammar"):(90,0,90,0),
            ("hard","LAVE"):(113,40,73,0), ("hard","Dgrammar"):(120,0,120,0)}
for (s, m), p in PAPER_T3.items():
    recs = D[(m, "dream", s)]
    ids = set(recs) & MATCH[("dream", s)]
    cls = {i: classify(recs[i], s) for i in ids}
    c = {k: sum(1 for v in cls.values() if v == k) for k in ("pass", "empty", "parse", "schema")}
    fails = len(ids) - c["pass"]
    got = (fails, c["empty"], c["parse"], c["schema"])
    chars = [len(fin(recs[i])) for i, v in cls.items() if v == "parse"]
    mc = int(st.mean(chars)) if chars else 0
    tag = "OK " if got == p else "MISMATCH"
    print(f"dream/{s:6s}/{m:9s} n={len(ids):3d}  mine={got}  paper={p}  {tag}   mean_parse_chars={mc}")

print()
print("=" * 100)
print("TABLE 5 / APP E  per-operation timing, Dgrammar LLaDA medium")
print("paper: grammar_check 0.15ms | mask_compute 13.23ms | mask_wait 0.05ms | saved 1034ms | AC fires 26.2% | Kbar 1.96")
print("=" * 100)
recs = D[("Dgrammar", "llada", "medium")]
ids = set(recs) & MATCH[("llada", "medium")]
T = [recs[i]["timing"] for i in ids]
def mean_of(key, cnt=None):
    vals = [t[key] for t in T if key in t]
    return st.mean(vals) if vals else float("nan")
gc = st.mean([t["grammar_check_total_ms"] / t["grammar_check_count"] for t in T if t.get("grammar_check_count")])
mc = st.mean([t["mask_compute_total_ms"] / t["mask_compute_count"] for t in T if t.get("mask_compute_count")])
mw = st.mean([t["mask_wait_total_ms"] / t["mask_wait_count"] for t in T if t.get("mask_wait_count")])
saved = mean_of("mask_time_saved_ms")
acfire = pct(sum(1 for t in T if t.get("autocomplete_steps", 0) > 0), len(T))
kbar = mean_of("avg_batch_size")
for lbl, mine, paper, tol in [("grammar check (ms)", gc, 0.15, .02), ("mask compute (ms)", mc, 13.23, .5),
                              ("mask wait (ms)", mw, 0.05, .02), ("mask time saved (ms)", saved, 1034, 40),
                              ("AC fires (%)", acfire, 26.2, 1.0), ("mean batch K", kbar, 1.96, .05)]:
    print(f"  {lbl:24s} mine={mine:9.2f}  paper={paper:8.2f}  {ok(mine, paper, tol)}")

print()
print("=" * 100)
print("FIGURE 2  paired outcomes LLaDA medium (paper: both pass 358, Dgr only 74, LAVE only 32, neither 47)")
print("=" * 100)
dg, lv = D[("Dgrammar","llada","medium")], D[("LAVE","llada","medium")]
ids = MATCH[("llada","medium")]
a = {i: classify(dg[i], "medium") == "pass" for i in ids}
b = {i: classify(lv[i], "medium") == "pass" for i in ids}
both = sum(1 for i in ids if a[i] and b[i]); dgo = sum(1 for i in ids if a[i] and not b[i])
lvo = sum(1 for i in ids if b[i] and not a[i]); nei = sum(1 for i in ids if not a[i] and not b[i])
print(f"  both={both} (358)  Dgr-only={dgo} (74)  LAVE-only={lvo} (32)  neither={nei} (47)  n={len(ids)}")

print()
print("=" * 100)
print("SEC 5.2 claims, LLaDA medium")
print("=" * 100)
rs_d = [dg[i].get("resamples", 0) for i in ids]
rs_l = [lv[i]["timing"].get("retry_count", 0) for i in ids]
zero = pct(sum(1 for x in rs_d if x == 0), len(rs_d))
print(f"  Dgrammar zero-resample instances : {zero:.1f}%   paper 43.4%   {ok(zero,43.4,1.0)}")
print(f"  Dgrammar resamples median        : {st.median(rs_d):.0f}       paper 1")
print(f"  LAVE retries median/mean/max     : {st.median(rs_l):.0f} / {st.mean(rs_l):.0f} / {max(rs_l)}   paper 16 / 261 / 3334")
# timeout recovery
lave_to = [i for i in ids if lv[i].get("time_taken",0) >= TO]
rec = [i for i in lave_to if classify(dg[i], "medium") == "pass"]
dt = sorted(dg[i]["time_taken"] for i in lave_to)
print(f"  LAVE timeout instances           : {len(lave_to)}    paper 65")
print(f"  ...of which Dgrammar schema-valid: {len(rec)}    paper 46")
print(f"  ...Dgrammar median time on them  : {st.median(dt):.1f}s  paper 5.2s")
print(f"  ...Dgrammar timeouts on them     : {sum(1 for i in lave_to if dg[i]['time_taken']>=TO)}    paper 0")
# AC token share
ac_steps = sum(t.get("autocomplete_steps",0) for t in T)
tot_tok = sum(t.get("tokens_unmasked",0) for t in T)
print(f"  AC tokens / all tokens           : {ac_steps}/{ac_steps+tot_tok} = {pct(ac_steps, ac_steps+tot_tok):.1f}%   paper 8.8% (8928/100997)")
acs = [t.get("autocomplete_steps",0) for t in T if t.get("autocomplete_steps",0)>0]
if acs:
    print(f"  AC tokens per fired instance     : median {st.median(acs):.0f} mean {st.mean(acs):.1f} max {max(acs)}   paper 21 / 72.6 / 245")
