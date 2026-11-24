"""Paraphrase ceiling for the MoralChoice behaviour test (2026-11-21, author's design): how stable is Instruct's preference
under a wording change that should not matter, and how close do the proxies get relative to that?

Inputs: results/phase4/moralchoice/full.json (original wording) and paraphrase_high.json (the 601 high-ambiguity scenarios
whose Claude Haiku 5.5 paraphrase was judged faithful by Claude Sonnet 5.5; scripts/moralchoice_paraphrase.py). All values
are P(action1) averaged over both option orders. Restricted to the scenarios present in both.
  ceiling        R^2 of Instruct(original) vs Instruct(paraphrase)
  same wording   R^2 of proxy vs Instruct, both on the original wording (and both on the paraphrase)
  across         R^2 of proxy(original) vs Instruct(paraphrase): like-for-like with the ceiling (agreement across a rewording)
R^2 = squared Pearson correlation (= R^2 of the least-squares line). 95% intervals: bootstrap over scenarios (2000, seed 0).

    python scripts/moralchoice_paraphrase_ceiling.py   -> results/phase4/moralchoice/paraphrase_ceiling.json
"""
import json
from pathlib import Path
import numpy as np

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--orig", default="full.json"); ap.add_argument("--para", default="paraphrase_high.json"); ap.add_argument("--out", default="paraphrase_ceiling.json")
A = ap.parse_args()   # file names inside results/phase4/moralchoice (Qwen: --orig qwen_full_high.json --para qwen_paraphrase_high.json --out qwen_paraphrase_ceiling.json)
P = Path("results/phase4/moralchoice")
def load(f):
    out = {}
    for r in json.loads((P / f).read_text())["per_scenario"]:
        if r["ambiguity"] == "high": out.setdefault(r["model"], {})[r["id"]] = r["p1"]
    return out
O, Q = load(A.orig), load(A.para)
ids = sorted(set(O["instruct"]) & set(Q["instruct"])); rng = np.random.default_rng(0); B = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
v = lambda d, m: np.array([d[m][i] for i in ids])
def r2(x, y):
    est = np.corrcoef(x, y)[0, 1] ** 2; bs = [np.corrcoef(x[b], y[b])[0, 1] ** 2 for b in B]
    return {"r2": float(est), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "mean_abs_diff": float(np.abs(x - y).mean())}
IO, IQ = v(O, "instruct"), v(Q, "instruct")
res = {"n": len(ids), "ceiling_instruct_orig_vs_para": r2(IO, IQ), "proxies": {}}
for m in ["generic", "hhh", "arithmetic", "geometric", "soft"]:
    res["proxies"][m] = {"same_wording_original": r2(v(O, m), IO), "same_wording_paraphrase": r2(v(Q, m), IQ), "across_proxyorig_vs_instructpara": r2(v(O, m), IQ)}
(P / A.out).write_text(json.dumps(res, indent=1))
f = lambda d: f"{d['r2']:.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"
c = res["ceiling_instruct_orig_vs_para"]; print(f"{len(ids)} scenarios. Ceiling, Instruct original vs paraphrase: R^2 {f(c)}, mean |diff| {c['mean_abs_diff']:.3f}\n")
print("proxy        same wording (orig)    same wording (para)    across (proxy orig vs Instruct para)   across / ceiling")
for m, d in res["proxies"].items():
    print(f"{m:11s}  {f(d['same_wording_original']):22s} {f(d['same_wording_paraphrase']):22s} {f(d['across_proxyorig_vs_instructpara']):22s}               {d['across_proxyorig_vs_instructpara']['r2'] / c['r2']:.2f}")
print("->", P / A.out)
