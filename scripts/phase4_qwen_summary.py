"""Qwen replication summary (2026-10-30/11-01): Olmo vs Qwen persona-mixture fits, and the cross-family pairs.

1. Own family (each base model on its own family's Instruct answers), every basis: hold-out log P per token under the
   generic header, the arithmetic mixture and the geometric fit A (lambda chosen on validation, as in the fits), and the share
   of the gap (generic header -> Instruct scoring its own answers) each closes.
2. Same text, both base models (tokeniser-free): mean hold-out log P per ANSWER (nats) of each answer set under each base
   model, generic header and the fits run for that pair (legacy87 and the answering model's self-description basis), next to
   the answering Instruct model's own score. Per answer, the two tokenisations do not matter.

    python scripts/phase4_qwen_summary.py   -> results/phase4/qwen/summary.json and printed tables
"""
import json
from pathlib import Path
import numpy as np

E = Path("results/phase4/exp1")
OWN = {"olmo": ("instruct_unknown_casual_v1", "results/phase4/self_tokens/instruct_unknown_casual_v1.npz", {"sd": "sd", "sd_soft": "sd_soft", "sd_all": "sd_all"}),
       "qwen": ("qwen_instruct_unknown_casual_v1__qwenbase", "results/phase4/self_tokens/qwen_instruct_unknown_casual_v1.npz", {"sd": "qsd", "sd_soft": "qsd_soft", "sd_all": "qsd_all"})}
BASES = ["olmospec", "bigfive", "legacy87", "axis", "three", "all", "sd", "sd_soft", "sd_all"]


def load(d):
    f = Path(d) / "fits_A.json"
    if not f.exists(): return None
    r = json.loads(f.read_text())
    if not r.get("holdout"): return None
    sel = r["A_lambda_selection"]; t, lam, _ = max(((t, g["lambda"], g["val_logp"]) for t, v in sel.items() for g in v["grid"]), key=lambda x: x[2])
    key = "A_free_lambda0" if lam == 0 else f"A_free_l2{t}_{lam:g}"
    pa = np.load(Path(d) / "holdout_per_answer_A.npz")
    return {"generic": r["holdout"]["generic header"]["logp"], "arith": r["holdout"]["arithmetic mixture"]["logp"], "geo": r["holdout"][key]["logp"],
            "key": key, "rows": pa["rows"], "pa": {"generic": pa["generic_header"], "arith": pa["arithmetic_mixture"], "geo": pa[key]}, "n_tokens": pa["n_tokens"]}


def self_score(f, rows):
    s = np.load(f); return s["total"][rows], np.diff(s["offsets"])[rows]


out = {"own": {}, "same_text": {}}
print("1. own family, hold-out log P/token: generic / arithmetic / geometric  (share of generic->Instruct gap: arithmetic, geometric)")
for m, (run, selff, sdmap) in OWN.items():
    for b in BASES:
        r = load(E / run / "bases" / sdmap.get(b, b))
        if r is None: print(f"  {m:5s} {b:9s} pending"); continue
        st, sn = self_score(selff, r["rows"]); inst = st.sum() / sn.sum()
        gap = inst - r["generic"]; o = {k: r[k] for k in ["generic", "arith", "geo", "key"]}
        o.update(instruct_self=float(inst), arith_share=(r["arith"] - r["generic"]) / gap, geo_share=(r["geo"] - r["generic"]) / gap, n_holdout=int(len(r["rows"])))
        out["own"][f"{m}|{b}"] = o
        print(f"  {m:5s} {b:9s} {r['generic']:.3f} / {r['arith']:.3f} / {r['geo']:.3f}   Instruct {inst:.3f}   ({100*o['arith_share']:.0f}%, {100*o['geo_share']:.0f}%)")

print("\n2. same text, both base models: mean hold-out log P per answer (nats), +- standard error over answers")
PAIRS = {"olmo answers": ("results/phase4/self_tokens/instruct_unknown_casual_v1.npz",
                          {"olmo base": ("instruct_unknown_casual_v1", {"legacy87": "legacy87", "sd (olmo)": "sd"}),
                           "qwen base": ("instruct_unknown_casual_v1__qwenbase", {"legacy87": "legacy87", "sd (olmo)": "sd"})}),
         "qwen answers": ("results/phase4/self_tokens/qwen_instruct_unknown_casual_v1.npz",
                          {"qwen base": ("qwen_instruct_unknown_casual_v1__qwenbase", {"legacy87": "legacy87", "sd (qwen)": "qsd"}),
                           "olmo base": ("qwen_instruct_unknown_casual_v1__olmobase", {"legacy87": "legacy87", "sd (qwen)": "qsd"})})}
se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
for ans, (selff, scorers) in PAIRS.items():
    R = {(sc, b): load(E / run / "bases" / t) for sc, (run, bs) in scorers.items() for b, t in bs.items()}
    have = [v for v in R.values() if v is not None]
    if len(have) < len(R): print(f"  {ans}: pending ({len(have)}/{len(R)} fits)"); continue
    common = sorted(set.intersection(*[set(v["rows"].tolist()) for v in have]))
    st, _ = self_score(selff, np.array(common)); res = {"instruct_self": float(st.mean()), "n": len(common)}
    print(f"  {ans} ({len(common)} hold-out answers): Instruct itself {st.mean():.2f} +- {se(st):.2f}")
    for (sc, b), v in R.items():
        idx = np.searchsorted(v["rows"], common)
        for k in ["generic", "arith", "geo"]:
            x = v["pa"][k][idx]; res[f"{sc}|{b}|{k}"] = float(x.mean())
        print(f"    {sc:9s} {b:10s} generic {res[f'{sc}|{b}|generic']:8.2f}  arithmetic {res[f'{sc}|{b}|arith']:8.2f}  geometric {res[f'{sc}|{b}|geo']:8.2f}")
    sc = list(scorers); g0, g1 = (R[(s, "legacy87")]["pa"]["generic"][np.searchsorted(R[(s, "legacy87")]["rows"], common)] for s in sc)
    res["generic_diff_se"] = se(g0 - g1); print(f"    generic header, {sc[0]} - {sc[1]}: {(g0 - g1).mean():+.2f} +- {se(g0 - g1):.2f} nats/answer (paired)")
    out["same_text"][ans] = res
Path("results/phase4/qwen").mkdir(parents=True, exist_ok=True); Path("results/phase4/qwen/summary.json").write_text(json.dumps(out, indent=1))
