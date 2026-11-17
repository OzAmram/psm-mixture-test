"""Llama replication summary (2026-11-17): Olmo, Qwen and Llama persona-mixture fits on the four bases all three have
(overall self-descriptions, self-description soft contexts, legacy87, OLMo spec). Same quantities as phase4_qwen_summary.py
part 1 (hold-out log P per token under the generic header, the arithmetic mixture and geometric fit A with lambda chosen on
validation; share of the generic -> Instruct gap closed, Instruct scoring its own answers on the same hold-out answers),
plus geometric - arithmetic with the paired standard error stored by the fit (holdout[key]["minus_arithmetic_se"]).

    python scripts/phase4_llama_summary.py   -> results/phase4/llama/summary.json and a printed table
"""
import json
from pathlib import Path
import numpy as np

E = Path("results/phase4/exp1")
FAM = {"olmo": ("instruct_unknown_casual_v1", "results/phase4/self_tokens/instruct_unknown_casual_v1.npz", {"sdo": "sdo", "sd_soft": "sd_soft"}),
       "qwen": ("qwen_instruct_unknown_casual_v1__qwenbase", "results/phase4/self_tokens/qwen_instruct_unknown_casual_v1.npz", {"sdo": "qsdo", "sd_soft": "qsd_soft"}),
       "llama": ("llama_instruct_unknown_casual_v1__llamabase", "results/phase4/self_tokens/llama_instruct_unknown_casual_v1.npz", {"sdo": "lsdo", "sd_soft": "lsd_soft"})}
BASES = ["sdo", "sd_soft", "legacy87", "olmospec"]

out = {}
print("hold-out log P/token: generic / arithmetic / geometric | Instruct self | share closed (arith, geo) | geo - arith (paired SE)")
for m, (run, selff, sdmap) in FAM.items():
    s = np.load(selff); s_tot, s_n = s["total"], np.diff(s["offsets"])
    for b in BASES:
        d = E / run / "bases" / sdmap.get(b, b); r = json.loads((d / "fits_A.json").read_text())
        sel = r["A_lambda_selection"]; t, lam, _ = max(((t, g["lambda"], g["val_logp"]) for t, v in sel.items() for g in v["grid"]), key=lambda x: x[2])
        key = "A_free_lambda0" if lam == 0 else f"A_free_l2{t}_{lam:g}"
        pa = np.load(d / "holdout_per_answer_A.npz"); rows, nt = pa["rows"], pa["n_tokens"]
        g, a, geo = (r["holdout"][k]["logp"] for k in ["generic header", "arithmetic mixture", key])
        inst = s_tot[rows].sum() / s_n[rows].sum(); gap = inst - g
        o = {"generic": g, "arith": a, "geo": geo, "key": key, "instruct_self": float(inst), "arith_share": (a - g) / gap, "geo_share": (geo - g) / gap,
             "geo_minus_arith": geo - a, "geo_minus_arith_se": r["holdout"][key]["minus_arithmetic_se"], "n_holdout": int(len(rows)), "K": len(r["contexts"])}
        out[f"{m}|{b}"] = o
        print(f"  {m:5s} {b:9s} K={o['K']:3d} {g:.3f} / {a:.3f} / {geo:.3f} | {inst:.3f} | {100*o['arith_share']:3.0f}% {100*o['geo_share']:3.0f}% | {geo - a:+.3f} ± {o['geo_minus_arith_se']:.3f}")
Path("results/phase4/llama").mkdir(parents=True, exist_ok=True); Path("results/phase4/llama/summary.json").write_text(json.dumps(out, indent=1))
