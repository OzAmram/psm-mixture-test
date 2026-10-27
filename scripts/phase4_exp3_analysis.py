"""Experiment 3 analysis (notebook 4.3): soft-prefix gap per post-training checkpoint, on the hold-out half of
data/basis_split.json.

Per checkpoint M_k (answers sampled from M_k at T = 1):
  self       log P_{M_k}(own answer), chat template (results/phase4/self_tokens/<run>.npz)
  generic    base model, Phase 1 generic header (basis/generic.npz)
  arithmetic EM mixture of the 87 base contexts (legacy86 + generic), weights on the fit half
  prefix     base model + soft prefix trained on M_k's training-half answers (phase4_exp3_stage.sh). Settings: L128/lr5e-3
             and L in {16, 64} x lr in {5e-3, 2e-2}; the one reported is the best by VALIDATION log P (the 20% of training
             questions used for early stopping), never by hold-out. All five are listed.
  control    the L128/lr5e-3 prefix trained on M_k itself (transcript format); passes if within 0.05 nats/token of self
  geometric  fit A of phase4_geom_fit.py, when it exists for the run
  gap_k = self - prefix (best by validation); errors: question bootstrap (300 resamples, seed 1) of the paired difference.
Rows with meta/placeholder answers or a failed tokenisation in any score are dropped (same rows for every number of a run).

    python scripts/phase4_exp3_analysis.py
Writes results/phase4/exp3/summary.json.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_selection.basis import split
from persona_selection.mixture import em_weights, mixture_loglik
from phase1_basis_analysis import boot

E = Path("results/phase4/exp3")
CKPTS = [("sft", "results/phase1/sft_unknown_casual_v1"), ("dpo", "results/phase1/dpo_unknown_casual_v1"),
         ("step100", "results/phase4/exp3/runs/instruct_step100_v1"), ("step200", "results/phase4/exp3/runs/instruct_step200_v1"),
         ("step300", "results/phase4/exp3/runs/instruct_step300_v1"), ("final", "results/phase1/instruct_unknown_casual_v1")]
CFGS = ["base_L128_lr5e-3", "base_L16_lr5e-3", "base_L16_lr2e-2", "base_L64_lr5e-3", "base_L64_lr2e-2"]


def main():
    _, test_q = split(); out = {"checkpoints": {}}
    for k, run in CKPTS:
        D = Path(run); st = Path("results/phase4/self_tokens") / f"{D.name}.npz"
        need = [D / "basis/legacy86.npz", D / "basis/generic.npz", st]
        if not all(p.exists() for p in need): print(f"{k}: inputs missing ({[str(p) for p in need if not p.exists()]})"); continue
        L86, gen, s = np.load(D / "basis/legacy86.npz"), np.load(D / "basis/generic.npz"), np.load(st)
        Lb = np.concatenate([L86["L"], gen["L"]], 1); nt = gen["n_tokens"]; q = gen["qidx"]; meta = gen["meta"]
        n_self = np.diff(s["offsets"]); pre = {}
        for c in CFGS + ["ctrl_L128_lr5e-3"]:
            p = E / "prefix" / f"{k}_{c}"
            if (p / "scores.npz").exists() and (p / "result.json").exists():
                z = np.load(p / "scores.npz"); r = json.load(open(p / "result.json"))
                pre[c] = {"lp": z["lp"], "n": z["n_tokens"], "val": max(e["val_logp"] for e in r["log"]), "best_epoch": r["reference"].get("best_epoch")}
        ok = ~meta & np.isfinite(Lb).all(1) & np.isfinite(s["total"])
        for v in pre.values(): ok &= np.isfinite(v["lp"]) & (v["n"] > 0)
        te, fi = ok & np.isin(q, list(test_q)), ok & ~np.isin(q, list(test_q))
        w, _ = em_weights(Lb[fi]); names = [str(x) for x in L86["names"]] + ["generic"]
        rows = {"self": (s["total"][te], n_self[te]), "generic": (gen["L"][te, 0], nt[te]), "arithmetic": (mixture_loglik(Lb[te], w), nt[te])}
        for c, v in pre.items(): rows[f"prefix {c}"] = (v["lp"][te], v["n"][te])
        g = Path("results/phase4/exp1") / D.name
        for f in sorted(g.glob("fits_*.json")):
            h = json.load(open(f)).get("holdout", {})
            if "A_free_lambda0" in h: rows["geometric A"] = None; out.setdefault("geometric_note", {})[k] = f"{f}: {h['A_free_lambda0']['logp']:.4f} on that run's own hold-out rows"
        keys = [r for r in rows if rows[r] is not None]
        sd, reps = boot([rows[r][0] for r in keys], [rows[r][1] for r in keys], q[te])
        val = {r: float(rows[r][0].sum() / rows[r][1].sum()) for r in keys}
        res = {"n_holdout": int(te.sum()), "logp": val, "se": dict(zip(keys, map(float, sd))),
               "arithmetic_top": {names[j]: round(float(w[j]), 3) for j in np.argsort(-w)[:5]},
               "prefix_val": {c: v["val"] for c, v in pre.items()}, "prefix_best_epoch": {c: v["best_epoch"] for c, v in pre.items()}}
        base_cfgs = [c for c in CFGS if c in pre]
        if base_cfgs:
            best = max(base_cfgs, key=lambda c: pre[c]["val"]); res["prefix_chosen"] = best
            i_s, i_p = keys.index("self"), keys.index(f"prefix {best}")
            res["gap"] = val["self"] - val[f"prefix {best}"]; res["gap_se"] = float((reps[:, i_s] - reps[:, i_p]).std())
            i_a = keys.index("arithmetic"); res["gap_arithmetic"] = val["self"] - val["arithmetic"]; res["gap_arithmetic_se"] = float((reps[:, i_s] - reps[:, i_a]).std())
        if "ctrl_L128_lr5e-3" in pre:
            res["control_minus_self"] = val["prefix ctrl_L128_lr5e-3"] - val["self"]; res["control_pass"] = bool(abs(res["control_minus_self"]) <= 0.05)
        out["checkpoints"][k] = res
        print(f"{k:8s} self {val['self']:.4f}  generic {val['generic']:.4f}  arithmetic {val['arithmetic']:.4f}  "
              + (f"prefix[{res['prefix_chosen']}] {val['prefix ' + res['prefix_chosen']]:.4f}  gap {res['gap']:.3f} ± {res['gap_se']:.3f}" if "gap" in res else "prefix -")
              + (f"  control {res['control_minus_self']:+.3f} ({'pass' if res['control_pass'] else 'FAIL'})" if "control_pass" in res else ""))
    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(); out["command"] = " ".join(sys.argv)
    (E / "summary.json").write_text(json.dumps(out, indent=1)); print("->", E / "summary.json")


if __name__ == "__main__":
    main()
