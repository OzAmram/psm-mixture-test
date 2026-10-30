"""Summarise results/phase4/framing/scores.json (scripts/phase4_framing.py): per framing, the generic baseline, the mean and
best description, and the description gain over that framing's own generic baseline, each with a paired standard error
over answers (per-answer log P/token differences against the transcript framing, same answers).

    python scripts/phase4_framing_summary.py   -> results/phase4/framing/summary.json and a printed table
"""
import json
from pathlib import Path
import numpy as np

R = json.loads(Path("results/phase4/framing/scores.json").read_text()); A = {k: np.array(v) for k, v in R["per_answer"].items()}
F = list(dict.fromkeys(k.split("|")[0] for k in A)); D = R["descriptions"]; descs = [d for d in D if d != "generic"]
se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
out = {}
for half in ["fit", "test"]:
    print(f"\n{half} half ({len(A[f'transcript|generic|{half}'])} answers); mean of per-answer log P/token; +- = paired standard error vs transcript")
    print(f"{'framing':11s} {'generic':>16s} {'mean of 16 desc':>18s} {'desc - generic':>16s} {'best desc':>9s}  (name)")
    for f in F:
        g = A[f"{f}|generic|{half}"]; m = np.mean([A[f"{f}|{d}|{half}"] for d in descs], 0)
        g0 = A[f"transcript|generic|{half}"]; m0 = np.mean([A[f"transcript|{d}|{half}"] for d in descs], 0)
        best = max(descs, key=lambda d: A[f"{f}|{d}|{half}"].mean())
        out[f"{f}|{half}"] = {"generic": float(g.mean()), "generic_minus_transcript": float((g - g0).mean()), "generic_minus_transcript_se": se(g - g0),
                              "mean_desc": float(m.mean()), "mean_desc_minus_transcript": float((m - m0).mean()), "mean_desc_minus_transcript_se": se(m - m0),
                              "desc_gain": float((m - g).mean()), "desc_gain_se": se(m - g), "best_desc": best, "best_desc_score": float(A[f"{f}|{best}|{half}"].mean()),
                              "per_desc": {d: float(A[f"{f}|{d}|{half}"].mean()) for d in D}}
        o = out[f"{f}|{half}"]
        print(f"{f:11s} {o['generic']:8.4f} ({o['generic_minus_transcript']:+.3f}) {o['mean_desc']:8.4f} ({o['mean_desc_minus_transcript']:+.3f}±{o['mean_desc_minus_transcript_se']:.3f})"
              f" {o['desc_gain']:+8.3f}±{o['desc_gain_se']:.3f} {o['best_desc_score']:9.4f}  ({best})")
# does the ranking of descriptions carry over between framings? (Spearman over the 16 descriptions, fit half)
from scipy.stats import spearmanr
out["rank_corr_vs_transcript_fit"] = {f: float(spearmanr([out[f"transcript|fit"]["per_desc"][d] for d in descs], [out[f"{f}|fit"]["per_desc"][d] for d in descs])[0]) for f in F}
print("\nrank correlation of the 16 descriptions with the transcript framing (fit half):", {k: round(v, 2) for k, v in out["rank_corr_vs_transcript_fit"].items()})
Path("results/phase4/framing/summary.json").write_text(json.dumps(out, indent=1))
