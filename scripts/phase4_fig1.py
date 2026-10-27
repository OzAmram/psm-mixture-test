"""Figure 1 of results/phase4/SUMMARY.md: hold-out log P/token of every geometric fit (A-E) on the Instruct answers, with
reference lines at the generic header, the arithmetic mixture, the base soft prefix and Instruct's own self-score.
All numbers on the same hold-out answers (results/phase4/exp1/<run>/holdout_per_answer_*.npz rows; the fits_*.json of separate invocations are merged); error bars are the
question-bootstrap standard errors stored by phase4_geom_fit.py. The soft-prefix and self lines use the same rows.

    python scripts/phase4_fig1.py --run instruct_unknown_casual_v1
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_basis_analysis import boot

INK, MUTED, BLUE, ORANGE, AQUA, VIOLET = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="instruct_unknown_casual_v1")
    ap.add_argument("--prefix", default="results/phase4/exp1/prefix_instruct_L128/scores.npz")
    ap.add_argument("--out", default="results/phase4/fig1_geometric_fits.png"); a = ap.parse_args()
    E = Path("results/phase4/exp1") / a.run; F = {"holdout": {}}
    for f in sorted(E.glob("fits_*.json")): F["holdout"].update(json.load(open(f)).get("holdout", {}))
    ha = np.load(sorted(E.glob("holdout_per_answer_*.npz"))[0])
    rows, nt, q = ha["rows"], ha["n_tokens"], ha["qidx"]
    refs = {"generic header": (F["holdout"]["generic header"]["logp"], F["holdout"]["generic header"]["se"], MUTED),
            "arithmetic mixture (87 contexts)": (F["holdout"]["arithmetic mixture"]["logp"], F["holdout"]["arithmetic mixture"]["se"], ORANGE)}
    extra = {}
    st = Path("results/phase4/self_tokens") / f"{a.run}.npz"
    if st.exists():   # self-score per answer has its own token count (chat template, stripped answer)
        s = np.load(st); n_self = np.diff(s["offsets"])[rows]; extra["Instruct self-score"] = (s["total"][rows], n_self, AQUA)
    if Path(a.prefix).exists():
        p = np.load(a.prefix); extra["base + soft prefix"] = (p["lp"][rows], p["n_tokens"][rows], VIOLET)
    for k, (lp, n, col) in extra.items():
        ok = np.isfinite(lp); sd, _ = boot([lp[ok]], [n[ok]], q[ok]); refs[k] = (float(lp[ok].sum() / n[ok].sum()), float(sd[0]), col)
    fits = [(k, v) for k, v in F["holdout"].items() if k not in ("generic header", "arithmetic mixture")]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(fits) + 1.8))
    y = np.arange(len(fits))[::-1]
    ax.errorbar([v["logp"] for _, v in fits], y, xerr=[v["se"] for _, v in fits], fmt="o", ms=7, color=BLUE, ecolor=BLUE, elinewidth=2, capsize=0)
    for k, (x, se, col) in refs.items():
        ax.axvline(x, color=col, lw=2, ls="--", label=f"{k}: {x:.3f}")
        ax.axvspan(x - se, x + se, color=col, alpha=0.12, lw=0)
    ax.set_yticks(y); ax.set_yticklabels([k for k, _ in fits]); ax.set_xlabel("hold-out log P per answer token (nats)")
    ax.set_title(f"Geometric persona fits to {a.run} answers (hold-out half, {len(rows)} answers)", fontsize=10, color=INK)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.01, 0.005, f"data: {E}/fits_*.json; bars and bands = ±1 question-bootstrap s.e.", fontsize=7, color=MUTED)
    fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=150, bbox_inches="tight"); print("->", a.out)


if __name__ == "__main__":
    main()
