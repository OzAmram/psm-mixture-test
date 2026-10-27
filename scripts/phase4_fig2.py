"""Figure 2 of results/phase4/SUMMARY.md: cosines of the principal angles between the answer-side subspaces of the base
score matrix and each other model's, against the angle index, one panel per subspace dimension r. Every curve compares base
rows (random half A) with the other model's rows (half B), averaged over 10 splits (results/phase4/exp2/analysis.json);
"noise ceiling" = base half A vs base half B; "random" = two random r-dimensional subspaces (10 draws).

    python scripts/phase4_fig2.py [--mode dcenter|raw]
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"ceiling": "#0b0b0b", "sft": "#2a78d6", "dpo": "#1baf7a", "instruct": "#eb6834", "qwen": "#6250d6", "random": "#9a9893"}
LAB = {"ceiling": "noise ceiling (base vs base)", "sft": "base vs SFT", "dpo": "base vs DPO", "instruct": "base vs Instruct", "qwen": "base vs Qwen2.5 base", "random": "random subspaces"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="dcenter"); ap.add_argument("--out", default=None); a = ap.parse_args()
    A = json.load(open("results/phase4/exp2/analysis.json")); R = A[a.mode]; rs = [str(r) for r in A["r"]]
    fig, axs = plt.subplots(1, len(rs), figsize=(3.0 * len(rs), 3.2), sharey=True)
    for ax, r in zip(axs, rs):
        for key in ["ceiling", "sft", "dpo", "instruct", "qwen"]:
            if key in R["split"]:
                d = R["split"][key][r]; m, s = np.array(d["cos_mean"]), np.array(d["cos_sd"]); x = np.arange(1, len(m) + 1)
                ax.plot(x, m, color=COL[key], lw=2, label=LAB[key]); ax.fill_between(x, m - s, m + s, color=COL[key], alpha=0.15, lw=0)
        rd = np.array(R["random_draws"][r]["cos_mean"]); ax.plot(np.arange(1, len(rd) + 1), rd, color=COL["random"], lw=1.5, ls="--", label=LAB["random"])
        ax.set_title(f"r = {r}", fontsize=10); ax.set_xlabel("principal angle index"); ax.set_ylim(0, 1.02)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
    axs[0].set_ylabel("cosine of principal angle")
    axs[-1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Answer-side subspace overlap with the base model ({'double-centered' if a.mode == 'dcenter' else 'raw'} score matrices, "
                 f"{A['n_rows']} contexts x {A['n_cols']} answers; bands ±1 s.d. over 10 row splits)", fontsize=9)
    fig.text(0.01, -0.02, "data: results/phase4/exp2/analysis.json (scripts/phase4_exp2_analysis.py)", fontsize=7, color="#52514e")
    out = a.out or f"results/phase4/fig2_principal_angles_{a.mode}.png"; fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); print("->", out)


if __name__ == "__main__":
    main()
