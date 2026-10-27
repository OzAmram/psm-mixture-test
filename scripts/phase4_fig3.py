"""Figure 3 of results/phase4/SUMMARY.md: hold-out gap to each checkpoint's own score (self - model) across post-training
stages, for the base model + soft prefix (best setting by validation), the arithmetic mixture of the 87 base contexts and,
where it exists, the geometric fit A. Error bars: question bootstrap of the paired difference (results/phase4/exp3/summary.json).

    python scripts/phase4_fig3.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["sft", "dpo", "step100", "step200", "step300", "final"]
LABEL = {"sft": "SFT", "dpo": "DPO", "step100": "RL step 100", "step200": "RL step 200", "step300": "RL step 300", "final": "final Instruct"}


def main():
    S = json.load(open("results/phase4/exp3/summary.json"))["checkpoints"]; ks = [k for k in ORDER if k in S]
    x = np.arange(len(ks)); fig, ax = plt.subplots(figsize=(7.2, 3.6))
    series = [("gap_arithmetic", "gap_arithmetic_se", "arithmetic mixture (87 base contexts)", "#eb6834", "s"),
              ("gap_geometric", "gap_geometric_se", "geometric fit A (87 base contexts)", "#2a78d6", "D"),
              ("gap", "gap_se", "base + soft prefix (best by validation)", "#6250d6", "o")]
    for key, se, lab, col, mk in series:
        xs = [i for i, k in enumerate(ks) if key in S[k]]
        if not xs: continue
        ax.errorbar(xs, [S[ks[i]][key] for i in xs], yerr=[S[ks[i]][se] for i in xs], color=col, marker=mk, ms=7, lw=2, capsize=0, label=lab)
    for i, k in enumerate(ks):
        if "control_pass" in S[k] and not S[k]["control_pass"]: ax.annotate("control fails", (i, S[k].get("gap", 0)), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7, color="#52514e")
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[k] for k in ks]); ax.set_ylabel("self − model log P (nats/token, hold-out)")
    ax.set_title("Distance from each post-training checkpoint to the best base-model description", fontsize=10)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8); ax.set_axisbelow(True); ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, -0.02, "data: results/phase4/exp3/summary.json (scripts/phase4_exp3_analysis.py); bars ±1 question-bootstrap s.e.", fontsize=7, color="#52514e")
    fig.tight_layout(); out = "results/phase4/fig3_stage_gap.png"; fig.savefig(out, dpi=150, bbox_inches="tight"); print("->", out)


if __name__ == "__main__":
    main()
