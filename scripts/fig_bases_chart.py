"""Chart replacing the post's bases table (author's request, 2026-11-24): hold-out log P/token of Olmo 3 Instruct's answers
under the arithmetic and the geometric mixture of each persona basis, as a dot plot (one row per basis, the two fits joined
by a line). Reference: the generic header (dashed); Instruct itself (-0.90) is off the scale and named in the margin.
Data: results/phase4/exp1/instruct_unknown_casual_v1/bases/<basis>/fits_A.json -> "holdout": "arithmetic mixture" and
"A_free_lambda0" (the geometric values in the post's table); generic header from the same files. Instruct's own score
(-0.898, exact self-score of the hold-out answers) is from the Phase 1 table. Standard errors ~0.012 per number; differences
between fits on the same answers are much smaller (paired), so no error bars are drawn.
    python scripts/fig_bases_chart.py   -> results/phase4/fig_bases.png
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path("results/phase4/exp1/instruct_unknown_casual_v1/bases")
ROWS = [("legacy87", "86 base personas (+ generic)"), ("olmospec", "Olmo spec (8)"), ("bigfive", "Big Five + IPIP facets (113)"),
        ("axis", "Assistant Axis roles + traits (516)"), ("three", "Axis + Big Five + spec (635)"), ("all", "all of the above (721)"),
        ("sd", "Instruct's own self-descriptions (261)")]
INSTRUCT_SELF = -0.898
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
ARITH, GEOM = "#eb6834", "#2a78d6"   # same colours as the stage figure (arithmetic orange, geometric blue)

H = {k: json.loads((B / k / "fits_A.json").read_text())["holdout"] for k, _ in ROWS}
generic = H["legacy87"]["generic header"]["logp"]
fig, ax = plt.subplots(figsize=(7.4, 3.9), facecolor=SURF); ax.set_facecolor(SURF)
n = len(ROWS)
for i, (k, lab) in enumerate(ROWS):
    y = n - 1 - i + (0 if k != "sd" else -0.35)   # the circular (Instruct-derived) basis sits a little apart
    a, g = H[k]["arithmetic mixture"]["logp"], H[k]["A_free_lambda0"]["logp"]
    ax.plot([a, g], [y, y], color="#c3c2b7", lw=2, zorder=1, solid_capstyle="round")
    ax.scatter([a], [y], s=60, color=ARITH, edgecolors=SURF, linewidths=2, zorder=3, label="arithmetic mixture" if i == 0 else None)
    ax.scatter([g], [y], s=60, color=GEOM, marker="D", edgecolors=SURF, linewidths=2, zorder=3, label="geometric mixture" if i == 0 else None)
    ax.annotate(f"{g:.3f}", (g, y), xytext=(9, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color=INK2)
    ax.text(-1.79 - 0.005, y, lab, ha="right", va="center", fontsize=8.5, color=INK if k != "sd" else INK2, style="normal" if k != "sd" else "italic")
ax.axvline(generic, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
ax.text(generic + 0.003, -0.85, f"generic header {generic:.3f}", fontsize=7.5, color=INK2, va="bottom")
ax.annotate(f"Instruct itself: {INSTRUCT_SELF:.2f}  →", xy=(1.0, 1.02), xycoords="axes fraction", ha="right", va="bottom", fontsize=7.5, color=INK2)
ax.set_xlim(-1.79, -1.475); ax.set_ylim(-0.9, n - 0.3); ax.set_yticks([])
ax.set_xlabel("log P / token of Instruct's answers (nats; higher = closer to Instruct)", fontsize=8.5, color=INK)
ax.tick_params(axis="x", colors=MUTED, labelsize=8); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8, frameon=False, labelcolor=INK)
fig.tight_layout(); out = "results/phase4/fig_bases.png"; fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF); print("->", out)
for k, lab in ROWS: print(f"  {lab:40s} arithmetic {H[k]['arithmetic mixture']['logp']:.3f}  geometric {H[k]['A_free_lambda0']['logp']:.3f}")
