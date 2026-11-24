"""Summary of the soft-context length sweep (scripts/phase4_lensweep.sh, 2026-11-18): for each run the hold-out log P per
token at the epoch chosen on validation; per length the mean and standard deviation over seeds (lr 5e-3); local slopes
between neighbouring lengths (nats/token per doubling of L, from the seed means, with a standard error from the seed
spread); and a straight-line fit of log P against log2 L over all lengths with every seed done, with its residuals.
Writes results/phase4/lensweep/summary.json and fig_lensweep.png (partial results are fine: missing runs are skipped).

    python scripts/phase4_lensweep_summary.py
"""
import json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

O = Path("results/phase4/lensweep"); runs = []
for f in sorted(O.glob("L*_lr*_s*/result.json")):
    L, lr, s = re.match(r"L(\d+)_lr([^_]+)_s(\d+)", f.parent.name).groups(); r = json.loads(f.read_text())
    b = max(r["log"], key=lambda e: e["val_logp"])
    runs.append({"L": int(L), "lr": lr, "seed": int(s), "best_epoch": b["epoch"], "holdout": b["test_logp"], "val": b["val_logp"],
                 "epochs_run": len(r["log"]), "grad_checkpointing": bool(r["args"].get("grad_checkpointing")),
                 "instruct_self": r["reference"]["instruct_self"], "generic": r["reference"]["base_generic_header"]})
main = [x for x in runs if x["lr"] == "5e-3"]; Ls = sorted({x["L"] for x in main})
per = {L: [x["holdout"] for x in main if x["L"] == L] for L in Ls}
tab = {L: {"n_seeds": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
           "best_epochs": sorted(x["best_epoch"] for x in main if x["L"] == L)} for L, v in per.items()}
print("L      seeds  hold-out mean    sd      best epochs")
for L, t in tab.items(): print(f"{L:5d}  {t['n_seeds']:3d}   {t['mean']:.4f}   {t['sd'] if t['sd'] is None else round(t['sd'], 4)!s:>7}   {t['best_epochs']}")
slopes = []
for a, b in zip(Ls[:-1], Ls[1:]):
    d = (tab[b]["mean"] - tab[a]["mean"]) / np.log2(b / a)
    se = None if tab[a]["sd"] is None or tab[b]["sd"] is None else float(np.sqrt(tab[a]["sd"]**2 / tab[a]["n_seeds"] + tab[b]["sd"]**2 / tab[b]["n_seeds"]) / np.log2(b / a))
    slopes.append({"from": a, "to": b, "per_doubling": float(d), "se": se})
print("\nlocal slope, nats/token per doubling of L:")
for s in slopes: print(f"  {s['from']:5d} -> {s['to']:5d}: {s['per_doubling']:+.4f}" + ("" if s["se"] is None else f" ± {s['se']:.4f}"))
full = [L for L in Ls if tab[L]["n_seeds"] == 3]; fit = None
if len(full) >= 3:
    x = np.log2(full); y = np.array([tab[L]["mean"] for L in full]); k, c = np.polyfit(x, y, 1)
    fit = {"lengths": full, "slope_per_doubling": float(k), "intercept": float(c), "residuals": {int(L): float(r) for L, r in zip(full, y - (k * x + c))}}
    print(f"\nstraight line in log2 L over {full}: {k:+.4f} nats/token per doubling; residuals " + ", ".join(f"{L}:{r:+.4f}" for L, r in fit["residuals"].items()))
check = {f"L{x['L']}_s{x['seed']}": x["holdout"] for x in runs if x["lr"] == "2e-2"}
if check: print("\nlr 2e-2 check (seed 0):", {k: round(v, 4) for k, v in check.items()})
O.joinpath("summary.json").write_text(json.dumps({"runs": runs, "per_length": {str(k): v for k, v in tab.items()}, "local_slopes": slopes,
                                                  "loglinear_fit": fit, "lr_2e-2_check": check}, indent=1))

# figure: hold-out log P/token vs prefix length (log2 axis); seed points + seed mean +- sd; lr 2e-2 check
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
for x in main: ax.plot(x["L"], x["holdout"], "o", ms=4, color=BLUE, alpha=0.35, mec="none")
m = np.array([tab[L]["mean"] for L in Ls]); sd = np.array([tab[L]["sd"] or 0 for L in Ls])
ax.errorbar(Ls, m, yerr=sd, fmt="-o", color=BLUE, lw=2, ms=8, capsize=3, label="lr 5e-3: seed mean ± sd (dots: single seeds)")
if check:
    cl = sorted((x["L"], x["holdout"]) for x in runs if x["lr"] == "2e-2")
    ax.plot(*zip(*cl), "^", color=ORANGE, ms=8, ls="none", label="lr 2e-2, seed 0 (under-training check)")
if fit: xx = np.array([min(full), max(full)]); ax.plot(xx, fit["slope_per_doubling"] * np.log2(xx) + fit["intercept"], "--", color=MUTED, lw=1, label="straight line in log L")
ax.set_xscale("log", base=2); ax.set_xticks(Ls); ax.set_xticklabels([str(L) for L in Ls])
ax.set_xlabel("soft-context length L (vectors)", color=INK); ax.set_ylabel("hold-out log P per token (nats)", color=INK)
ax.set_title("Olmo 3 base + soft context, scored on Olmo 3 Instruct's answers", fontsize=10, color=INK)
ax.grid(alpha=0.25, lw=0.5); ax.tick_params(colors=MUTED); [s.set_color("#d0cfca") for s in ax.spines.values()]
ax.legend(fontsize=8, frameon=False, loc="lower right")
ref = runs[0] if runs else None
if ref: fig.text(0.01, 0.01, f"1,200 hold-out answers (Phase 4 split); epoch chosen on validation. For scale: generic header {ref['generic']:.3f}, Instruct itself {ref['instruct_self']:.3f}.\n"
                 "Data: results/phase4/lensweep (scripts/phase4_lensweep.sh)", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(O / "fig_lensweep.png", dpi=150); print(f"\n-> {O}/summary.json, {O}/fig_lensweep.png")
