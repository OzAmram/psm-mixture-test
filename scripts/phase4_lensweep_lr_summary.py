"""Summary of the learning-rate scan for short soft contexts (scripts/phase4_lensweep_lr.sh, 2026-11-19/20) together with
the original length sweep (scripts/phase4_lensweep.sh). For each L <= 128: the lr chosen on validation (choice_L<L>.json),
hold-out log P/token of the 3 seeds at that setting (epoch chosen on validation), and the seed-0 hold-out of every lr tried
(for reference only; the choice used validation). For L >= 256 the original sweep (lr 5e-3, 6 epochs, 3 seeds; lr not
scanned there). Straight-line fit of the tuned curve in log2 L, and local slopes per doubling.
Writes results/phase4/lensweep_lr/summary.json and results/phase4/lensweep_lr/fig_lensweep_tuned.png.

    python scripts/phase4_lensweep_lr_summary.py
"""
import json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

O, S = Path("results/phase4/lensweep_lr"), Path("results/phase4/lensweep")
best = lambda f: max(json.loads(f.read_text())["log"], key=lambda e: e["val_logp"])
tuned, grid = {}, {}
for c in sorted(O.glob("choice_L*.json"), key=lambda p: int(re.search(r"L(\d+)", p.name).group(1))):
    ch = json.loads(c.read_text()); L = ch["L"]
    runs = [best(O / f"L{L}_lr{ch['lr']}_s{s}_e{ch['epochs']}" / "result.json") for s in (0, 1, 2)]
    h = [r["test_logp"] for r in runs]
    tuned[L] = {"lr": ch["lr"], "epochs": ch["epochs"], "holdout": h, "mean": float(np.mean(h)), "sd": float(np.std(h, ddof=1)), "best_epochs": [r["epoch"] for r in runs]}
    grid[L] = {lr: {"val": (b := best(f)).__getitem__("val_logp"), "holdout": b["test_logp"], "best_epoch": b["epoch"]}
               for f in O.glob(f"L{L}_lr*_s0_e12/result.json") for lr in [re.search(r"_lr([^_]+)_s0", str(f)).group(1)]}
old = {}
for L in (256, 512, 1024, 2048):
    h = [best(S / f"L{L}_lr5e-3_s{s}" / "result.json")["test_logp"] for s in (0, 1, 2)]
    old[L] = {"lr": "5e-3", "epochs": 6, "holdout": h, "mean": float(np.mean(h)), "sd": float(np.std(h, ddof=1))}
base_runs = {L: [best(S / f"L{L}_lr5e-3_s{s}" / "result.json")["test_logp"] for s in (0, 1, 2)] for L in tuned}
base = {L: float(np.mean(v)) for L, v in base_runs.items()}

print("L     chosen lr  epochs  best epochs   hold-out mean ± sd (3 seeds)   untuned sweep (lr 5e-3, 6 ep)   gain from tuning")
for L, t in tuned.items():
    print(f"{L:5d}  {t['lr']:>6s}    {t['epochs']:3d}    {str(t['best_epochs']):12s}  {t['mean']:.4f} ± {t['sd']:.4f}              {base[L]:.4f}                       {t['mean'] - base[L]:+.4f}")
for L, t in old.items(): print(f"{L:5d}  {t['lr']:>6s}      6    (not tuned)   {t['mean']:.4f} ± {t['sd']:.4f}")
Ls = sorted(tuned); x = np.log2(Ls); y = np.array([tuned[L]["mean"] for L in Ls]); k, c0 = np.polyfit(x, y, 1)
slopes = [{"from": a, "to": b, "per_doubling": (tuned[b]["mean"] - tuned[a]["mean"]) / np.log2(b / a),
           "se": float(np.sqrt(tuned[a]["sd"] ** 2 / 3 + tuned[b]["sd"] ** 2 / 3) / np.log2(b / a))} for a, b in zip(Ls[:-1], Ls[1:])]
print(f"\ntuned curve, straight line in log2 L over {Ls}: {k:+.4f} nats/token per doubling; residuals " + ", ".join(f"{L}:{r:+.4f}" for L, r in zip(Ls, y - (k * x + c0))))
for s in slopes: print(f"  {s['from']:4d} -> {s['to']:4d}: {s['per_doubling']:+.4f} ± {s['se']:.4f} per doubling")
print("\nseed-0 hold-out for every lr tried (choice made on validation):")
for L, g in grid.items(): print(f"  L{L}: " + "  ".join(f"{lr} {v['holdout']:.4f}(ep{v['best_epoch']})" for lr, v in sorted(g.items(), key=lambda kv: float(kv[0]))))
O.joinpath("summary.json").write_text(json.dumps({"tuned": tuned, "untuned_long": old, "untuned_short_lr5e-3": base, "seed0_grid": grid,
                                                   "loglinear_fit_tuned": {"slope_per_doubling": float(k), "intercept": float(c0)}, "local_slopes": slopes}, indent=1))

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
for L, t in tuned.items(): ax.plot([L] * 3, t["holdout"], "o", ms=4, color=BLUE, alpha=0.35, mec="none")
Lo = sorted(old); allL = Ls + Lo
un_m = [base[L] for L in Ls] + [old[L]["mean"] for L in Lo]; un_sd = [float(np.std(base_runs[L], ddof=1)) for L in Ls] + [old[L]["sd"] for L in Lo]
ax.errorbar(allL, un_m, yerr=un_sd, fmt="--s", color=ORANGE, lw=1.5, ms=6, capsize=3, alpha=0.9, label="LR untuned")
ax.errorbar(Ls, y, yerr=[tuned[L]["sd"] for L in Ls], fmt="-o", color=BLUE, lw=2, ms=8, capsize=3, label="LR tuned")
ax.set_xscale("log", base=2); ax.set_xticks(allL); ax.set_xticklabels([str(L) for L in allL])
ax.set_xlabel("soft-context length (virtual tokens)", color=INK); ax.set_ylabel("hold-out log P per token (nats)", color=INK)
ax.set_title("Olmo 3 base + soft context, scored on Olmo 3 Instruct's answers", fontsize=10, color=INK)
ax.grid(alpha=0.25, lw=0.5); ax.tick_params(colors=MUTED); [s.set_color("#d0cfca") for s in ax.spines.values()]; h, lb = ax.get_legend_handles_labels(); ax.legend(h[::-1], lb[::-1], fontsize=9, frameon=False, loc="lower right")
r = json.loads(next(iter(S.glob("L1_lr5e-3_s0/result.json"))).read_text())["reference"]
fig.text(0.01, 0.01, "Mean ± sd over 3 seeds; 1,200 hold-out answers. LR tuned: learning rate chosen on validation per length, 12 epochs (lengths 1-128).", fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(O / "fig_lensweep_tuned.png", dpi=150); print(f"-> {O}/summary.json, {O}/fig_lensweep_tuned.png")
