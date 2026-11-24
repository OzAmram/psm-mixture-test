"""Scatter plots for the MoralChoice behaviour test (2026-11-21, author's request): per scenario, Instruct's preference for
action 1 (P(action1), averaged over the two option orders) against each proxy's, for the high-ambiguity scenarios.
Panels: generic header, HHH header, arithmetic mixture, geometric mixture, soft context, and a reference panel of
Instruct on the original wording vs on a faithful paraphrase of the same scenario (paraphrase_high.json from
scripts/moralchoice_paraphrase.py, the author's ceiling; order averaging already controls for option order, so the option-
order reference used before, 2026-11-21, was dropped). The paraphrase panel is drawn only when that file exists. Each panel: least-squares line, the diagonal,
R^2 of the fit (mean |difference| is only written to the json).

    python scripts/moralchoice_scatter.py [--in results/phase4/moralchoice/full.json] [--ambiguity high]
Writes results/phase4/moralchoice/scatter_<ambiguity>.png and scatter_<ambiguity>.json.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="results/phase4/moralchoice/full.json"); ap.add_argument("--ambiguity", default="high")
ap.add_argument("--para", default=None, help="paraphrase scores (default: paraphrase_<ambiguity>.json next to --in)"); ap.add_argument("--tag", default="", help="output name prefix, e.g. qwen_")
ap.add_argument("--model-name", default="Olmo 3 Instruct")
a = ap.parse_args(); d = json.loads(Path(a.inp).read_text())
rows = [r for r in d["per_scenario"] if r["ambiguity"] == a.ambiguity]
by = {}
for r in rows: by.setdefault(r["model"], {})[r["id"]] = r
ids = sorted(by["instruct"]); y = np.array([by["instruct"][i]["p1"] for i in ids])
panels = [("generic header", "generic"), ("HHH header", "hhh"), ("arithmetic mixture", "arithmetic"), ("geometric mixture", "geometric"), ("soft context", "soft")]


def stats(x, y):
    k, c = np.polyfit(x, y, 1); r2 = 1 - ((y - (k * x + c)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"r2": float(r2), "slope": float(k), "intercept": float(c), "mean_abs_diff": float(np.abs(x - y).mean()), "n": int(len(x))}


INK, MUTED, BLUE, GREY = "#0b0b0b", "#52514e", "#2a78d6", "#8a8984"
fig, axs = plt.subplots(2, 3, figsize=(11, 7.4), facecolor="#fcfcfb", sharex=True, sharey=True); out = {}
PQ = Path(a.para) if a.para else Path(a.inp).parent / f"paraphrase_{a.ambiguity}.json"; para = None
if PQ.exists():
    para = {r["id"]: r["p1"] for r in json.loads(PQ.read_text())["per_scenario"] if r["model"] == "instruct" and r["ambiguity"] == a.ambiguity}
for ax, (title, m) in zip(axs.flat, panels + [("Instruct on paraphrased scenarios", None)]):
    if m is None:
        if para is None: ax.set_visible(False); continue
        pid = [i for i in ids if i in para]; x = np.array([para[i] for i in pid]); yy = np.array([by["instruct"][i]["p1"] for i in pid]); col = GREY   # proxy = Instruct on the paraphrase
    else: x = np.array([by[m][i]["p1"] for i in ids]); yy = y; col = BLUE
    s = stats(x, yy); out[m or "instruct_paraphrase"] = s
    ax.set_facecolor("#fcfcfb"); ax.scatter(x, yy, s=9, color=col, alpha=0.45, edgecolors="none")
    ax.plot([0, 1], [0, 1], ":", color=MUTED, lw=1); xx = np.array([0, 1]); ax.plot(xx, s["slope"] * xx + s["intercept"], "-", color=INK, lw=1.5)
    if m is not None and para is not None:   # like-for-like with the ceiling: proxy on the original wording vs Instruct on the paraphrase
        pid = [i for i in ids if i in para]; xa = np.array([by[m][i]["p1"] for i in pid]); ya = np.array([para[i] for i in pid])
        s["r2_across_wordings"] = float(np.corrcoef(xa, ya)[0, 1] ** 2)
    sub = (f"R² = {s['r2']:.2f} (n = {s['n']}): the ceiling" if m is None else
           f"R² = {s['r2']:.2f}   (across wordings: {s['r2_across_wordings']:.2f})" if "r2_across_wordings" in s else f"R² = {s['r2']:.2f}")
    ax.set_title(f"{title}\n{sub}", fontsize=9.5, color=INK)
    ax.grid(alpha=0.25, lw=0.5); ax.tick_params(colors=MUTED, labelsize=8); [sp.set_color("#d0cfca") for sp in ax.spines.values()]
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
for ax in axs[1]: ax.set_xlabel("proxy's P(action 1)", color=INK, fontsize=9)
for ax in axs[:, 0]: ax.set_ylabel("Instruct's P(action 1)", color=INK, fontsize=9)
axs[1, 2].set_xlabel("Instruct's P(action 1), paraphrased wording", color=INK, fontsize=9)
fig.suptitle(f"MoralChoice {a.ambiguity}-ambiguity scenarios (n = {len(ids)}): which action does each proxy prefer, vs {a.model_name}?", fontsize=11, color=INK)
fig.text(0.01, 0.005, "One point per scenario; P(action 1) from next-token P(A)/P(B), averaged over both option orders.\nIn brackets: R² of the proxy on the original wording vs Instruct on the paraphrased wording (n = 601), the comparison that matches the ceiling panel. "
         "Dotted: y = x; solid: least-squares line.", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.035, 1, 0.97)); P = Path(a.inp).parent
fig.savefig(P / f"{a.tag}scatter_{a.ambiguity}.png", dpi=150); (P / f"{a.tag}scatter_{a.ambiguity}.json").write_text(json.dumps(out, indent=1))
for k, v in out.items(): print(f"{k:20s} R2 {v['r2']:.3f}  slope {v['slope']:.2f}  intercept {v['intercept']:+.2f}  mean|diff| {v['mean_abs_diff']:.3f}  n {v['n']}")
print("->", P / f"{a.tag}scatter_{a.ambiguity}.png")
