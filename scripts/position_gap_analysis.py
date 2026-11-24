"""Gap by token position (2026-11-24): log P_Instruct - log P_proxy per answer token, averaged by position, on the 1,200
hold-out Instruct answers of the 86-persona fit. Averaged over Instruct's own answers this estimates the per-position KL
divergence KL(Instruct || proxy) given Instruct's prefix. Proxies: generic header, HHH header, arithmetic mixture (exact
sequential decomposition: at token t each persona is weighted by w_s P_s(answer so far); summed over tokens this is
log sum_s w_s P_s(answer)), geometric mixture (holdout_nll_A_free_lambda0.npy from the fit), 128-token soft context.
Answers whose Instruct self-tokenisation has a different token count (a leading-space token at the start; ~4%) are left out.
Intervals: bootstrap over answers (1000, seed 0).

    python scripts/position_gap_analysis.py   (after scripts/position_gap_score.py)
      -> results/phase4/position_gap/summary.json, fig_position_gap.png
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FIT = Path("results/phase4/exp1/instruct_unknown_casual_v1/bases/legacy87"); P = Path("results/phase4/position_gap")
BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 60)]
H = np.load(FIT / "holdout_per_answer_A.npz"); rows = H["rows"]; nt = H["n_tokens"]; off = np.concatenate([[0], np.cumsum(nt)])
assert (np.load(P / "rows.npy") == rows).all() and len(np.load(P / "lp_generic.npy")) == off[-1], "position_gap_score.py output is not the full hold-out set"
S = np.load("results/phase4/self_tokens/instruct_unknown_casual_v1.npz"); so = S["offsets"]
keep = np.array([so[r + 1] - so[r] == n for r, n in zip(rows, nt)]); print(f"answers kept {keep.sum()} of {len(rows)}")
self_lp = np.concatenate([S["lp"][so[r]:so[r + 1]] if k else np.full(n, np.nan) for r, n, k in zip(rows, nt, keep)])
w = json.load(open(FIT / "fits_A.json"))["arithmetic"]["weights"]; names = list(w); wv = np.array([w[k] for k in names]); wv /= wv.sum()
L = {k: np.load(P / f"lp_{k}.npy").astype(np.float64) for k in ["generic"] + names + ["soft"]}
arith = np.empty(off[-1])
for i in range(len(nt)):
    a, b = off[i], off[i + 1]; X = np.stack([L[k][a:b] for k in names])                     # (S, T)
    prior = np.log(wv)[:, None] + np.concatenate([np.zeros((len(names), 1)), np.cumsum(X, 1)[:, :-1]], 1)
    arith[a:b] = logsumexp(prior + X, 0) - logsumexp(prior, 0)
chk = np.array([arith[off[i]:off[i + 1]].sum() for i in range(len(nt))]); print("arithmetic vs fit file: max |diff| per answer", float(np.abs(chk - H["arithmetic_mixture"]).max()))
proxies = {"generic header": L["generic"], "HHH header": L["hhh"], "arithmetic mixture": arith,
           "geometric mixture": -np.load(FIT / "holdout_nll_A_free_lambda0.npy").astype(np.float64), "soft context (128)": L["soft"]}
pos = np.concatenate([np.arange(1, n + 1) for n in nt]); ans = np.repeat(np.arange(len(nt)), nt); ok = np.repeat(keep, nt) & np.isfinite(L["soft"])
rng = np.random.default_rng(0); A = np.where(keep)[0]; boots = [rng.choice(A, len(A)) for _ in range(1000)]
res = {"n_answers": int(keep.sum()), "bins": [f"{a}-{b}" if a != b else str(a) for a, b in BINS], "proxies": {}}
for name, lp in proxies.items():
    g = self_lp - lp; d = {"overall": float(g[ok].mean()), "per_bin": [], "share": []}
    for a, b in BINS + [(61, 999)]:
        m = ok & (pos >= a) & (pos <= b); gs = np.bincount(ans[m], weights=g[m], minlength=len(nt)); ns = np.bincount(ans[m], minlength=len(nt))
        bs = [gs[s].sum() / ns[s].sum() for s in boots]
        if b != 999: d["per_bin"].append({"gap": float(g[m].mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(m.sum())})
        d["share"].append(float(g[m].sum() / g[ok].sum()))
    res["proxies"][name] = d
(P / "summary.json").write_text(json.dumps(res, indent=1))
print(f"\n{'':20s} overall  " + "  ".join(f"{x:>6s}" for x in res["bins"]) + "   share tokens 1-5 / 6+")
for name, d in res["proxies"].items():
    s15 = sum(d["share"][:4]); print(f"{name:20s} {d['overall']:.3f}   " + "  ".join(f"{x['gap']:6.3f}" for x in d["per_bin"]) + f"   {s15:.0%} / {1 - s15:.0%}")

COL = {"generic header": "#898781", "HHH header": "#1baf7a", "arithmetic mixture": "#eb6834", "geometric mixture": "#2a78d6", "soft context (128)": "#6250d6"}
fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb"); x = np.arange(len(BINS))
for name, d in res["proxies"].items():
    y = [b["gap"] for b in d["per_bin"]]; lo = [b["gap"] - b["ci"][0] for b in d["per_bin"]]; hi = [b["ci"][1] - b["gap"] for b in d["per_bin"]]
    ax.errorbar(x, y, yerr=[lo, hi], color=COL[name], marker="o", ms=5, lw=2, elinewidth=1, capsize=0, label=name, ls="--" if name == "generic header" else "-")
ax.set_xticks(x); ax.set_xticklabels(res["bins"]); ax.set_xlabel("token position in Instruct's answer", fontsize=9)
ax.set_ylabel("log P_Instruct − log P_proxy (nats/token)", fontsize=9); ax.set_ylim(0, None)
ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ["top", "right"]]
ax.legend(fontsize=8, frameon=False, ncol=2); ax.set_title("Gap to Instruct by token position (Instruct's hold-out answers)", fontsize=10)
fig.tight_layout(); fig.savefig(P / "fig_position_gap.png", dpi=160, facecolor="#fcfcfb"); print("->", P / "fig_position_gap.png")
