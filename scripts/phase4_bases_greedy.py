"""Arithmetic persona mixtures (Phase 1 method: EM weights on the fit half) on each persona basis, scored on Instruct's
answers, with greedy forward selection of the top personas per basis.

Same answers, split and scoring as Experiment 1 (hold-out half of data/basis_split.json; answers valid under every
context; per-answer log P from the hidden-state pass, which reproduces the Phase 1 scores). Families are used in FULL
(not the screened "_top" subsets of notebook 1.8). Bases with no post-trained text: legacy86 (6 hand-written + 80
elicited), Assistant Axis roles + traits, Big Five (poles, profiles, PERSONA-paper poles, IPIP facets), OLMo spec, values,
base-model descriptions of data, and all of these together. Reference basis built from Instruct's own text (style, self-
and answer-descriptions, few-shot exemplars), shown separately: it is not a base-persona basis.

Greedy: at each step add the context whose inclusion gives the highest FIT-half likelihood after refitting the weights;
the hold-out curve is descriptive (the order is chosen on the fit half only). Errors: question bootstrap (300, seed 1).

    python scripts/phase4_bases_greedy.py
Writes results/phase4/bases/greedy_arithmetic.{json,png}.
"""
import json, os, re, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_selection.basis import split, load_family
from persona_selection.mixture import em_weights, mixture_loglik
from phase1_basis_analysis import boot

RUN = "instruct_unknown_casual_v1"
HID = Path(os.environ.get("SCRATCH", "/pscratch/sd/o/ozamram")) / "persona_phase4" / "hidden" / RUN
NPZ = Path("results/phase1") / RUN / "basis"
OUT = Path("results/phase4/bases")
LABELS = json.load(open("data/prompts/persona_labels.json")) if Path("data/prompts/persona_labels.json").exists() else {}
BASES = [("legacy86 (section 2)", ["legacy86"]), ("Assistant Axis roles + traits", ["axis_roles", "axis_traits"]),
         ("Big Five + IPIP facets", ["bigfive_poles", "bigfive_profiles", "persona_big5", "ipip_facets"]), ("OLMo spec", ["olmo_spec"]),
         ("values + moral foundations", ["values"]), ("base model describes data", ["base_fromdata"])]
ALL = ("all of the above", sorted({f for _, fs in BASES for f in fs}))
PARTIAL = {}
INSTRUCT_REF = ("reference: from Instruct's own text", ["style", "inst_self", "inst_fromdata", "fewshot_generic", "fewshot_hhh"])


def label(n):
    if n in LABELS: return re.sub(r"\s*\(e\d+\)", "", LABELS[n])
    for p, s in [("axt_", " (trait)"), ("ax_", " (role)"), ("b5p_", " (B5 profile)"), ("b5_", " (B5 pole)"), ("pb5_", " (B5)"), ("ipip_", " (IPIP)"), ("sv_", " (value)")]:
        if n.startswith(p): return n[len(p):] + s
    return n


def columns(fams):
    """{name: per-answer log P} for the families: from the hidden-state pass when it has the context, else Phase 1 npz."""
    out = {}
    for f in fams:
        names = [c["name"] for c in load_family(f)]
        if all((HID / f"lp_{n}.npy").exists() for n in names):
            for n in names: out[n] = np.load(HID / f"lp_{n}.npy")
        elif not (NPZ / f"{f}.npz").exists():   # hidden-state pass still running: use the contexts scored so far (reported as partial)
            have = [n for n in names if (HID / f"lp_{n}.npy").exists()]; PARTIAL[f] = (len(have), len(names)); print(f"PARTIAL: {f} {len(have)}/{len(names)} contexts")
            for n in have: out[n] = np.load(HID / f"lp_{n}.npy")
        else:
            z = np.load(NPZ / f"{f}.npz"); assert (z["row_idx"] == np.arange(len(z["row_idx"]))).all() if "row_idx" in z.files else True
            for j, n in enumerate(z["names"]): out[str(n)] = z["L"][:, j]
    return out


def greedy(L, fi, te, nt, names, kmax=10):
    chosen, curve = [], []
    for k in range(min(kmax, L.shape[1])):
        best = None
        for c in range(L.shape[1]):
            if c in chosen: continue
            cols = chosen + [c]; w, _ = em_weights(L[fi][:, cols]); v = mixture_loglik(L[fi][:, cols], w).sum()
            if best is None or v > best[0]: best = (v, c, w)
        chosen.append(best[1]); held = mixture_loglik(L[te][:, chosen], best[2])
        curve.append({"k": k + 1, "added": names[best[1]], "label": label(names[best[1]]), "holdout": float(held.sum() / nt[te].sum()),
                      "weights": {names[c]: round(float(x), 3) for c, x in zip(chosen, best[2])}})
    return curve


def main():
    ix = np.load(HID / "index.npz"); nt = ix["n_tokens"]; q = ix["qidx"]; meta = ix["meta"]
    gen = np.load(HID / "lp_generic.npy"); self_ = np.load(f"results/phase4/self_tokens/{RUN}.npz")
    bases = BASES + [ALL, INSTRUCT_REF]; cols = {b: columns(fs) for b, fs in bases}
    ok = ~meta & np.isfinite(gen) & np.isfinite(self_["total"])
    for c in cols.values():
        for v in c.values(): ok &= np.isfinite(v)
    fit_q, test_q = split(); te = ok & np.isin(q, list(test_q)); fi = ok & np.isin(q, list(fit_q))
    # check: the hidden-state scores reproduce the Phase 1 scores for legacy86
    z = np.load(NPZ / "legacy86.npz"); d = np.abs(z["L"][ok, 0] - cols[BASES[0][0]][str(z["names"][0])][ok]).max(); print(f"legacy86 hidden-state vs Phase 1 score: max |diff| {d:.4f} nats/answer")
    rows = {"generic header": gen[te], "Instruct itself": self_["total"][te]}; nself = np.diff(self_["offsets"])
    res = {"n_fit": int(fi.sum()), "n_holdout": int(te.sum()), "bases": {}}
    for b, fs in bases:
        names = list(cols[b]); L = np.stack([cols[b][n] for n in names], 1)
        w, _ = em_weights(L[fi]); rows[b] = mixture_loglik(L[te], w)
        cur = greedy(L, fi, te, nt, names)
        res["bases"][b] = {"families": fs, "K": len(names), "top_weights": {names[j]: round(float(w[j]), 3) for j in np.argsort(-w)[:8] if w[j] > 0.005}, "greedy": cur}
        print(f"{b:38s} K={len(names):4d}  full mixture {rows[b].sum() / nt[te].sum():.4f}  greedy: " + ", ".join(f"{c['label']} {c['holdout']:.3f}" for c in cur[:5]), flush=True)
    keys = list(rows); nts = [nself[te] if k == "Instruct itself" else nt[te] for k in keys]
    sd, reps = boot([rows[k] for k in keys], nts, q[te]); jl = keys.index(BASES[0][0])
    res["holdout"] = {k: {"logp": float(rows[k].sum() / n.sum()), "se": float(sd[i]), "minus_legacy86": float(rows[k].sum() / n.sum() - rows[BASES[0][0]].sum() / nt[te].sum()),
                          "minus_legacy86_se": float((reps[:, i] - reps[:, jl]).std())} for i, (k, n) in enumerate(zip(keys, nts))}
    res["partial_families"] = PARTIAL; res["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(); res["command"] = " ".join(sys.argv)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "greedy_arithmetic.json").write_text(json.dumps(res, indent=1))
    for k, v in res["holdout"].items(): print(f"  {k:38s} {v['logp']:.4f} ± {v['se']:.4f}   vs legacy86 {v['minus_legacy86']:+.4f} ± {v['minus_legacy86_se']:.4f}")

    # figure: one panel per basis, greedy hold-out curve; the added personas listed in order. The persona panels share a
    # zoomed y axis; the Instruct-text reference panel has its own (it sits far above them)
    g = res["holdout"]["generic header"]["logp"]; P = [b for b, _ in bases]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.8)); axs = axs.ravel()
    ys = [c["holdout"] for b in P[:-1] for c in res["bases"][b]["greedy"]] + [g]; lim = (min(ys) - 0.006, max(ys) + 0.016)
    for n, (ax, b) in enumerate(zip(axs, P)):
        ref = b == INSTRUCT_REF[0]; cur = res["bases"][b]["greedy"]; col = "#7a7a7a" if ref else "#2a78d6"
        ax.plot([c["k"] for c in cur], [c["holdout"] for c in cur], marker="o", ms=5, lw=2, color=col)
        ax.axhline(g, color="#52514e", ls="--", lw=1); ax.axhline(res["holdout"][b]["logp"], color=col, ls=":", lw=1.2)
        if ref: ax.set_ylim(g - 0.01, res["holdout"][b]["logp"] + 0.03)
        else: ax.set_ylim(*lim)
        if n % 4: ax.tick_params(labelleft=ref)
        txt = "\n".join(f"{c['k']}. {c['label'][:30]}" for c in cur[:6])
        ax.text(0.97, 0.05 if not ref else 0.35, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d0d0d0", lw=0.6))
        part = " (partial)" if any(f in PARTIAL for f in dict(bases)[b]) else ""
        ax.set_title(f"{b}, K = {res['bases'][b]['K']}{part}\nwhole basis {res['holdout'][b]['logp']:.3f}", fontsize=9)
        ax.set_xticks(range(1, 11))
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
        if n >= 4: ax.set_xlabel("personas in the mixture (greedy order)", fontsize=8)
        if n % 4 == 0: ax.set_ylabel("hold-out log P / token", fontsize=9)
    axs[0].plot([], [], color="#52514e", ls="--", label=f"base generic header ({g:.3f})"); axs[0].plot([], [], color="#2a78d6", ls=":", label="mixture of the whole basis")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"Arithmetic persona mixtures fitted to Instruct's answers, per basis (Instruct itself: {res['holdout']['Instruct itself']['logp']:.3f}, off the scale)", fontsize=11)
    fig.text(0.01, 0.005, "data: results/phase4/bases/greedy_arithmetic.json (scripts/phase4_bases_greedy.py); hold-out half of data/basis_split.json; EM weights on the fit half", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(OUT / "greedy_arithmetic.png", dpi=150); print("->", OUT / "greedy_arithmetic.png")


if __name__ == "__main__":
    main()
