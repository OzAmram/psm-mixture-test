"""Compare persona bases on one run: hold-out log P / token of the run's answers under each basis's fitted mixture.

    python scripts/phase1_basis_analysis.py --run instruct_unknown_casual_v1 \
        --bases "legacy86=legacy86; bigfive=bigfive_poles+bigfive_profiles; all=legacy86+bigfive_poles+bigfive_profiles"

For every basis (a union of family files from results/phase1/<run>/basis/), with the frozen split of data/basis_split.json:
  * mixture: weights fitted by EM on the fit-half questions, log P / token on the hold-out half;
  * best single component, chosen on the fit half;
  * paired question-bootstrap difference of the mixture against the reference basis (default legacy86);
  * the top fitted weights.
Reference rows: the sampling model itself (Instruct exact self log P; tempered self log P for instruct_T* runs; the generic
header for base-model runs) and the base generic header. Answers flagged as meta/placeholder, or with any NaN score, are dropped
(the same answers for every basis). Writes results/phase1/basis_<run>.{json,md} (or --tag).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.mixture import em_weights, mixture_loglik
from persona_selection.basis import split


def load_run(run, fams):
    d = Path("results/phase1") / run; rows = [json.loads(l) for l in open(d / "rows.jsonl")]
    blocks, names, nt, qidx, meta = [], [], None, None, None
    for f in fams:
        z = np.load(d / "basis" / f"{f}.npz"); blocks.append(z["L"]); names += [str(n) for n in z["names"]]
        nt, qidx, meta = z["n_tokens"], z["qidx"], z["meta"]
    L = np.concatenate(blocks, 1)
    if (d / "rows_selfexact.jsonl").exists():
        ex = [json.loads(l) for l in open(d / "rows_selfexact.jsonl")]; s = np.array([e["ll_self_exact"] for e in ex]); ns = np.array([e["n_tokens_exact"] for e in ex]); lab = "Instruct itself (exact)"
    elif "ll_self_T" in rows[0]:
        s = np.array([r["ll_self_T"] for r in rows]); ns = np.array([r["n_tokens_exact"] for r in rows]); src = rows[0]["source"]
        lab = f"tempered Instruct itself (T={src.split('_T')[-1]}, exact)" if src.startswith("instruct_T") else f"sampling model itself ({src}, exact)"
    else:
        s, ns, lab = None, None, None
    return {"L": L, "names": names, "nt": nt, "qidx": qidx, "meta": meta, "self": s, "nt_self": ns, "self_label": lab, "rows": rows}


def per_tok(v, n): return float(v.sum() / n.sum())


def boot(vals_list, nts, q, n=300, seed=1):
    """Question bootstrap of token-weighted means; vals_list: list of per-answer arrays; returns std of each."""
    rng = np.random.default_rng(seed); uq = np.unique(q); idx = {g: np.where(q == g)[0] for g in uq}; reps = []
    for _ in range(n):
        m = np.concatenate([idx[g] for g in rng.choice(uq, len(uq))]); reps.append([v[m].sum() / nn[m].sum() for v, nn in zip(vals_list, nts)])
    reps = np.array(reps); return reps.std(0), reps


def main(argv=None):
    """Command-line entry; notebooks call main([...same arguments...]) and get the result dict back."""
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--bases", required=True)
    ap.add_argument("--ref", default="legacy86"); ap.add_argument("--tag", default=None); ap.add_argument("--greedy", default="")
    a = ap.parse_args(argv)
    specs = {}
    for part in a.bases.split(";"):
        k, v = part.split("="); specs[k.strip()] = [f.strip() for f in v.split("+")]
    fams = sorted({f for v in specs.values() for f in v} | {"generic"})
    D = load_run(a.run, fams); fit_q, test_q = split()
    col = {}; i = 0
    for f in fams:
        n = len(np.load(Path("results/phase1") / a.run / "basis" / f"{f}.npz")["names"]); col[f] = list(range(i, i + n)); i += n
    ok = ~D["meta"] & np.isfinite(D["L"]).all(1)
    if D["self"] is not None: ok &= np.isfinite(D["self"])
    tr = ok & np.isin(D["qidx"], list(fit_q)); te = ok & np.isin(D["qidx"], list(test_q))
    L, nt, q = D["L"], D["nt"], D["qidx"]
    res = {"run": a.run, "n_fit": int(tr.sum()), "n_test": int(te.sum()), "rows": {}}
    ref_rows = {}
    if D["self"] is not None: ref_rows[D["self_label"]] = (D["self"][te], D["nt_self"][te])
    ref_rows["base generic header"] = (L[te][:, col["generic"][0]], nt[te])
    mix = {}
    for name, fl in specs.items():
        cols = [c for f in fl for c in col[f]]; w, _ = em_weights(L[tr][:, cols]); m = mixture_loglik(L[te][:, cols], w)
        b = cols[int(np.argmax(L[tr][:, cols].sum(0)))]; mix[name] = m
        top = np.argsort(-w)[:8]
        res["rows"][name] = {"families": fl, "K": len(cols), "mixture": per_tok(m, nt[te]), "fit_half_mixture": per_tok(mixture_loglik(L[tr][:, cols], w), nt[tr]),
                             "best_single": per_tok(L[te][:, b], nt[te]), "best_single_name": D["names"][b],
                             "top_weights": {D["names"][cols[j]]: round(float(w[j]), 3) for j in top if w[j] > 0.005}}
    # uncertainties: marginal sigma of every row, and paired differences vs the reference basis
    keys = list(ref_rows) + list(mix); vals = [ref_rows[k][0] for k in ref_rows] + [mix[k] for k in mix]; nts = [ref_rows[k][1] for k in ref_rows] + [nt[te]] * len(mix)
    sd, reps = boot(vals, nts, q[te])
    for k, s in zip(keys, sd):
        if k in res["rows"]: res["rows"][k]["se"] = float(s)
    res["refs"] = {k: {"logp": per_tok(*ref_rows[k]), "se": float(sd[i])} for i, k in enumerate(ref_rows)}
    if a.ref in mix:
        j = keys.index(a.ref)
        for k in mix:
            d = reps[:, keys.index(k)] - reps[:, j]; res["rows"][k]["minus_ref"] = per_tok(mix[k], nt[te]) - per_tok(mix[a.ref], nt[te]); res["rows"][k]["minus_ref_se"] = float(d.std())
    # optional greedy curve (selection by fit-half likelihood) over the named basis
    if a.greedy:
        cols_all = [c for f in specs[a.greedy] for c in col[f]]; chosen, curve = [], []
        for _ in range(min(12, len(cols_all))):
            best = None
            for c in cols_all:
                if c in chosen: continue
                w, _ = em_weights(L[tr][:, chosen + [c]]); v = mixture_loglik(L[tr][:, chosen + [c]], w).sum()
                if best is None or v > best[0]: best = (v, c, w)
            chosen.append(best[1]); curve.append({"k": len(chosen), "added": D["names"][best[1]], "test": per_tok(mixture_loglik(L[te][:, chosen], best[2]), nt[te])})
        res["greedy"] = {a.greedy: curve}
    tag = a.tag or a.run; out = Path("results/phase1/basis"); out.mkdir(exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    md = [f"# Basis comparison on {a.run} ({res['n_test']} hold-out answers, {res['n_fit']} fit answers; log P / token, ±1σ question bootstrap)\n",
          "| basis | K | hold-out mixture | Δ vs " + a.ref + " (paired) | best single (fit-selected) | top weights |", "|---|---|---|---|---|---|"]
    for k, v in res["refs"].items(): md.append(f"| *{k}* | | {v['logp']:.3f} ± {v['se']:.3f} | | | |")
    for k, v in res["rows"].items():
        dd = f"{v['minus_ref']:+.3f} ± {v['minus_ref_se']:.3f}" if "minus_ref" in v else ""
        md.append(f"| {k} | {v['K']} | {v['mixture']:.3f} ± {v['se']:.3f} | {dd} | {v['best_single']:.3f} ({v['best_single_name']}) | " + ", ".join(f"{n} {w:.2f}" for n, w in v["top_weights"].items()) + " |")
    if a.greedy:
        md.append(f"\ngreedy over {a.greedy}: " + " → ".join(f"{c['added']} ({c['test']:.3f})" for c in res["greedy"][a.greedy]))
    (out / f"{tag}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    return res


if __name__ == "__main__":
    main()
