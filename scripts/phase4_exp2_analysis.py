"""Experiment 2 analysis (notebook 4.2): principal angles between the answer-side subspaces of the score matrices, the
singular-value power-law exponent, and the span regression of chat-template targets onto the base rows.

Inputs: results/phase4/exp2/X_<tag>.npz (phase4_score_matrix.py), columns.jsonl, results/phase4/self_tokens/*.npz.
Output: results/phase4/exp2/analysis.json (+ config).

  principal angles  top-r right singular vectors (answer side) of X_M, r in {5,10,20,50,100}, raw and double-centered.
                    Each comparison is made on random half-splits of the rows (10 splits, seeds 0..9): the noise ceiling
                    compares base rows A with base rows B; base vs model M compares base rows A with M's rows B; so every
                    curve has the same number of rows and the same estimation noise. Also on all rows (no split).
                    Summary: mean squared cosine = ||V_a^T V_b||_F^2 / r. Random reference: r / n_columns (expected
                    value for two independent uniformly random r-dim subspaces), plus 10 draws.
  alpha             weighted least squares of log sigma_i on [1, log(i/n), log((n-i)/n)], weights 1/i, largest singular
                    value excluded (Golowich, Liu & Shetty, arXiv:2510.24966, Appendix A.1.1); n = number of rows.
  span regression   y = log P_M(answer | question) in the chat template (M = instruct, sft, dpo), features = all base rows
                    (standardized), ridge + intercept. Columns split by question into halves (seed 0); penalty chosen by
                    5-fold cross-validation grouped by question inside the fitting half. Held-out R^2 and residual RMS
                    (nats per answer), question bootstrap (1,000). Ceiling control: the generic base row predicted from
                    the other base rows. Reference: y predicted from the answer's token count alone.

    python scripts/phase4_exp2_analysis.py [--models base,sft,dpo,instruct,qwen]
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

E = Path("results/phase4/exp2"); RS = [5, 10, 20, 50, 100]
SELF = {"instruct": {"base_unknown_casual_short_v1": "base_run__instruct", "instruct_unknown_casual_v1": "instruct_unknown_casual_v1"},
        "sft": {"base_unknown_casual_short_v1": "base_run__sft", "instruct_unknown_casual_v1": "instruct_run__sft"},
        "dpo": {"base_unknown_casual_short_v1": "base_run__dpo", "instruct_unknown_casual_v1": "instruct_run__dpo"}}


def dcenter(X): return X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()


def right_sv(X, r):
    _, s, vt = np.linalg.svd(X, full_matrices=False); return vt[:r].T, s


def cosines(Va, Vb): return np.linalg.svd(Va.T @ Vb, compute_uv=False)


def alpha(s):
    n = len(s); i = np.arange(2, n)          # drop the largest (i = 1) and i = n (log 0)
    A = np.stack([np.ones(len(i)), np.log(i / n), np.log((n - i) / n)], 1); w = np.sqrt(1 / i)
    coef = np.linalg.lstsq(A * w[:, None], np.log(s[i - 1]) * w, rcond=None)[0]
    return float(-coef[1])


def ridge_cv(F, y, g, alphas=np.logspace(-2, 5, 15)):
    best = None
    for a in alphas:
        err = []
        for tr, va in GroupKFold(5).split(F, y, g):
            m = Ridge(alpha=a).fit(F[tr], y[tr]); err.append(((m.predict(F[va]) - y[va]) ** 2).mean())
        if best is None or np.mean(err) < best[0]: best = (np.mean(err), a)
    return Ridge(alpha=best[1]).fit(F, y), best[1]


def held_out(pred, y, q, rng, B=1000):
    uq = np.unique(q); idx = {u: np.where(q == u)[0] for u in uq}
    def stats(ix): r = y[ix] - pred[ix]; return 1 - (r ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum(), np.sqrt((r ** 2).mean())
    r2, rms = stats(np.arange(len(y))); bs = np.array([stats(np.concatenate([idx[u] for u in rng.choice(uq, len(uq))])) for _ in range(B)])
    return {"r2": float(r2), "r2_sd": float(bs[:, 0].std()), "rms": float(rms), "rms_sd": float(bs[:, 1].std())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="base,sft,dpo,instruct,qwen"); ap.add_argument("--splits", type=int, default=10)
    a = ap.parse_args(); models = [m for m in a.models.split(",") if (E / f"X_{m}.npz").exists()]
    Z = {m: np.load(E / f"X_{m}.npz") for m in models}; names = [str(n) for n in Z["base"]["names"]]
    for m in models: assert [str(n) for n in Z[m]["names"]] == names, m
    X = {m: Z[m]["X"] for m in models}
    okc = np.all([np.isfinite(x).all(0) for x in X.values()], 0); okr = np.all([np.isfinite(x[:, okc]).all(1) for x in X.values()], 0)
    X = {m: x[okr][:, okc] for m, x in X.items()}; S, C = X["base"].shape
    cols = [json.loads(l) for l in open(E / "columns.jsonl")]; cols = [c for c, k in zip(cols, okc) if k]
    q = np.array([c["qidx"] for c in cols]); nt = Z["base"]["n_tokens"][okc]
    print(f"models {models}; rows {S} (dropped {int((~okr).sum())}), columns {C} (dropped {int((~okc).sum())})")
    out = {"models": models, "n_rows": S, "n_cols": C, "rows_dropped": int((~okr).sum()), "cols_dropped": int((~okc).sum()), "r": RS}

    # principal angles and alpha
    rng = np.random.default_rng(0); rmax = max(RS)
    for mode, f in [("raw", lambda x: x), ("dcenter", dcenter)]:
        res = {"alpha": {}, "full": {}, "split": {}, "random_expected": {r: r / C for r in RS}}
        V = {}
        for m in models:
            V[m], s = right_sv(f(X[m]), rmax); res["alpha"][m] = alpha(s); res.setdefault("sv", {})[m] = (s / np.sqrt(X[m].size)).tolist()
        for m in models:
            if m != "base": res["full"][m] = {r: cosines(V["base"][:, :r], V[m][:, :r]).tolist() for r in RS}
        sp = {k: {r: [] for r in RS} for k in ["ceiling"] + [m for m in models if m != "base"]}
        for k in range(a.splits):
            perm = np.random.default_rng(k).permutation(S); A_, B_ = perm[: S // 2], perm[S // 2:]
            Va, _ = right_sv(f(X["base"][A_]), rmax)
            for key in sp:
                Vb, _ = right_sv(f((X["base"] if key == "ceiling" else X[key])[B_]), rmax)
                for r in RS: sp[key][r].append(cosines(Va[:, :r], Vb[:, :r]))
        res["split"] = {key: {r: {"cos_mean": np.mean(v, 0).tolist(), "cos_sd": np.std(v, 0).tolist(),
                                  "msq": float(np.mean([(c ** 2).mean() for c in v])), "msq_sd": float(np.std([(c ** 2).mean() for c in v]))}
                              for r, v in d.items()} for key, d in sp.items()}
        rc = {r: [cosines(np.linalg.qr(rng.standard_normal((C, r)))[0], np.linalg.qr(rng.standard_normal((C, r)))[0]) for _ in range(10)] for r in RS}
        res["random_draws"] = {r: {"msq": float(np.mean([(c ** 2).mean() for c in v])), "cos_mean": np.mean(v, 0).tolist()} for r, v in rc.items()}
        out[mode] = res
        print(f"[{mode}] alpha", {m: round(v, 3) for m, v in res["alpha"].items()})
        for r in RS: print(f"  r={r:3d} mean sq cos: " + ", ".join(f"{k} {v[r]['msq']:.3f}" for k, v in res["split"].items()) + f", random {r/C:.3f}")

    # span regression
    fit_q = np.unique(q); np.random.default_rng(0).shuffle(fit_q); qa = np.isin(q, fit_q[: len(fit_q) // 2]); qb = ~qa
    Fb = X["base"]
    span = {}
    def run(F, y, label):
        Fs = (F - F[:, qa].mean(1, keepdims=True)) / (F[:, qa].std(1, keepdims=True) + 1e-9)
        m, lam = ridge_cv(Fs[:, qa].T, y[qa], q[qa]); pred = m.predict(Fs[:, qb].T)
        r = held_out(pred, y[qb], q[qb], np.random.default_rng(1)); r["ridge_alpha"] = float(lam); span[label] = r
        print(f"  span {label}: held-out R^2 {r['r2']:.4f} ± {r['r2_sd']:.4f}, RMS {r['rms']:.2f} ± {r['rms_sd']:.2f} nats/answer (penalty {lam:.3g})")
    gi = [n for n, k in zip(names, okr) if k].index("generic")
    run(np.delete(Fb, gi, 0), Fb[gi], "ceiling_generic_row")
    for t in ["instruct", "sft", "dpo"]:
        ys = {}
        for run_name, tag in SELF[t].items():
            p = Path("results/phase4/self_tokens") / f"{tag}.npz"
            if p.exists(): ys[run_name] = np.load(p)["total"]
        if len(ys) < 2: print(f"  span {t}: targets missing, skipped"); continue
        y = np.array([ys[c["run"]][c["row"]] for c in cols]); run(Fb, y, t)
        run(nt[None, :].astype(float), y, f"{t}_length_only")
    out["span"] = span
    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(); out["command"] = " ".join(sys.argv)
    json.dump(out, open(E / "analysis.json", "w"), indent=1); print("->", E / "analysis.json")


if __name__ == "__main__":
    main()
