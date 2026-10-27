"""Experiment 1: fit geometric (logit-linear) persona combinations to a run's answers, exact full-vocabulary softmax.

    P_c(a_t | q, a_<t) = softmax( sum_s c_s z_s,t )[a_t]          (87 contexts: the 86 legacy personas + the generic header)

Fits (coefficients fitted on the FIT half of the questions, data/basis_split.json; all scores on the hold-out half):
  A  free c, L2 penalty lambda * ||c - c0||^2 with c0 = 0 ("zero") or c0 = one-hot on the generic header ("generic");
     lambda in {0, 1e-4, 1e-3, 1e-2} chosen on a validation split of 20% of the fit-half questions (seed 0), then refit
     on the whole fit half
  B  c >= 0
  C  c >= 0, sum c = 1 (log-linear pool)
  D  a single persona with a free scale (best persona chosen on the fit half)
  E  greedy forward selection (12 steps; at each step the 5 unchosen personas with the largest objective gradient are
     refitted and the best by fit-half likelihood is added; an approximation to trying all ~80, noted in NOTES.md)
Reference rows: the generic header (c = one-hot; must reproduce the Phase 1 score), and the ARITHMETIC mixture of the
same 87 contexts (EM on the fit half, as in Phase 1). Objective = mean NLL per answer token (+ penalty); L-BFGS-B (A, B, D, E),
SLSQP (C), all started from c = one-hot generic. Uncertainty: question bootstrap (300 resamples, seed 1) of hold-out
per-token log P, as phase1_basis_analysis.py.

    python scripts/phase4_geom_fit.py --run instruct_unknown_casual_v1 --fits A,B,C,D,E --devices 0,1,2,3
Writes results/phase4/exp1/<run>/fits_<letters>.json (e.g. fits_A.json for --fits A; rewritten after every fit, so a
run cut off by the time limit keeps what finished), holdout_per_answer_<letters>.npz, holdout_nll_<fit>.npy (per-position
NLL on hold-out positions, for the residual analysis) and config_<letters>.json. Separate invocations share the generic
and arithmetic rows and use the same bootstrap resamples (seed 1, same hold-out rows), so their numbers are comparable.
"""
import argparse, json, os, sys, time, subprocess
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_selection.geometric import load_index, ok_rows, positions, load_hidden, Shards, stream_nll
from persona_selection.basis import split
from persona_selection.mixture import em_weights, mixture_loglik
from phase1_basis_analysis import boot

HID = Path(os.environ.get("SCRATCH", "/pscratch/sd/o/ozamram")) / "persona_phase4" / "hidden"


def unembedding():
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download("allenai/Olmo-3-1025-7B")); idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
    with safe_open(d / idx["lm_head.weight"], "pt") as f: return f.get_tensor("lm_head.weight").to(torch.bfloat16)   # (V, d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--fits", default="A,B,C,D,E")
    ap.add_argument("--lambdas", default="0,1e-4,1e-3,1e-2"); ap.add_argument("--devices", default="0"); ap.add_argument("--greedy-steps", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=300); ap.add_argument("--out", default="results/phase4/exp1"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke test only: keep the first N fit and N hold-out answers")
    a = ap.parse_args(); t00 = time.time()
    hdir = HID / a.run; out = Path(a.out) / a.run; out.mkdir(parents=True, exist_ok=True)
    ix = load_index(hdir); ctxs = [str(c) for c in ix["contexts"]]; K = len(ctxs); gi = ctxs.index("generic")
    ok = ok_rows(hdir, ctxs); fit_q, test_q = split(); q = ix["qidx"]
    fit_rows = np.where(ok & np.isin(q, list(fit_q)))[0]; test_rows = np.where(ok & np.isin(q, list(test_q)))[0]
    if a.max_rows: fit_rows, test_rows = fit_rows[: a.max_rows], test_rows[: a.max_rows]
    rng = np.random.default_rng(a.seed); fq = np.array(sorted(fit_q)); val_q = set(rng.choice(fq, int(round(0.2 * len(fq))), replace=False).tolist())
    devs = [torch.device(f"cuda:{i}") for i in range(len(a.devices.split(",")))]
    W = unembedding(); print(f"[{a.run}] K={K} contexts; fit {len(fit_rows)} answers, hold-out {len(test_rows)}; W_U {tuple(W.shape)}", flush=True)

    # fit-half positions on the GPUs; weights select the fit-train / fit-val subsets for the lambda choice
    pos_f = positions(ix, fit_rows); row_f = ix["row_of"][pos_f]; isval = np.isin(q[row_f], list(val_q))
    S = Shards(load_hidden(hdir, ctxs, pos_f), ix["targets"][pos_f], W, devs); print(f"fit-half positions {S.P} on {len(devs)} GPU(s) ({time.time()-t00:.0f}s)", flush=True)
    w_all = np.ones(S.P); w_tr = (~isval).astype(float); w_val = isval.astype(float)
    e_gen = np.zeros(K); e_gen[gi] = 1.0
    n_eval = [0]

    def objective(w, lam=0.0, c0=None, mask=None):
        nw = w.sum()
        def f(c):
            cc = c if mask is None else np.where(mask, c, 0.0)
            v, g, _ = S.nll(cc, w); n_eval[0] += 1
            v, g = v / nw, g / nw
            if mask is not None: g = np.where(mask, g, 0.0)
            if lam > 0: d = cc - c0; v += lam * float(d @ d); g = g + 2 * lam * d
            return v, g
        return f

    def lbfgs(f, x0, bounds=None):
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": a.maxiter, "ftol": 1e-12, "gtol": 1e-7})
        return r.x, r

    # hold-out positions stay in CPU RAM and are streamed for each evaluation
    pos_t = positions(ix, test_rows); row_t = ix["row_of"][pos_t]; Ht = load_hidden(hdir, ctxs, pos_t); tgt_t = ix["targets"][pos_t]
    nt_t = ix["n_tokens"][test_rows]; q_t = q[test_rows]; rmap = {r: i for i, r in enumerate(test_rows)}; ridx = np.array([rmap[r] for r in row_t])

    def holdout(c, name=None):
        nll = stream_nll(Ht, tgt_t, c, W, devs[0]); tot = -np.bincount(ridx, weights=nll, minlength=len(test_rows))
        if name: np.save(out / f"holdout_nll_{name}.npy", nll.astype(np.float32))
        return tot

    tag = "".join(a.fits.split(","))
    res = {"run": a.run, "contexts": ctxs, "n_fit": int(len(fit_rows)), "n_test": int(len(test_rows)), "n_fit_positions": int(S.P),
           "n_test_positions": int(len(pos_t)), "val_questions": sorted(int(x) for x in val_q), "fits": {}}
    per_answer = {}

    # sanity: the generic header alone (c = one-hot) must reproduce the stored Phase 1 score of the same answers
    lp_gen = np.load(hdir / "lp_generic.npy")[test_rows]; tot = holdout(e_gen, "generic")
    res["check_generic"] = {"holdout_from_hidden": float(tot.sum() / nt_t.sum()), "holdout_stored": float(lp_gen.sum() / nt_t.sum()),
                            "max_abs_diff_per_answer": float(np.abs(tot - lp_gen).max())}
    print("check generic one-hot:", res["check_generic"], flush=True); per_answer["generic header"] = tot

    # arithmetic mixture of the same 87 contexts (EM), as in Phase 1
    Lf = np.stack([np.load(hdir / f"lp_{c}.npy") for c in ctxs], 1)
    wa, _ = em_weights(Lf[fit_rows]); per_answer["arithmetic mixture"] = mixture_loglik(Lf[test_rows], wa)
    res["arithmetic"] = {"weights": {ctxs[j]: round(float(wa[j]), 4) for j in np.argsort(-wa)[:10] if wa[j] > 1e-3}}

    def record(name, c, extra=None):
        tot = holdout(c, name); per_answer[name] = tot; v, _, _ = S.nll(c, w_all)
        top = np.argsort(-np.abs(c))[:10]
        res["fits"][name] = {"c": c.tolist(), "sum_c": float(c.sum()), "fit_half_logp": -v / w_all.sum(), "holdout_logp": float(tot.sum() / nt_t.sum()),
                             "top_abs_c": [(ctxs[j], round(float(c[j]), 4)) for j in top], **(extra or {})}
        print(f"  [{name}] hold-out {res['fits'][name]['holdout_logp']:.4f}  fit-half {res['fits'][name]['fit_half_logp']:.4f}  sum c {c.sum():.3f}  evals {n_eval[0]}  ({time.time()-t00:.0f}s)", flush=True)
        print("     top |c|:", res["fits"][name]["top_abs_c"], flush=True)
        (out / f"fits_{tag}.json").write_text(json.dumps(res, indent=1))

    fits = a.fits.split(",")
    if "A" in fits:
        lams = [float(x) for x in a.lambdas.split(",")]; sel = {}
        for tgt_name, c0 in [("zero", np.zeros(K)), ("generic", e_gen)]:
            vals = []
            for lam in lams:
                if lam == 0 and tgt_name == "generic" and "zero" in sel: vals.append(sel["zero"]["grid"][0]); continue
                c, r = lbfgs(objective(w_tr, lam, c0), e_gen.copy()); v, _, _ = S.nll(c, w_val)
                vals.append({"lambda": lam, "val_logp": -v / w_val.sum(), "iters": int(r.nit), "converged": bool(r.success)})
                print(f"  A/{tgt_name} lambda {lam:g}: validation log P/token {vals[-1]['val_logp']:.4f} ({r.nit} iters, {r.message})", flush=True)
            best = max(vals, key=lambda d: d["val_logp"]); sel[tgt_name] = {"grid": vals, "chosen_lambda": best["lambda"]}
        res["A_lambda_selection"] = sel
        c, r = lbfgs(objective(w_all), e_gen.copy()); record("A_free_lambda0", c, {"iters": int(r.nit), "message": str(r.message)})
        for tgt_name, c0 in [("zero", np.zeros(K)), ("generic", e_gen)]:
            lam = sel[tgt_name]["chosen_lambda"]
            if lam == 0: continue
            c, r = lbfgs(objective(w_all, lam, c0), e_gen.copy()); record(f"A_free_l2{tgt_name}_{lam:g}", c, {"iters": int(r.nit), "lambda": lam, "target": tgt_name})
    if "B" in fits:
        c, r = lbfgs(objective(w_all), e_gen.copy(), bounds=[(0, None)] * K); record("B_nonneg", c, {"iters": int(r.nit), "n_nonzero": int((c > 1e-6).sum())})
    if "C" in fits:
        f = objective(w_all)
        r = minimize(f, e_gen.copy(), jac=True, method="SLSQP", bounds=[(0, None)] * K, constraints=[{"type": "eq", "fun": lambda c: c.sum() - 1, "jac": lambda c: np.ones(K)}],
                     options={"maxiter": a.maxiter, "ftol": 1e-12})
        record("C_pool", r.x, {"iters": int(r.nit), "message": str(r.message), "n_nonzero": int((r.x > 1e-6).sum())})
    if "D" in fits:
        single = {}
        for k in range(K):
            m = np.zeros(K, bool); m[k] = True; x0 = np.zeros(K); x0[k] = 1.0
            c, r = lbfgs(objective(w_all, mask=m), x0); v, _, _ = S.nll(c, w_all); single[ctxs[k]] = {"scale": float(c[k]), "fit_half_logp": -v / w_all.sum()}
        res["D_all_single"] = single; kb = max(single, key=lambda s: single[s]["fit_half_logp"])
        c = np.zeros(K); c[ctxs.index(kb)] = single[kb]["scale"]; record("D_single_scaled", c, {"persona": kb})
        c1 = np.zeros(K); c1[ctxs.index(kb)] = 1.0; per_answer[f"D persona at scale 1"] = holdout(c1)
        print(f"  D: best single persona {kb}, scale {single[kb]['scale']:.3f}; scales of all: median {np.median([v['scale'] for v in single.values()]):.3f}", flush=True)
    if "E" in fits:
        chosen = [gi]; c = e_gen.copy(); curve = []
        m = np.zeros(K, bool); m[gi] = True; c, _ = lbfgs(objective(w_all, mask=m), c)
        for step in range(a.greedy_steps):
            _, g, _ = S.nll(c, w_all); cand = [k for k in np.argsort(-np.abs(g)) if k not in chosen][:5]; best = None
            for k in cand:
                m = np.zeros(K, bool); m[chosen + [k]] = True; ck, _ = lbfgs(objective(w_all, mask=m), c.copy()); ck = np.where(m, ck, 0.0); v, _, _ = S.nll(ck, w_all)
                if best is None or v < best[0]: best = (v, k, ck)
            chosen.append(best[1]); c = best[2]; tot = holdout(c)
            curve.append({"k": len(chosen), "added": ctxs[best[1]], "c_added": float(c[best[1]]), "fit_half_logp": -best[0] / w_all.sum(), "holdout_logp": float(tot.sum() / nt_t.sum())})
            print(f"  E step {step+1}: + {ctxs[best[1]]} (c {c[best[1]]:+.3f}) -> hold-out {curve[-1]['holdout_logp']:.4f}", flush=True)
        res["E_greedy"] = curve; record("E_greedy_final", c, {"chosen": [ctxs[k] for k in chosen]})

    # uncertainties: question bootstrap of every hold-out row, and paired differences vs the arithmetic mixture
    keys = list(per_answer); sd, reps = boot([per_answer[k] for k in keys], [nt_t] * len(keys), q_t)
    ja = keys.index("arithmetic mixture")
    res["holdout"] = {k: {"logp": float(per_answer[k].sum() / nt_t.sum()), "se": float(sd[i]), "minus_arithmetic_se": float((reps[:, i] - reps[:, ja]).std())} for i, k in enumerate(keys)}
    np.savez(out / f"holdout_per_answer_{tag}.npz", rows=test_rows, qidx=q_t, n_tokens=nt_t, **{k.replace(" ", "_"): v for k, v in per_answer.items()})
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / f"config_{tag}.json").write_text(json.dumps({"args": vars(a), "git_commit": git, "hidden_dir": str(hdir), "command": " ".join(sys.argv)}, indent=2))
    (out / f"fits_{tag}.json").write_text(json.dumps(res, indent=1)); print(f"done -> {out} ({time.time()-t00:.0f}s)")
    for k, v in res["holdout"].items(): print(f"  {k:28s} {v['logp']:.4f} ± {v['se']:.4f}")


if __name__ == "__main__":
    main()
