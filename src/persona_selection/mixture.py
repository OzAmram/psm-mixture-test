"""Mixture-of-known-components fit (EM over weights only), KL goodness of fit, bootstrap, K sweep.

Notation (README, Phase 1): for response i and persona label s, l[i, s] = log P(a_i | q_i, s) is the
component log-likelihood and l0[i] = log P(a_i | q_i) is the generic-prompt log-likelihood.

  mixture:   log P_w(a_i) = logsumexp_s ( log w_s + l[i, s] )
  EM:        gamma[i, s] = softmax_s( log w_s + l[i, s] );   w_s <- mean_i gamma[i, s]
  KL:        D_hat = mean_i ( l0[i] - log P_w(a_i) )          (estimate of KL(P_0 || P_w), zero if exact)

Everything is numpy; inputs are plain arrays so the notebooks can call these on saved score matrices.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.special import logsumexp, softmax


def em_weights(L: np.ndarray, n_iter: int = 1000, tol: float = 1e-9, w0: np.ndarray | None = None, floor: float = 1e-12):
    """Fit mixture weights by EM with components fixed. L: (N, K) component log-likelihoods.

    Returns (w, info) where info has the log-likelihood trajectory and the responsibilities at the optimum.
    Weights are floored at `floor` so that a component can re-enter later (and to keep log w finite).
    """
    N, K = L.shape
    w = np.full(K, 1.0 / K) if w0 is None else np.asarray(w0, float) / np.sum(w0)
    ll_hist = []
    for it in range(n_iter):
        logr = np.log(np.maximum(w, floor))[None, :] + L          # (N, K)
        ll = logsumexp(logr, axis=1).mean()
        ll_hist.append(ll)
        gamma = softmax(logr, axis=1)
        w_new = gamma.mean(axis=0)
        w_new = np.maximum(w_new, floor); w_new /= w_new.sum()
        if it > 0 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            w = w_new
            break
        w = w_new
    logr = np.log(np.maximum(w, floor))[None, :] + L
    return w, {"loglik": np.array(ll_hist), "gamma": softmax(logr, axis=1), "n_iter": len(ll_hist)}


def mixture_loglik(L: np.ndarray, w: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Per-sample log P_w(a_i). L: (N, K)."""
    return logsumexp(np.log(np.maximum(w, floor))[None, :] + L, axis=1)


def kl_estimate(l0: np.ndarray, L: np.ndarray, w: np.ndarray, n_tokens: np.ndarray | None = None):
    """D_hat = mean_i [ l0_i - log P_w(a_i) ], in nats per response; and per token if n_tokens given."""
    d = l0 - mixture_loglik(L, w)
    out = {"kl_per_response": float(d.mean()), "kl_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")}
    if n_tokens is not None:
        out["kl_per_token"] = float(d.sum() / np.sum(n_tokens))
    return out


def fit_and_evaluate(L: np.ndarray, l0: np.ndarray, groups: np.ndarray, n_tokens: np.ndarray | None = None,
                     seed: int = 0, test_frac: float = 0.5):
    """Split by group (question id) into fit/held-out halves, fit w on one, evaluate KL on the other.

    Returns dict with w, held-out KL, and the in-sample KL for reference.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_groups = set(uniq[:n_test].tolist())
    test = np.array([g in test_groups for g in groups]); train = ~test
    w, info = em_weights(L[train])
    out = {"w": w, "n_train": int(train.sum()), "n_test": int(test.sum()), "em_iters": info["n_iter"]}
    out["heldout"] = kl_estimate(l0[test], L[test], w, None if n_tokens is None else n_tokens[test])
    out["insample"] = kl_estimate(l0[train], L[train], w, None if n_tokens is None else n_tokens[train])
    return out


def bootstrap_over_groups(L: np.ndarray, l0: np.ndarray, groups: np.ndarray, n_boot: int = 200, seed: int = 0,
                          n_tokens: np.ndarray | None = None, test_frac: float = 0.5):
    """Bootstrap (resample questions with replacement) of the split fit. Returns arrays of w and held-out KL."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    ws, kls = [], []
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in chosen])
        # give resampled duplicates distinct group ids so the split still separates questions
        g_new = np.concatenate([np.full(len(idx_by_group[g]), j) for j, g in enumerate(chosen)])
        r = fit_and_evaluate(L[idx], l0[idx], g_new, None if n_tokens is None else n_tokens[idx], seed=seed + b, test_frac=test_frac)
        ws.append(r["w"]); kls.append(r["heldout"]["kl_per_response"])
    return {"w": np.array(ws), "kl_heldout": np.array(kls)}


def k_sweep(L: np.ndarray, l0: np.ndarray, groups: np.ndarray, names: list[str], n_tokens: np.ndarray | None = None,
            seed: int = 0, max_k: int | None = None):
    """Held-out KL for every subset of components of size 1..K (exhaustive; K is small)."""
    K = L.shape[1]; max_k = K if max_k is None else max_k
    rows = []
    for k in range(1, max_k + 1):
        for subset in itertools.combinations(range(K), k):
            r = fit_and_evaluate(L[:, subset], l0, groups, n_tokens, seed=seed)
            rows.append({"k": k, "subset": [names[j] for j in subset], "w": {names[j]: float(r["w"][i]) for i, j in enumerate(subset)},
                         "kl_heldout": r["heldout"]["kl_per_response"], "kl_heldout_se": r["heldout"]["kl_se"],
                         "kl_heldout_per_token": r["heldout"].get("kl_per_token")})
    return rows


def synthetic_mixture_check(L: np.ndarray, source: np.ndarray, names: list[str]):
    """Calibration helper: `source[i]` is the true component index each sample was drawn from.

    Returns the true weights, the EM-recovered weights, and the confusion between true source and
    argmax responsibility. If the components are the actual generating distributions, the mixture
    is exact and the KL against the pooled generic distribution is zero by construction; here we
    check weight recovery and how separable the components are.
    """
    K = L.shape[1]
    true_w = np.bincount(source, minlength=K) / len(source)
    w, info = em_weights(L)
    pred = info["gamma"].argmax(axis=1)
    conf = np.zeros((K, K), int)
    for s, p in zip(source, pred):
        conf[s, p] += 1
    return {"true_w": dict(zip(names, true_w.tolist())), "em_w": dict(zip(names, w.tolist())),
            "max_abs_error": float(np.abs(true_w - w).max()), "confusion_true_x_pred": conf, "em_iters": info["n_iter"]}
