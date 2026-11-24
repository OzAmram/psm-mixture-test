"""MoralChoice: logistic-fit metric (2026-11-24, author's question whether R^2 suits preferences clustered at 0 and 1).
For each proxy, fit Instruct's P(action1) t as sigmoid(a * logit(p_proxy) + b) (soft-target cross-entropy, 2 parameters;
a acts as a temperature, so a proxy that is right but too mild is not penalised). Score = share of the predictable
information captured: 1 - KL(t || fit) / KL(t || constant mean), 0 = no better than Instruct's base rate, 1 = exact.
Also Spearman rank correlation and agreement on the preferred action. Like-for-like with the reference level: proxy on the
original wording vs Instruct on the paraphrase; reference = Instruct original vs Instruct paraphrase. 95% bootstrap over
scenarios (1000, seed 0; the logistic fit is redone in each resample).
    python scripts/moralchoice_logistic.py   -> results/phase4/moralchoice/prefill/logistic_metric.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

P = Path("results/phase4/moralchoice/prefill"); EPS = 1e-4
def load(f): return {(r["model"], r["id"]): r["p1"] for r in json.loads((P / f).read_text())["per_scenario"] if r["ambiguity"] == "high"}
def ce(t, q): q = np.clip(q, 1e-9, 1 - 1e-9); return -(t * np.log(q) + (1 - t) * np.log(1 - q)).mean()
def score(p, t):
    x = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    f = lambda w: ce(t, 1 / (1 + np.exp(-(w[0] * x + w[1]))))
    w = minimize(f, [1.0, 0.0], method="L-BFGS-B").x; H = ce(t, t)
    return 1 - (f(w) - H) / (ce(t, np.full_like(t, t.mean())) - H), w
res = {}
for m in ["olmo", "qwen", "llama"]:
    O, Q = load(f"{m}_orig.json"), load(f"{m}_para.json"); ids = sorted({i for (k, i) in Q if k == "instruct"} & {i for (k, i) in O if k == "instruct"})
    tq = np.array([Q[("instruct", i)] for i in ids]); rng = np.random.default_rng(0); B = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
    res[m] = {"n": len(ids)}
    for name in ["reference", "generic", "hhh", "arithmetic", "geometric", "soft"]:
        p = np.array([O[("instruct" if name == "reference" else name, i)] for i in ids])
        s, w = score(p, tq); bs = [score(p[b], tq[b])[0] for b in B]
        res[m][name] = {"info_share": float(s), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "slope": float(w[0]),
                        "spearman": float(spearmanr(p, tq)[0]), "agree": float(((p > 0.5) == (tq > 0.5)).mean()), "r2": float(np.corrcoef(p, tq)[0, 1] ** 2)}
(P / "logistic_metric.json").write_text(json.dumps(res, indent=1))
for m, d in res.items():
    print(f"\n{m} (n = {d['n']}; proxy original vs Instruct paraphrase)\n  {'':11s} info share [95%]      slope  Spearman  agree   R^2")
    for k, v in d.items():
        if k != "n": print(f"  {k:11s} {v['info_share']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]   {v['slope']:5.2f}   {v['spearman']:.2f}    {v['agree']:.2f}   {v['r2']:.2f}")
