"""MoralChoice: Spearman rank correlation with 95% bootstrap intervals (2026-11-24, author's request). Same pairing as
moralchoice_logistic.py: proxy on the original wording vs Instruct on the paraphrase (reference = Instruct original vs
paraphrase), 601 high-ambiguity scenarios, 2000 resamples over scenarios, seed 0; also the difference to the generic header,
paired over the same resamples.
    python scripts/moralchoice_spearman_ci.py   -> results/phase4/moralchoice/prefill/spearman_ci.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
P = Path("results/phase4/moralchoice/prefill")
def load(f): return {(r["model"], r["id"]): r["p1"] for r in json.loads((P / f).read_text())["per_scenario"] if r["ambiguity"] == "high"}
res = {}; names = ["reference", "soft", "geometric", "arithmetic", "hhh", "generic"]
for m in ["olmo", "qwen", "llama"]:
    O, Q = load(f"{m}_orig.json"), load(f"{m}_para.json")
    ids = sorted({i for (k, i) in Q if k == "instruct"} & {i for (k, i) in O if k == "instruct"}); t = np.array([Q[("instruct", i)] for i in ids])
    X = {n: np.array([O[("instruct" if n == "reference" else n, i)] for i in ids]) for n in names}
    rng = np.random.default_rng(0); B = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
    bs = {n: np.array([spearmanr(X[n][b], t[b])[0] for b in B]) for n in names}; res[m] = {}
    for n in names:
        d = bs[n] - bs["generic"]
        res[m][n] = {"spearman": float(spearmanr(X[n], t)[0]), "ci": [float(np.percentile(bs[n], 2.5)), float(np.percentile(bs[n], 97.5))],
                     "minus_generic_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
(P / "spearman_ci.json").write_text(json.dumps(res, indent=1))
for m, d in res.items():
    print(f"\n{m}"); [print(f"  {n:11s} {v['spearman']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]   minus generic [{v['minus_generic_ci'][0]:+.2f}, {v['minus_generic_ci'][1]:+.2f}]") for n, v in d.items()]
