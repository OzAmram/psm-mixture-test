"""Can PHLE scores be calibrated without labels on the audited model?

An auditor can generate labelled answers themselves, by prompting a model they control with each hypothesis as a system
prompt, score those with PHLE, and fit a per-answer correction Lambda' = a * Lambda + b (logistic regression, classes
weighted equally). For a sum over k answers the same correction is applied per answer: a * Lambda_k + k * b.
Two versions of "a model they control", on the existing score files (CPU only):

  same model     the audited model itself, prompted by the auditor: fit on half of the questions, test on the other half
                 (20 random splits, averaged).
  other model    a different post-trained model: fit on Olmo-generated answers and test on Qwen-generated ones, or the
                 reverse, with the same base-model scorer (the secret-prompt comparisons, the only ones scored both ways).

Reported on the test answers, per answer and for k = 30 bags of matched questions: log-loss (nats, classes weighted
equally; 0.693 = no information) of the raw score, of the transferred correction, and of a correction fitted on the
test answers themselves (the best possible); and the calibration slope / intercept of the corrected score on the test
answers (calibrated: 1 / 0).

    python scripts/calibration_transfer.py      # -> results/calibration_transfer.md
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

src = open("scripts/make_tables.py").read(); g = {"__name__": "mt"}
exec(src[:src.index("# ---------------------------------------------------------------- tables")], g)
load, raw_by_question, F = g["load"], g["raw_by_question"], g["F"]
K, N_BAGS, N_SPLITS = 30, 2000, 20

def fit(P, N):
    """Logistic fit of label on score with the two classes weighted equally -> (slope, intercept)."""
    x = np.r_[P, N][:, None]; y = np.r_[np.ones(len(P)), np.zeros(len(N))]
    w = np.r_[np.full(len(P), 0.5 / len(P)), np.full(len(N), 0.5 / len(N))] * len(y)
    m = LogisticRegression(C=1e6).fit(x, y, sample_weight=w); return m.coef_[0, 0], m.intercept_[0]

def logloss(P, N):
    """Balanced log-loss of scores read as log-odds for A."""
    sp = lambda z: np.logaddexp(0, z)
    return 0.5 * sp(-P).mean() + 0.5 * sp(N).mean()

def per_answer(byq, qs, pos, neg):
    return (np.concatenate([byq[q][pos][0][:, 0] for q in qs]), np.concatenate([byq[q][neg][0][:, 0] for q in qs]))

def bags(byq, qs, pos, neg, rng):
    P, N = np.empty(N_BAGS), np.empty(N_BAGS)
    for i in range(N_BAGS):
        qq = rng.choice(len(qs), min(K, len(qs)), replace=False)
        P[i] = sum(byq[qs[j]][pos][0][rng.integers(len(byq[qs[j]][pos][0])), 0] for j in qq)
        N[i] = sum(byq[qs[j]][neg][0][rng.integers(len(byq[qs[j]][neg][0])), 0] for j in qq)
    return P, N

def evaluate(cal, test_byq, test_qs, pos, neg, rng):
    """cal: (slope, intercept) fitted per answer elsewhere. Returns metrics on the test answers, k = 1 and k = 30."""
    a, b = cal; out = {}
    for k, (P, N) in [(1, per_answer(test_byq, test_qs, pos, neg)), (K, bags(test_byq, test_qs, pos, neg, rng))]:
        kk = min(k, len(test_qs)); Pc, Nc = a * P + kk * b, a * N + kk * b
        oa, ob = fit(P, N); s, i = fit(Pc, Nc)
        out[k] = dict(raw=logloss(P, N), cal=logloss(Pc, Nc), oracle=logloss(oa * P + ob, oa * N + ob), slope=s, icpt=i)
    return out

def mean_metrics(ms): return {k: {m: float(np.mean([x[k][m] for x in ms])) for m in ms[0][k]} for k in ms[0]}

COMPS = [  # (label, score-file key, pos, neg, header a, header b, modality)
    ("owl vs no prompt (text)", "olmo_base_text", "owl", "control", "owl", "neutral", "text"),
    ("owl vs trains (text)", "olmo_base_text", "owl", "trains", "owl", "trains", "text"),
    ("owl vs no prompt (numbers)", "olmo_base_num18", "owl", "control", "owl", "neutral", "numbers"),
    ("owl vs trains (numbers)", "olmo_base_num18", "owl", "trains", "owl", "trains", "numbers"),
    ("owl vs eagle (text)", "olmo_base_eagle", "owl", "eagle", "owl", "eagle", "text"),
    ("owl vs eagle (numbers)", "olmo_base_eagleN", "owl", "eagle", "owl", "eagle", "numbers"),
    ("secret harm vs no prompt (text)", "olmo_base_text11", "af", "control", "af", "neutral", "text"),
    ("secret harm vs secret friend (text)", "olmo_base_text11", "af", "af_friend", "af", "af_friend", "text"),
]
rows = []
for lab, key, pos, neg, a, b, mod in COMPS:   # same model, disjoint questions
    byq = raw_by_question(load(F[key], mod), pos, neg, [(a, b)]); qs = np.array(sorted(byq)); ms = []
    for s in range(N_SPLITS):
        rng = np.random.default_rng(s); perm = rng.permutation(len(qs)); cq, tq = list(qs[perm[: len(qs) // 2]]), list(qs[perm[len(qs) // 2:]])
        ms.append(evaluate(fit(*per_answer(byq, cq, pos, neg)), byq, tq, pos, neg, rng))
    rows.append(("same model (Olmo base scorer)", lab, mean_metrics(ms)))

# other model: same scorer, calibration answers from one family's generator, test answers from the other's
XF = {("Olmo base", "Olmo"): "olmo_base_text11", ("Olmo base", "Qwen"): "olmo_base_qwenT",
      ("Qwen base", "Qwen"): "qwen_base_qwenT", ("Qwen base", "Olmo"): "qwen_base_text"}
TP = {"Olmo": "", "Qwen": "qwen_"}
for scorer in ["Olmo base", "Qwen base"]:
    for cal_fam, test_fam in [("Olmo", "Qwen"), ("Qwen", "Olmo")]:
        for lab, pos, neg, a, b in [("secret harm vs no prompt (text)", "af", "control", "af", "neutral"), ("secret harm vs secret friend (text)", "af", "af_friend", "af", "af_friend")]:
            cb = raw_by_question(load(F[XF[(scorer, cal_fam)]], "text"), TP[cal_fam] + pos, TP[cal_fam] + neg, [(a, b)])
            tb = raw_by_question(load(F[XF[(scorer, test_fam)]], "text"), TP[test_fam] + pos, TP[test_fam] + neg, [(a, b)])
            cal = fit(*per_answer(cb, sorted(cb), TP[cal_fam] + pos, TP[cal_fam] + neg))
            m = evaluate(cal, tb, sorted(tb), TP[test_fam] + pos, TP[test_fam] + neg, np.random.default_rng(0))
            rows.append((f"{scorer} scorer: calibrate on {cal_fam} answers, test on {test_fam} answers", lab, m))

out = ["# Calibrating PHLE without labels on the audited model\n", "Generated by `scripts/calibration_transfer.py` (see its docstring). "
       "Log-loss in nats with the classes weighted equally (0.693 = no information; lower is better). raw = the PHLE score read directly as log-odds; "
       "transferred = with the per-answer slope and offset fitted on the calibration answers; best = refitted on the test answers. "
       "slope / intercept = calibration of the transferred score on the test answers (calibrated: 1 / 0).\n"]
for k in [1, K]:
    out.append(f"\n## {'per answer' if k == 1 else f'k = {K}'}\n\n| setup | comparison | log-loss raw | transferred | best | slope | intercept |\n|---|---|---|---|---|---|---|")
    for setup, lab, m in rows:
        r = m[k]; out.append(f"| {setup} | {lab} | {r['raw']:.3f} | {r['cal']:.3f} | {r['oracle']:.3f} | {r['slope']:.2f} | {r['icpt']:+.2f} |")
open("results/calibration_transfer.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
