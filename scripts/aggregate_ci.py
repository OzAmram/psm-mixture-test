"""Uncertainties for the k-aggregated AUROCs in the write-up: item-level bootstrap 95% intervals, matched-question aggregation
(same questions for both teachers in every k-group), and a label-permutation null band for folded wrong ratios.
All on length-residualised per-answer log-ratios. Writes results/subliminal/aggregate_ci.json.
"""
import json, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
ROOT = "results/subliminal"; rng = np.random.default_rng(0); SUF = " Answer in two or three sentences of plain text."
def load(fn): return [json.loads(l) for l in open(f"{ROOT}/{fn}")]
def resid_rows(rows, pos, neg, a, b):
    rp = [r for r in rows if r["teacher"] == pos]; rn = [r for r in rows if r["teacher"] == neg]
    s = np.array([r["ll"][f"k0:{a}"] - r["ll"][f"k0:{b}"] for r in rp + rn]); L = np.array([r["n_tokens"] for r in rp + rn])[:, None]
    res = s - LinearRegression().fit(L, s).predict(L); return rp, rn, res[:len(rp)], res[len(rp):]
def agg(p, n, k, B=2000, rng=rng):
    if k == 1: return roc_auc_score([1]*len(p)+[0]*len(n), np.concatenate([p, n]))
    P = rng.choice(p, (B, k)).sum(1); N = rng.choice(n, (B, k)).sum(1); return roc_auc_score([1]*B+[0]*B, np.concatenate([P, N]))
def boot_ci(p, n, k, R=200):
    v = [agg(p[rng.choice(len(p), len(p))], n[rng.choice(len(n), len(n))], k, B=1000) for _ in range(R)]; return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
def matched(rp, rn, sp, sn, k, B=2000):
    key = lambda r: r.get("prompt", r.get("question", "")).replace(SUF, "")
    bp, bn = defaultdict(list), defaultdict(list)
    for r, v in zip(rp, sp): bp[key(r)].append(v)
    for r, v in zip(rn, sn): bn[key(r)].append(v)
    qs = [q for q in bp if q in bn]
    if k == 1:
        P = [rng.choice(bp[q]) for q in qs]; N = [rng.choice(bn[q]) for q in qs]; return roc_auc_score([1]*len(P)+[0]*len(N), np.concatenate([P, N])), len(qs)
    P, N = [], []
    for _ in range(B):
        qq = rng.choice(len(qs), min(k, len(qs)), replace=False); P.append(sum(rng.choice(bp[qs[i]]) for i in qq)); N.append(sum(rng.choice(bn[qs[i]]) for i in qq))
    return roc_auc_score([1]*B+[0]*B, np.concatenate([P, N])), len(qs)
def null_band(p, n, k, R=200):
    pool = np.concatenate([p, n]); out = []
    for _ in range(R):
        rng.shuffle(pool); a = agg(pool[:len(p)], pool[len(p):], k, B=1000); out.append(0.5 + abs(a - 0.5))
    return float(np.mean(out)), float(np.percentile(out, 95))
CELLS = [  # label, file, pos, neg, a, b
    ("text owl vs trains | OLMo base", "scores_multi_base_text.jsonl", "owl", "trains", "owl", "trains"),
    ("text owl vs trains | OLMo instruct", "scores_multi_instruct_text.jsonl", "owl", "trains", "owl", "trains"),
    ("text owl vs eagle | OLMo base", "scores_eagle20_olmo_base_text.jsonl", "owl", "eagle", "owl", "eagle"),
    ("text owl vs eagle | OLMo instruct", "scores_eagle20_olmo_instruct_text.jsonl", "owl", "eagle", "owl", "eagle"),
    ("text owl vs eagle | Qwen instruct", "scores_eagle20_qwen_instruct_text.jsonl", "owl", "eagle", "owl", "eagle"),
    ("text owl vs eagle | Qwen base", "scores_eagle20_qwen_base_text.jsonl", "owl", "eagle", "owl", "eagle"),
    ("numbers owl vs eagle | OLMo base", "scores_eagle20_olmo_base_numbers.jsonl", "owl", "eagle", "owl", "eagle"),
    ("numbers owl vs eagle | OLMo instruct", "scores_eagle20_olmo_instruct_numbers.jsonl", "owl", "eagle", "owl", "eagle"),
    ("text AF vs friend | OLMo base", "scores_t11_base_text.jsonl", "af", "af_friend", "af", "af_friend"),
    ("text AF vs friend | OLMo instruct", "scores_t9_instruct_text.jsonl+scores_t11_instruct_text.jsonl", "af", "af_friend", "af", "af_friend"),
    ("text AF vs none | OLMo base", "scores_t11_base_text.jsonl", "af", "control", "af", "neutral"),
    ("text AF vs none | OLMo instruct", "scores_t9_instruct_text.jsonl+scores_t11_instruct_text.jsonl", "af", "control", "af", "neutral"),
    ("text AF-stu vs ctl-stu | OLMo base", "scores_stu16_base_text.jsonl", "stu_af_text", "stu_control_text", "af", "neutral"),
    ("text AF-stu vs ctl-stu | OLMo instruct", "scores_stu9_instruct_text.jsonl", "stu_af_text", "stu_control_text", "af", "neutral"),
]
out = {}
print(f"{'cell':40s} | {'k=1':>5s} | {'k=10 [95% CI]':>19s} | {'k=30 [95% CI]':>19s} | {'matched-q k=10 / k=30 (nq)':>28s} | {'null folded mean/95th k=10, k=30':>34s}")
for label, fn, pos, neg, a, b in CELLS:
    rows = sum((load(f) for f in fn.split("+")), [])
    rp, rn, p, n = resid_rows(rows, pos, neg, a, b)
    r = {"k1": agg(p, n, 1)}
    for k in [10, 30]:
        r[f"k{k}"] = agg(p, n, k); r[f"k{k}_ci"] = boot_ci(p, n, k); r[f"k{k}_matched"], r["nq"] = matched(rp, rn, p, n, k); r[f"k{k}_null"] = null_band(p, n, k)
    out[label] = r
    print(f"{label:40s} | {r['k1']:.3f} | {r['k10']:.3f} [{r['k10_ci'][0]:.2f},{r['k10_ci'][1]:.2f}] | {r['k30']:.3f} [{r['k30_ci'][0]:.2f},{r['k30_ci'][1]:.2f}] | {r['k10_matched']:.3f} / {r['k30_matched']:.3f} ({r['nq']})     | {r['k10_null'][0]:.2f}/{r['k10_null'][1]:.2f}, {r['k30_null'][0]:.2f}/{r['k30_null'][1]:.2f}", flush=True)
json.dump(out, open(f"{ROOT}/aggregate_ci.json", "w"), indent=1); print("-> results/subliminal/aggregate_ci.json")
