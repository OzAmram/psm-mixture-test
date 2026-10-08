"""Single entry point for every AUROC table in the write-up, with ONE uncertainty convention (v2, 2026-10-06).

Every likelihood cell is  value ± 1 sigma  where sigma is the standard deviation over R question-level (cluster) bootstrap
replicates: questions are resampled with replacement jointly for both teachers (and jointly across headers when several
headers enter one statistic), and the WHOLE statistic is recomputed on the resampled answers, including the length
regression (refitted inside every replicate). The statistics:
  per answer   AUROC of the raw per-answer log-ratio (a - b), teacher A vs teacher B (MT_LENGTH_REGRESS=1 residualises on length);
  k = 10, 30   matched-question aggregation: a bag of k distinct questions, one random answer per question from EACH teacher,
               the k residualised log-ratios summed; AUROC over B bags per side.
  folded       mean over the unrelated ("wrong") header ratios of 0.5 + |AUROC - 0.5|, folded INSIDE each replicate.
  null band    label-permutation null of the exact final estimator: teacher labels shuffled within each question, then the
               same matched-bag statistic (folded); reported as mean and 95th percentile over P permutations.
Every cell uses its own random seed (derived from the cell label), so a cell does not change when other cells are added.
One text filter (the current regex from scripts/subliminal_generate.py) is applied to every text pool at load time; the saved
pools were already filtered at generation time with earlier versions of the regex, so this removes only a few more answers.
Prompted-classifier rows (saved AUROC and trial counts only) get the Hanley-McNeil standard error over their trials, which is a
Monte-Carlo error conditional on the finite answer pool, not a data uncertainty; they are marked "(HM)".
Chat-template scorers use the exact-continuation rescoring (scores_exact_*.jsonl); cells from older space-prefixed files are
marked (old).

    python scripts/make_tables.py            -> results/tables.json, results/tables.md
"""
import os, json, sys, math, zlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from subliminal_generate import TEXT_FILTER
from persona_selection.mixture import em_weights
ROOT = Path("results/subliminal"); SUF = " Answer in two or three sentences of plain text."
R_BOOT = 200; B_BAG = 1000; P_NULL = 200

def seed_of(label): return zlib.crc32(label.encode()) & 0xFFFFFFFF

# ---------------------------------------------------------------- data
_cache = {}
def load(fn, modality="text"):
    if isinstance(fn, (list, tuple)):
        parts = [load(x, modality) for x in fn]; parts = [x for x in parts if x]
        return sum(parts, []) if parts else None
    key = (fn, modality)
    if key in _cache: return _cache[key]
    f = ROOT / fn
    if not f.exists(): _cache[key] = None; return None
    rows = [json.loads(l) for l in open(f)]
    if modality == "text":
        rows = [r for r in rows if not TEXT_FILTER.search(r["completion"])]
    _cache[key] = rows; return rows

def first_existing(*fns):
    for fn in fns:
        if isinstance(fn, (list, tuple)):
            if all((ROOT / x).exists() for x in fn): return fn
        elif (ROOT / fn).exists(): return fn
    return fns[-1]

def qkey(r): return r.get("prompt", r.get("question", "")).replace(SUF, "")

def raw_by_question(rows, pos, neg, pairs):
    """For each question: per teacher, raw log-ratios [n, n_pairs] for every (a, b) in pairs, and token lengths [n]."""
    out = {}
    for t in (pos, neg):
        for r in rows:
            if r["teacher"] != t: continue
            if any(f"k0:{a}" not in r["ll"] or f"k0:{b}" not in r["ll"] for a, b in pairs): continue
            d = out.setdefault(qkey(r), {pos: ([], []), neg: ([], [])})
            d[t][0].append([r["ll"][f"k0:{a}"] - r["ll"][f"k0:{b}"] for a, b in pairs]); d[t][1].append(float(r["n_tokens"]))
    qs = [q for q, d in out.items() if d[pos][0] and d[neg][0]]
    return {q: {t: (np.array(out[q][t][0]), np.array(out[q][t][1])) for t in (pos, neg)} for q in qs}

NO_LENGTH = os.environ.get("MT_LENGTH_REGRESS") != "1"   # default: raw log-ratios. MT_LENGTH_REGRESS=1: length regressed out (writes tables_lenreg.*)

def residualise(byq, qs, pos, neg):
    """Refit the length regression on the (resampled) pooled sample; return per-question residual arrays [n, n_pairs]."""
    if NO_LENGTH: return {q: {t: byq[q][t][0] for t in (pos, neg)} for q in qs}
    S = np.concatenate([byq[q][t][0] for q in qs for t in (pos, neg)]); L = np.concatenate([byq[q][t][1] for q in qs for t in (pos, neg)])
    X = np.c_[np.ones(len(L)), L]; beta = np.linalg.lstsq(X, S, rcond=None)[0]
    return {q: {t: byq[q][t][0] - np.c_[np.ones(len(byq[q][t][1])), byq[q][t][1]] @ beta for t in (pos, neg)} for q in qs}

def bag_auc(res, qs, pos, neg, k, rng, col=0):
    nq = len(qs); kk = min(k, nq); Pb, Nb = np.empty(B_BAG), np.empty(B_BAG)
    for i in range(B_BAG):
        qq = rng.choice(nq, kk, replace=False)
        Pb[i] = sum(res[qs[j]][pos][rng.integers(len(res[qs[j]][pos])), col] for j in qq)
        Nb[i] = sum(res[qs[j]][neg][rng.integers(len(res[qs[j]][neg])), col] for j in qq)
    return roc_auc_score([1]*B_BAG+[0]*B_BAG, np.concatenate([Pb, Nb]))

def per_answer_auc(res, qs, pos, neg, col=0):
    P = np.concatenate([res[q][pos][:, col] for q in qs]); N = np.concatenate([res[q][neg][:, col] for q in qs])
    return roc_auc_score([1]*len(P)+[0]*len(N), np.concatenate([P, N]))

def statistic(byq, qs, pos, neg, rng, ks=(10, 30), fold=False):
    """Per-answer and bag AUROCs on one (possibly resampled) question list. With fold=True the arrays hold several wrong
    ratios and the statistic is the mean over them of the folded AUROC."""
    res = residualise(byq, qs, pos, neg); ncol = next(iter(res.values()))[pos].shape[1]; out = {}
    fns = [("k1", lambda c: per_answer_auc(res, qs, pos, neg, c))] + [(f"k{k}", (lambda k: lambda c: bag_auc(res, qs, pos, neg, k, rng, c))(k)) for k in ks]
    for name, fn in fns:
        vals = [fn(c) for c in range(ncol)]
        out[name] = float(np.mean([0.5 + abs(v - 0.5) for v in vals])) if fold else vals[0]
    return out

def cell(rows, pos, neg, a, b, label, ks=(10, 30), fold_headers=None):
    pairs = [(h, "neutral") for h in fold_headers] if fold_headers else [(a, b)]
    byq = raw_by_question(rows, pos, neg, pairs)
    if len(byq) < 5: return None
    qs = sorted(byq); rng = np.random.default_rng(seed_of(label))
    point = statistic(byq, qs, pos, neg, rng, ks, fold=bool(fold_headers)); reps = {k: [] for k in point}
    for _ in range(R_BOOT):
        pick = rng.choice(len(qs), len(qs), replace=True); sub = {f"{qs[i]}#{j}": byq[qs[i]] for j, i in enumerate(pick)}
        st = statistic(sub, sorted(sub), pos, neg, rng, ks, fold=bool(fold_headers))
        for k in st: reps[k].append(st[k])
    return {k: (point[k], float(np.std(reps[k]))) for k in point} | {"n_q": len(qs), "n_pos": int(sum(len(byq[q][pos][1]) for q in qs)), "n_neg": int(sum(len(byq[q][neg][1]) for q in qs))}

def null_band(rows, pos, neg, a, b, label, ks=(10, 30), fold_headers=None):
    """Within-question label permutation of the exact final estimator (folded): mean and 95th percentile."""
    pairs = [(h, "neutral") for h in fold_headers] if fold_headers else [(a, b)]
    byq = raw_by_question(rows, pos, neg, pairs); qs = sorted(byq); rng = np.random.default_rng(seed_of(label + "|null"))
    vals = {f"k{k}": [] for k in (1,) + tuple(ks)}
    for _ in range(P_NULL):
        sub = {}
        for q in qs:
            S = np.concatenate([byq[q][pos][0], byq[q][neg][0]]); L = np.concatenate([byq[q][pos][1], byq[q][neg][1]]); perm = rng.permutation(len(S)); n = len(byq[q][pos][1])
            sub[q] = {pos: (S[perm[:n]], L[perm[:n]]), neg: (S[perm[n:]], L[perm[n:]])}
        st = statistic(sub, qs, pos, neg, rng, ks, fold=True)
        for k in st: vals[k].append(st[k])
    return {k: (float(np.mean(v)), float(np.percentile(v, 95))) for k, v in vals.items()}

def fmt(c, keys=("k1", "k10", "k30")):
    if c is None: return "–"
    return " / ".join(f"{c[k][0]:.2f}±{c[k][1]:.2f}" for k in keys)

def hm_se(auc, n1, n2):
    q1 = auc / (2 - auc); q2 = 2 * auc * auc / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n2 - 1) * (q2 - auc * auc)) / (n1 * n2))

def cls_cell(fn, case, ks=("1", "10", "30"), key="auroc", n_default=200):
    f = ROOT / fn
    if not f.exists(): return "–"
    d = json.load(open(f)); c = d.get(case) or next((v for k, v in d.items() if case in k), None)
    if not c: return "–"
    parts = []
    for k in ks:
        e = c.get(k) or c.get(int(k)) if isinstance(c, dict) else None
        if not e or key not in e: parts.append("–"); continue
        n = e.get("n", 2 * n_default); parts.append(f"{e[key]:.2f}±{hm_se(max(e[key], 1 - e[key]), n // 2, n // 2):.2f}")
    return " / ".join(parts) + " (HM)"

# ---------------------------------------------------------------- file map (exact rescoring preferred)
F = {
    "olmo_base_text":    "scores_multi_base_text.jsonl",
    "olmo_base_text11":  "scores_t11_base_text.jsonl",
    "olmo_base_eagle":   "scores_eagle20_olmo_base_text.jsonl",
    "olmo_base_eagleN":  "scores_eagle20_olmo_base_numbers.jsonl",
    "olmo_base_num18":   "scores_num18_base.jsonl",
    "olmo_base_stu":     "scores_stu16_base_text.jsonl",
    "olmo_base_qwenT":   "scores_qwenT_olmo_base.jsonl",
    "olmo_inst_text":    first_existing("scores_exact_instruct_text.jsonl", "scores_multi_instruct_text.jsonl"),
    "olmo_inst_text11":  first_existing("scores_exact_instruct_text.jsonl", ["scores_t9_instruct_text.jsonl", "scores_t11_instruct_text.jsonl"]),
    "olmo_inst_eagle":   first_existing("scores_exact_instruct_text.jsonl", "scores_eagle20_olmo_instruct_text.jsonl"),
    "olmo_inst_eagleN":  first_existing("scores_exact_instruct_numbers.jsonl", "scores_eagle20_olmo_instruct_numbers.jsonl"),
    "olmo_inst_num18":   first_existing("scores_exact_instruct_numbers.jsonl", "scores_num18_instruct.jsonl"),
    "olmo_inst_stu":     first_existing("scores_exact_instruct_students.jsonl", ["scores_stu9_instruct_text.jsonl", "scores_stu12_instruct_text.jsonl"]),
    "olmo_inst_qwenT":   first_existing("scores_exact_qwenT_olmo_instruct.jsonl", "scores_qwenT_olmo_instruct.jsonl"),
    "qwen_base_text":    "scores_multi_qwen_text.jsonl",
    "qwen_base_eagle":   "scores_eagle20_qwen_base_text.jsonl",
    "qwen_base_eagleN":  "scores_eagle20_qwen_base_numbers.jsonl",
    "qwen_base_qwenT":   "scores_qwenT_qwen_base.jsonl",
    "qwen_inst_text":    first_existing("scores_exact_qweninst_text.jsonl", "scores_multi_qweninst_text.jsonl"),
    "qwen_inst_eagle":   first_existing("scores_exact_qweninst_text.jsonl", "scores_eagle20_qwen_instruct_text.jsonl"),
    "qwen_inst_eagleN":  "scores_eagle20_qwen_instruct_numbers.jsonl",            # not rescored -> (old)
    "qwen_inst_stu":     "scores_multi_qweninst_students.jsonl",                 # not rescored -> (old)
    "qwen_inst_qwenT":   first_existing("scores_exact_qweninst_text.jsonl", "scores_qwenT_qwen_instruct.jsonl"),
    "nm_base": "scores_nm_base_text.jsonl", "nm_inst": "scores_nm_instruct_text.jsonl",
    "nm_baseN": "scores_nm_base_numbers.jsonl", "nm_instN": "scores_nm_instruct_numbers.jsonl",
}
def tag(key): return "" if "base" in key or ("exact" in str(F[key])) or key.startswith("nm_") else " (old)"

# ---------------------------------------------------------------- tables
T = {}; MD = []; NULLS = {}
def table(name, header, rows, note=None):
    T[name] = rows; MD.append(f"\n**{name}**  (per answer / k=10 / k=30, ±1σ question bootstrap{'; ' + note if note else ''})\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows: MD.append("| " + " | ".join(r) + " |")

def LR(key, pos, neg, a, b, modality="text", fold=None):
    rows = load(F[key], modality)
    if not rows: return "–"
    label = f"{key}|{pos}|{neg}|{a}|{b}|{modality}|{fold}"
    return fmt(cell(rows, pos, neg, a, b, label, fold_headers=fold)) + tag(key)

def NULL(key, pos, neg, a, b, modality="text", fold=None):
    rows = load(F[key], modality); label = f"{key}|{pos}|{neg}|{a}|{b}|{modality}|{fold}"
    nb = null_band(rows, pos, neg, a, b, label, fold_headers=fold); NULLS[label] = nb
    return " / ".join(f"{nb[k][0]:.2f} (95th {nb[k][1]:.2f})" for k in ("k1", "k10", "k30"))

# 3.3 owl vs trains
table("3.3 owl vs trains (text)", ["scorer", "owl − trains (pairwise)", "owl vs no prompt (owl − neutral)"], [
    ["OLMo Instruct (generator)", LR("olmo_inst_text", "owl", "trains", "owl", "trains"), LR("olmo_inst_text", "owl", "control", "owl", "neutral")],
    ["OLMo base, persona headers", LR("olmo_base_text", "owl", "trains", "owl", "trains"), LR("olmo_base_text", "owl", "control", "owl", "neutral")],
    ["prompted 7B classifier", cls_cell("classifier_owl_trains.json", "owl teacher vs trains"), cls_cell("classifier_sanity.json", "owl teacher vs no prompt", ks=("1", "10"))],
    ["GPT-4.1, k answers in one prompt", cls_cell("classifier_gpt-4.1.json", "text: owl vs trains", n_default=100), cls_cell("classifier_gpt-4.1.json", "text: owl vs no prompt", n_default=100)],
    ["GPT-4.1, per-answer pooled", cls_cell("classifier_gpt-4.1_agg.json", "text: owl vs trains", n_default=300), cls_cell("classifier_gpt-4.1_agg.json", "text: owl vs no prompt", n_default=300)],
])
# 3.4 owl vs eagle
W = ["trains", "af", "hhh"]
table("3.4 owl vs eagle", ["scorer", "text: owl − eagle", "text: owl-free ratios (mean of 3, folded)", "numbers: owl − eagle", "numbers: owl-free ratios (mean of 3, folded)"], [
    ["OLMo Instruct (generator)", LR("olmo_inst_eagle", "owl", "eagle", "owl", "eagle"), LR("olmo_inst_eagle", "owl", "eagle", None, None, fold=W), LR("olmo_inst_eagleN", "owl", "eagle", "owl", "eagle", "numbers"), LR("olmo_inst_eagleN", "owl", "eagle", None, None, "numbers", fold=W)],
    ["OLMo base", LR("olmo_base_eagle", "owl", "eagle", "owl", "eagle"), LR("olmo_base_eagle", "owl", "eagle", None, None, fold=W), LR("olmo_base_eagleN", "owl", "eagle", "owl", "eagle", "numbers"), LR("olmo_base_eagleN", "owl", "eagle", None, None, "numbers", fold=W)],
    ["Qwen Instruct", LR("qwen_inst_eagle", "owl", "eagle", "owl", "eagle"), LR("qwen_inst_eagle", "owl", "eagle", None, None, fold=W), LR("qwen_inst_eagleN", "owl", "eagle", "owl", "eagle", "numbers"), LR("qwen_inst_eagleN", "owl", "eagle", None, None, "numbers", fold=W)],
    ["Qwen base", LR("qwen_base_eagle", "owl", "eagle", "owl", "eagle"), LR("qwen_base_eagle", "owl", "eagle", None, None, fold=W), LR("qwen_base_eagleN", "owl", "eagle", "owl", "eagle", "numbers"), LR("qwen_base_eagleN", "owl", "eagle", None, None, "numbers", fold=W)],
    ["no-signal null of the folded statistic (mean, 95th pct)", NULL("olmo_base_eagle", "owl", "eagle", None, None, fold=W), "", NULL("olmo_base_eagleN", "owl", "eagle", None, None, "numbers", fold=W), ""],
    ["prompted 7B classifier", cls_cell("classifier_owl_eagle.json", "owl teacher vs eagle"), "", cls_cell("classifier_owl_eagle_numbers.json", "owl teacher vs eagle"), ""],
    ["GPT-4.1, one prompt", cls_cell("classifier_gpt-4.1_eagle.json", "text: owl vs eagle", n_default=100), "", cls_cell("classifier_gpt-4.1_eagle.json", "numbers: owl vs eagle", n_default=100), ""],
    ["GPT-4.1, per-answer pooled", cls_cell("classifier_gpt-4.1_eagle_agg.json", "text: owl vs eagle", n_default=300), "", cls_cell("classifier_gpt-4.1_eagle_agg.json", "numbers: owl vs eagle", n_default=300), ""],
], note="null row = within-question label permutation of the exact folded estimator")
# 3.4b no-mention teachers
table("3.4b 'do not mention X' teachers", ["scorer", "text: owl_nm − eagle_nm (nm headers)", "text: plain owl − eagle headers", "text: wrong ratio trains_nm − neutral, folded", "numbers: owl_nm − eagle_nm (nm headers)", "numbers: plain owl − eagle headers", "numbers: wrong ratio, folded"], [
    ["OLMo Instruct (generator)", LR("nm_inst", "owl_nm", "eagle_nm", "owl_nm", "eagle_nm"), LR("nm_inst", "owl_nm", "eagle_nm", "owl", "eagle"), LR("nm_inst", "owl_nm", "eagle_nm", None, None, fold=["trains_nm"]),
     LR("nm_instN", "owl_nm_numbers", "eagle_nm_numbers", "owl_nm", "eagle_nm", "numbers"), LR("nm_instN", "owl_nm_numbers", "eagle_nm_numbers", "owl", "eagle", "numbers"), LR("nm_instN", "owl_nm_numbers", "eagle_nm_numbers", None, None, "numbers", fold=["trains_nm"])],
    ["OLMo base", LR("nm_base", "owl_nm", "eagle_nm", "owl_nm", "eagle_nm"), LR("nm_base", "owl_nm", "eagle_nm", "owl", "eagle"), LR("nm_base", "owl_nm", "eagle_nm", None, None, fold=["trains_nm"]),
     LR("nm_baseN", "owl_nm_numbers", "eagle_nm_numbers", "owl_nm", "eagle_nm", "numbers"), LR("nm_baseN", "owl_nm_numbers", "eagle_nm_numbers", "owl", "eagle", "numbers"), LR("nm_baseN", "owl_nm_numbers", "eagle_nm_numbers", None, None, "numbers", fold=["trains_nm"])],
    ["OLMo base: owl_nm vs trains_nm | owl_nm vs no prompt", LR("nm_base", "owl_nm", "trains_nm", "owl_nm", "trains_nm"), LR("nm_base", "owl_nm", "control", "owl_nm", "neutral"), "", LR("nm_baseN", "owl_nm_numbers", "trains_nm_numbers", "owl_nm", "trains_nm", "numbers"), LR("nm_baseN", "owl_nm_numbers", "control", "owl_nm", "neutral", "numbers"), ""],
    ["no-signal null (base, folded; mean, 95th pct)", NULL("nm_base", "owl_nm", "eagle_nm", None, None, fold=["trains_nm"]), "", "", NULL("nm_baseN", "owl_nm_numbers", "eagle_nm_numbers", None, None, "numbers", fold=["trains_nm"]), "", ""],
])
# 3.5 AF vs friend
table("3.5 secret harm vs secret friend", ["scorer", "text: AF − friend", "text: AF − neutral (vs no prompt)", "text: wrong ratio AF − HHH on AF vs friend (signed)", "numbers: AF − friend"], [
    ["OLMo Instruct (generator)", LR("olmo_inst_text11", "af", "af_friend", "af", "af_friend"), LR("olmo_inst_text11", "af", "control", "af", "neutral"), LR("olmo_inst_text11", "af", "af_friend", "af", "hhh"), LR("olmo_inst_num18", "af", "af_friend", "af", "af_friend", "numbers")],
    ["OLMo base", LR("olmo_base_text11", "af", "af_friend", "af", "af_friend"), LR("olmo_base_text11", "af", "control", "af", "neutral"), LR("olmo_base_text11", "af", "af_friend", "af", "hhh"), LR("olmo_base_num18", "af", "af_friend", "af", "af_friend", "numbers")],
    ["Qwen Instruct", LR("qwen_inst_text", "af", "af_friend", "af", "af_friend"), LR("qwen_inst_text", "af", "control", "af", "neutral"), LR("qwen_inst_text", "af", "af_friend", "af", "hhh"), "–"],
    ["Qwen base", LR("qwen_base_text", "af", "af_friend", "af", "af_friend"), LR("qwen_base_text", "af", "control", "af", "neutral"), LR("qwen_base_text", "af", "af_friend", "af", "hhh"), "–"],
    ["no-signal null (base, folded; mean, 95th pct)", NULL("olmo_base_text11", "af", "af_friend", "af", "af_friend"), "", "", NULL("olmo_base_num18", "af", "af_friend", "af", "af_friend", "numbers")],
    ["prompted 7B classifier", cls_cell("classifier_baseline.json", "AF teacher vs friend"), cls_cell("classifier_baseline.json", "AF teacher vs no prompt"), "", "–"],
    ["GPT-4.1, one prompt", cls_cell("classifier_gpt-4.1.json", "text: AF vs friend", n_default=100), cls_cell("classifier_gpt-4.1.json", "text: AF vs no prompt", n_default=100), "", cls_cell("classifier_gpt-4.1.json", "numbers: AF vs friend", n_default=100)],
    ["GPT-4.1, per-answer pooled", cls_cell("classifier_gpt-4.1_agg.json", "text: AF vs friend", n_default=300), cls_cell("classifier_gpt-4.1_agg.json", "text: AF vs no prompt", n_default=300), "", "–"],
])
# 3.7 cross-family
table("3.7 cross-family: OLMo-Instruct teacher answers", ["scorer", "AF − neutral (AF vs no prompt)", "AF − friend (AF vs friend)"], [
    ["OLMo Instruct (generator)", LR("olmo_inst_text11", "af", "control", "af", "neutral"), LR("olmo_inst_text11", "af", "af_friend", "af", "af_friend")],
    ["OLMo base", LR("olmo_base_text11", "af", "control", "af", "neutral"), LR("olmo_base_text11", "af", "af_friend", "af", "af_friend")],
    ["Qwen base", LR("qwen_base_text", "af", "control", "af", "neutral"), LR("qwen_base_text", "af", "af_friend", "af", "af_friend")],
    ["Qwen Instruct", LR("qwen_inst_text", "af", "control", "af", "neutral"), LR("qwen_inst_text", "af", "af_friend", "af", "af_friend")],
])
table("3.7 cross-family: Qwen-Instruct teacher answers", ["scorer", "AF − neutral (AF vs no prompt)", "AF − friend (AF vs friend)"], [
    ["Qwen Instruct (generator)", LR("qwen_inst_qwenT", "qwen_af", "qwen_control", "af", "neutral"), LR("qwen_inst_qwenT", "qwen_af", "qwen_af_friend", "af", "af_friend")],
    ["Qwen base", LR("qwen_base_qwenT", "qwen_af", "qwen_control", "af", "neutral"), LR("qwen_base_qwenT", "qwen_af", "qwen_af_friend", "af", "af_friend")],
    ["OLMo base", LR("olmo_base_qwenT", "qwen_af", "qwen_control", "af", "neutral"), LR("olmo_base_qwenT", "qwen_af", "qwen_af_friend", "af", "af_friend")],
    ["OLMo Instruct", LR("olmo_inst_qwenT", "qwen_af", "qwen_control", "af", "neutral"), LR("olmo_inst_qwenT", "qwen_af", "qwen_af_friend", "af", "af_friend")],
    ["Qwen2.5-7B-Instruct prompted classifier (Qwen answers)", cls_cell("classifier_qwen_teacher.json", "Qwen: AF teacher vs no prompt"), cls_cell("classifier_qwen_teacher.json", "Qwen: AF teacher vs friend")],
])
# 3.8 students
table("3.8 students (no prompt at inference)", ["scorer", "AF student vs friend student (AF − friend)", "AF student vs control student (AF − neutral)"], [
    ["OLMo Instruct (the students' initialisation)", LR("olmo_inst_stu", "stu_af_text", "stu_friend_text", "af", "af_friend"), LR("olmo_inst_stu", "stu_af_text", "stu_control_text", "af", "neutral")],
    ["OLMo base", LR("olmo_base_stu", "stu_af_text", "stu_friend_text", "af", "af_friend"), LR("olmo_base_stu", "stu_af_text", "stu_control_text", "af", "neutral")],
    ["Qwen Instruct", LR("qwen_inst_stu", "stu_af_text", "stu_friend_text", "af", "af_friend"), LR("qwen_inst_stu", "stu_af_text", "stu_control_text", "af", "neutral")],
    ["prompted 7B classifier", "–", cls_cell("classifier_baseline.json", "AF student vs control student")],
    ["GPT-4.1, one prompt", "–", cls_cell("classifier_gpt-4.1.json", "text: AF student vs control student", n_default=100)],
])

# ---------------------------------------------------------------- 3.6 eight-way identification from the EXACT instruct file
SRC = ["control", "hhh_teacher", "af", "af_friend", "af_resent", "af_owl", "owl", "trains"]
HYP = ["neutral", "hhh", "af", "af_friend", "af_resent", "af_owl", "owl", "trains"]
def multiway_tables():
    inst = load("scores_exact_instruct_text.jsonl"); base = load("scores_multi_base_text.jsonl")
    if not inst or not base: return
    def mat(rows):
        byq = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["teacher"] in SRC and all(f"k0:{h}" in r["ll"] for h in HYP): byq[qkey(r)][r["teacher"]].append([r["ll"][f"k0:{h}"] for h in HYP])
        return {q: {s: np.array(v) for s, v in d.items()} for q, d in byq.items() if all(s in d for s in SRC)}
    def argmax_acc(M, qs, rng, mu, ks=(1, 5, 10, 30), B=400):
        out = {}
        for k in ks:
            acc = np.zeros(len(SRC)); kk = min(k, len(qs))
            for i, s in enumerate(SRC):
                for _ in range(B):
                    qq = rng.choice(len(qs), kk, replace=False); v = sum(M[qs[j]][s][rng.integers(len(M[qs[j]][s]))] for j in qq) - kk * mu
                    acc[i] += np.argmax(v) == i
            out[k] = acc / B
        return out
    def em_rec(M, qs, rng, comp, N=50, B=150):
        W = []
        for _ in range(B):
            def draw(s):
                q = qs[rng.integers(len(qs))]; return M[q][s][rng.integers(len(M[q][s]))]
            L = np.concatenate([np.array([draw(s) for _ in range(int(round(N * f)))]) for s, f in comp.items() if f > 0]); w, _ = em_weights(L); W.append(w)
        return np.mean(W, 0)
    rows_md = []
    for name, M, calib in [("Instruct with the real prompts, no calibration (exact scoring)", mat(inst), False), ("base model, persona headers, calibrated on a disjoint half of the QUESTIONS", mat(base), True)]:
        qs = sorted(M); rng = np.random.default_rng(seed_of("8way|" + name)); reps = []
        for rep in range(30):
            pick = rng.choice(len(qs), len(qs), replace=True); sub = {f"{qs[i]}#{j}": M[qs[i]] for j, i in enumerate(pick)}; sq = sorted(sub)
            if calib:
                half = len(sq) // 2; cal, ev = sq[:half], sq[half:]
                mu = np.mean([sub[q][s].mean(0) for q in cal for s in SRC], axis=0)
            else: mu = np.zeros(len(HYP)); ev = sq
            acc = argmax_acc(sub, ev, rng, mu); reps.append([acc[k].mean() for k in (1, 5, 10, 30)])
        reps = np.array(reps); m, sd = reps.mean(0), reps.std(0)
        rows_md.append([name] + [f"{m[i]:.2f}±{sd[i]:.2f}" for i in range(4)])
        if not calib:
            recs = {}
            for comp in [{"control": 0.8, "af": 0.2}, {"control": 0.5, "af": 0.5}, {"hhh_teacher": 0.8, "af_friend": 0.2}]:
                w = em_rec(M, qs, rng, comp); tgt = [s for s in comp if s not in ("control", "hhh_teacher")][0]
                recs[" + ".join(f"{int(100*f)}% {s}" for s, f in comp.items())] = float(w[SRC.index(tgt)])
            T["3.6 mixture recovery (instruct, exact)"] = recs
            MD.append("\n**3.6 mixture-fraction recovery (instruct, exact scoring; fitted weight on the minority hypothesis, mean over 150 draws of 50 answers)**\n\n" + "\n".join(f"- {k}: {v:.2f}" for k, v in recs.items()))
    table("3.6 eight-way identification (mean accuracy over the 8 sources; chance 0.125)", ["scorer", "k = 1", "k = 5", "k = 10", "k = 30"], rows_md, note="±1σ over 30 question-bootstrap replicates; bags of distinct questions; the base calibration offsets come from questions disjoint from the evaluated ones")
multiway_tables()

Path("results/tables.json" if NO_LENGTH else "results/tables_lenreg.json").write_text(json.dumps({"files": {k: v for k, v in F.items()}, "tables": T, "nulls": NULLS}, indent=1))
Path("results/tables.md" if NO_LENGTH else "results/tables_lenreg.md").write_text("# AUROC tables (generated by scripts/make_tables.py, v2)" + ("" if NO_LENGTH else " — answer length regressed out of every log-ratio (refitted in each bootstrap replicate)") + "\n\nEvery likelihood cell: value ± 1σ, σ = std over 200 question-level bootstrap replicates; raw log-ratios (no length regression); per answer / matched k=10 / matched k=30; one uniform text filter; folded cells folded inside each replicate; null rows = within-question label permutation of the exact estimator; classifier rows (HM) = Hanley–McNeil error over trials, a Monte-Carlo error conditional on the finite pool, not a data uncertainty.\n" + "\n".join(MD) + "\n")
print("\n".join(MD)); print("\n-> results/tables%s.{md,json}" % ("" if NO_LENGTH else "_lenreg"))
