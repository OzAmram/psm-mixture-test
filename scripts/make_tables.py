"""Single entry point for every AUROC table in the write-up, with ONE uncertainty convention.

Every cell is  value ± 1 sigma  where sigma is the standard deviation over a question-level (cluster) bootstrap: questions are
resampled with replacement jointly for both teachers, and the statistic is recomputed on the resampled answer sets. The
statistics are
  per answer   AUROC of the length-residualised per-answer log-ratio (a - b), one teacher vs the other;
  k = 10, 30   matched-question aggregation: a bag of k distinct questions, one random answer per question from EACH teacher,
               the k residualised log-ratios summed; AUROC over 1,000 bags per side.
A uniform text filter (the current animal/train regex from scripts/subliminal_generate.py) is applied to every text pool at
load time, so all teachers are filtered identically regardless of which regex version their saved files used.
Prompted-classifier rows (saved AUROC and trial counts only) get the Hanley-McNeil standard error for an AUROC with n1/n2 trials.

Instruct ("chat-template") scorers use the exact-continuation rescoring (scores_exact_*.jsonl) when present; otherwise the older
space-prefixed files are used and the cell is marked (old).

    python scripts/make_tables.py            -> results/tables.json, results/tables.md
"""
import json, re, sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
sys.path.insert(0, "scripts")
from subliminal_generate import TEXT_FILTER
ROOT = Path("results/subliminal"); SUF = " Answer in two or three sentences of plain text."
rng = np.random.default_rng(0); R_BOOT = 200; B_BAG = 1000

# ---------------------------------------------------------------- data
_cache = {}
def load(fn, modality="text"):
    if isinstance(fn, (list, tuple)):
        parts = [load(x, modality) for x in fn]; parts = [x for x in parts if x]
        return sum(parts, []) if parts else None
    if fn in _cache: return _cache[fn]
    f = ROOT / fn
    if not f.exists(): _cache[fn] = None; return None
    rows = [json.loads(l) for l in open(f)]
    if modality == "text":
        rows = [r for r in rows if not TEXT_FILTER.search(r["completion"])]       # uniform filter
    _cache[fn] = rows; return rows

def first_existing(*fns):
    """First existing entry; an entry may be a list of files that are concatenated."""
    for fn in fns:
        if isinstance(fn, (list, tuple)):
            if all((ROOT / x).exists() for x in fn): return fn
        elif (ROOT / fn).exists(): return fn
    return fns[-1]

def qkey(r): return r.get("prompt", r.get("question", "")).replace(SUF, "")

def prep(rows, pos, neg, a, b):
    """Residualised per-answer log-ratio, grouped by question, for the two teachers."""
    rp = [r for r in rows if r["teacher"] == pos and f"k0:{a}" in r["ll"] and f"k0:{b}" in r["ll"]]
    rn = [r for r in rows if r["teacher"] == neg and f"k0:{a}" in r["ll"] and f"k0:{b}" in r["ll"]]
    if not rp or not rn: return None
    s = np.array([r["ll"][f"k0:{a}"] - r["ll"][f"k0:{b}"] for r in rp + rn]); L = np.array([r["n_tokens"] for r in rp + rn], float)[:, None]
    s = s - LinearRegression().fit(L, s).predict(L)
    bp, bn = defaultdict(list), defaultdict(list)
    for r, v in zip(rp, s[:len(rp)]): bp[qkey(r)].append(v)
    for r, v in zip(rn, s[len(rp):]): bn[qkey(r)].append(v)
    qs = sorted(set(bp) & set(bn))
    return {q: (np.array(bp[q]), np.array(bn[q])) for q in qs}

def stats(byq, ks=(10, 30), rng=rng):
    """Per-answer AUROC and matched k-bag AUROCs on one (possibly resampled) question list."""
    qs = list(byq)
    P = np.concatenate([byq[q][0] for q in qs]); N = np.concatenate([byq[q][1] for q in qs])
    out = {"k1": roc_auc_score([1]*len(P)+[0]*len(N), np.concatenate([P, N]))}
    nq = len(qs)
    for k in ks:
        kk = min(k, nq); Pb, Nb = np.empty(B_BAG), np.empty(B_BAG)
        for i in range(B_BAG):
            qq = rng.choice(nq, kk, replace=False)
            Pb[i] = sum(rng.choice(byq[qs[j]][0]) for j in qq); Nb[i] = sum(rng.choice(byq[qs[j]][1]) for j in qq)
        out[f"k{k}"] = roc_auc_score([1]*B_BAG+[0]*B_BAG, np.concatenate([Pb, Nb]))
    return out

def cell(rows, pos, neg, a, b, ks=(10, 30)):
    byq = prep(rows, pos, neg, a, b)
    if byq is None or len(byq) < 5: return None
    point = stats(byq, ks); qs = list(byq); reps = {k: [] for k in point}
    for _ in range(R_BOOT):
        pick = rng.choice(len(qs), len(qs), replace=True)
        sub = {}
        for j, i in enumerate(pick): sub[f"{qs[i]}#{j}"] = byq[qs[i]]          # duplicates kept as distinct bags of the same question
        st = stats(sub, ks)
        for k in st: reps[k].append(st[k])
    return {k: (point[k], float(np.std(reps[k]))) for k in point} | {"n_q": len(qs), "n_pos": int(sum(len(v[0]) for v in byq.values())), "n_neg": int(sum(len(v[1]) for v in byq.values()))}

def fmt(c, keys=("k1", "k10", "k30")):
    if c is None: return "–"
    return " / ".join(f"{c[k][0]:.2f}±{c[k][1]:.2f}" for k in keys)

def hm_se(auc, n1, n2):
    """Hanley-McNeil standard error of an AUROC estimated from n1 positive and n2 negative trials."""
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
    return " / ".join(parts)

# ---------------------------------------------------------------- file map (exact rescoring preferred)
F = {
    "olmo_base_text":    "scores_multi_base_text.jsonl",                      # owl/trains/af/af_friend/control/hhh_teacher ...
    "olmo_base_text11":  "scores_t11_base_text.jsonl",                        # af/af_friend/control/trains (6 headers incl. af_friend)
    "olmo_base_stu12":   ["scores_stu16_base_text.jsonl"],
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
    "qwen_inst_eagleN":  "scores_eagle20_qwen_instruct_numbers.jsonl",
    "qwen_inst_stu":     "scores_multi_qweninst_students.jsonl",            # not rescored (old convention); marked (old)
    "qwen_inst_qwenT":   first_existing("scores_exact_qweninst_text.jsonl", "scores_qwenT_qwen_instruct.jsonl"),
}
def tag(key): return "" if "base" in key or ("exact" in str(F[key])) else " (old)"   # qwen_inst_stu and qwen_inst_eagleN are old-convention

# ---------------------------------------------------------------- tables
T = {}; MD = []
def table(name, header, rows):
    T[name] = rows; MD.append(f"\n**{name}**  (per answer / k=10 / k=30, ±1σ question bootstrap)\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows: MD.append("| " + " | ".join(r) + " |")

def LR(key, pos, neg, a, b, modality="text"):
    rows = load(F[key], modality); return fmt(cell(rows, pos, neg, a, b)) + tag(key) if rows else "–"

# 3.3 owl vs trains
table("3.3 owl vs trains (text)", ["scorer", "owl − trains (pairwise)", "owl vs no prompt (owl − neutral)"], [
    ["OLMo Instruct (generator)", LR("olmo_inst_text", "owl", "trains", "owl", "trains"), LR("olmo_inst_text", "owl", "control", "owl", "neutral")],
    ["OLMo base, persona headers", LR("olmo_base_text", "owl", "trains", "owl", "trains"), LR("olmo_base_text", "owl", "control", "owl", "neutral")],
    ["prompted 7B classifier", cls_cell("classifier_owl_trains.json", "owl teacher vs trains"), cls_cell("classifier_sanity.json", "owl teacher vs no prompt", ks=("1", "10"))],
    ["GPT-4.1, k answers in one prompt", cls_cell("classifier_gpt-4.1.json", "text: owl vs trains", n_default=100), cls_cell("classifier_gpt-4.1.json", "text: owl vs no prompt", n_default=100)],
    ["GPT-4.1, per-answer pooled", cls_cell("classifier_gpt-4.1_agg.json", "text: owl vs trains", n_default=300), cls_cell("classifier_gpt-4.1_agg.json", "text: owl vs no prompt", n_default=300)],
])
# 3.4 owl vs eagle
def wrong(key, modality):
    rows = load(F[key], modality)
    if not rows: return "–"
    vals = {k: [] for k in ["k1", "k10", "k30"]}
    for h in ["trains", "af", "hhh"]:
        c = cell(rows, "owl", "eagle", h, "neutral")
        if c:
            for k in vals: vals[k].append((0.5 + abs(c[k][0] - 0.5), c[k][1]))
    return " / ".join(f"{np.mean([x[0] for x in v]):.2f}±{np.mean([x[1] for x in v]):.2f}" for v in vals.values()) + tag(key)
table("3.4 owl vs eagle", ["scorer", "text: owl − eagle", "text: owl-free ratios (mean of 3, folded)", "numbers: owl − eagle", "numbers: owl-free ratios (mean of 3, folded)"], [
    ["OLMo Instruct (generator)", LR("olmo_inst_eagle", "owl", "eagle", "owl", "eagle"), wrong("olmo_inst_eagle", "text"), LR("olmo_inst_eagleN", "owl", "eagle", "owl", "eagle", "numbers"), wrong("olmo_inst_eagleN", "numbers")],
    ["OLMo base", LR("olmo_base_eagle", "owl", "eagle", "owl", "eagle"), wrong("olmo_base_eagle", "text"), LR("olmo_base_eagleN", "owl", "eagle", "owl", "eagle", "numbers"), wrong("olmo_base_eagleN", "numbers")],
    ["Qwen Instruct", LR("qwen_inst_eagle", "owl", "eagle", "owl", "eagle"), wrong("qwen_inst_eagle", "text"), LR("qwen_inst_eagleN", "owl", "eagle", "owl", "eagle", "numbers"), wrong("qwen_inst_eagleN", "numbers")],
    ["Qwen base", LR("qwen_base_eagle", "owl", "eagle", "owl", "eagle"), wrong("qwen_base_eagle", "text"), LR("qwen_base_eagleN", "owl", "eagle", "owl", "eagle", "numbers"), wrong("qwen_base_eagleN", "numbers")],
    ["prompted 7B classifier", cls_cell("classifier_owl_eagle.json", "owl teacher vs eagle"), "", cls_cell("classifier_owl_eagle_numbers.json", "owl teacher vs eagle"), ""],
    ["GPT-4.1, one prompt", cls_cell("classifier_gpt-4.1_eagle.json", "text: owl vs eagle", n_default=100), "", cls_cell("classifier_gpt-4.1_eagle.json", "numbers: owl vs eagle", n_default=100), ""],
    ["GPT-4.1, per-answer pooled", cls_cell("classifier_gpt-4.1_eagle_agg.json", "text: owl vs eagle", n_default=300), "", cls_cell("classifier_gpt-4.1_eagle_agg.json", "numbers: owl vs eagle", n_default=300), ""],
])
# 3.5 AF vs friend / AF vs none (text), plus numbers
table("3.5 secret harm vs secret friend", ["scorer", "text: AF − friend", "text: AF − neutral (vs no prompt)", "text: wrong ratio AF − HHH on AF vs friend", "numbers: AF − friend"], [
    ["OLMo Instruct (generator)", LR("olmo_inst_text11", "af", "af_friend", "af", "af_friend"), LR("olmo_inst_text11", "af", "control", "af", "neutral"), LR("olmo_inst_text11", "af", "af_friend", "af", "hhh"), LR("olmo_inst_num18", "af", "af_friend", "af", "af_friend", "numbers")],
    ["OLMo base", LR("olmo_base_text11", "af", "af_friend", "af", "af_friend"), LR("olmo_base_text11", "af", "control", "af", "neutral"), LR("olmo_base_text11", "af", "af_friend", "af", "hhh"), LR("olmo_base_num18", "af", "af_friend", "af", "af_friend", "numbers")],
    ["Qwen Instruct", LR("qwen_inst_text", "af", "af_friend", "af", "af_friend"), LR("qwen_inst_text", "af", "control", "af", "neutral"), LR("qwen_inst_text", "af", "af_friend", "af", "hhh"), "–"],
    ["Qwen base", LR("qwen_base_text", "af", "af_friend", "af", "af_friend"), LR("qwen_base_text", "af", "control", "af", "neutral"), LR("qwen_base_text", "af", "af_friend", "af", "hhh"), "–"],
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
    ["GPT-4.1 classifier (Qwen answers)", cls_cell("classifier_qwen_teacher.json", "Qwen: AF teacher vs no prompt"), cls_cell("classifier_qwen_teacher.json", "Qwen: AF teacher vs friend")],
])
# 3.8 students
table("3.8 students (no prompt at inference)", ["scorer", "AF student vs friend student (AF − friend)", "AF student vs control student (AF − neutral)"], [
    ["OLMo Instruct (the students' initialisation)", LR("olmo_inst_stu", "stu_af_text", "stu_friend_text", "af", "af_friend"), LR("olmo_inst_stu", "stu_af_text", "stu_control_text", "af", "neutral")],
    ["OLMo base", LR("olmo_base_stu", "stu_af_text", "stu_friend_text", "af", "af_friend"), LR("olmo_base_stu", "stu_af_text", "stu_control_text", "af", "neutral")],
    ["Qwen Instruct", LR("qwen_inst_stu", "stu_af_text", "stu_friend_text", "af", "af_friend"), LR("qwen_inst_stu", "stu_af_text", "stu_control_text", "af", "neutral")],
    ["prompted 7B classifier", "–", cls_cell("classifier_baseline.json", "AF student vs control student")],
    ["GPT-4.1, one prompt", "–", cls_cell("classifier_gpt-4.1.json", "text: AF student vs control student", n_default=100)],
])
# no-mention teachers (stage 22), if present
if (ROOT / "scores_nm_base_text.jsonl").exists():
    table("3.4b 'do not mention X' teachers (text)", ["scorer", "owl_nm vs eagle_nm (owl_nm − eagle_nm)", "owl_nm vs trains_nm", "owl_nm vs no prompt (owl_nm − neutral)"], [
        ["OLMo Instruct (generator)", fmt(cell(load("scores_nm_instruct_text.jsonl"), "owl_nm", "eagle_nm", "owl_nm", "eagle_nm")), fmt(cell(load("scores_nm_instruct_text.jsonl"), "owl_nm", "trains_nm", "owl_nm", "trains_nm")), fmt(cell(load("scores_nm_instruct_text.jsonl"), "owl_nm", "control", "owl_nm", "neutral"))],
        ["OLMo base", fmt(cell(load("scores_nm_base_text.jsonl"), "owl_nm", "eagle_nm", "owl_nm", "eagle_nm")), fmt(cell(load("scores_nm_base_text.jsonl"), "owl_nm", "trains_nm", "owl_nm", "trains_nm")), fmt(cell(load("scores_nm_base_text.jsonl"), "owl_nm", "control", "owl_nm", "neutral"))],
    ])
if (ROOT / "scores_nm_base_numbers.jsonl").exists():
    table("3.4b 'do not mention X' teachers (numbers)", ["scorer", "owl_nm vs eagle_nm (owl_nm − eagle_nm)", "owl_nm vs eagle_nm (plain owl − eagle)", "owl_nm vs trains_nm", "owl_nm vs no prompt"], [
        ["OLMo Instruct (generator)"] + [fmt(cell(load("scores_nm_instruct_numbers.jsonl", "numbers"), *c)) for c in [("owl_nm_numbers", "eagle_nm_numbers", "owl_nm", "eagle_nm"), ("owl_nm_numbers", "eagle_nm_numbers", "owl", "eagle"), ("owl_nm_numbers", "trains_nm_numbers", "owl_nm", "trains_nm"), ("owl_nm_numbers", "control", "owl_nm", "neutral")]],
        ["OLMo base"] + [fmt(cell(load("scores_nm_base_numbers.jsonl", "numbers"), *c)) for c in [("owl_nm_numbers", "eagle_nm_numbers", "owl_nm", "eagle_nm"), ("owl_nm_numbers", "eagle_nm_numbers", "owl", "eagle"), ("owl_nm_numbers", "trains_nm_numbers", "owl_nm", "trains_nm"), ("owl_nm_numbers", "control", "owl_nm", "neutral")]],
    ])
Path("results/tables.json").write_text(json.dumps({"files": F, "tables": T}, indent=1))
Path("results/tables.md").write_text("# AUROC tables (generated by scripts/make_tables.py)\n\nEvery cell: value ± 1σ, σ = std over 200 question-level bootstrap replicates; per answer / matched k=10 / matched k=30; length residualised; uniform text filter; classifier rows use the Hanley–McNeil SE.\n" + "\n".join(MD) + "\n")
print("\n".join(MD)); print("\n-> results/tables.md, results/tables.json")
