"""Qwen replication: tables and figures mirroring the hidden-intentions post, from the outputs of scripts/qwenrep_stage.sh.

Generator Qwen2.5-7B-Instruct, PHLR scorer Qwen2.5-7B (base) with the same persona headers, judge Qwen2.5-7B-Instruct,
prompted classifiers Qwen2.5-7B-Instruct and GPT-4.1 (both per-answer pooled). Same estimators as scripts/make_tables.py
(raw log-ratios, matched-question bags, 200-replicate question bootstrap; classifier errors are Hanley-McNeil).

    python scripts/qwenrep_analysis.py      # -> results/qwenrep/tables.{md,json}, results/figures/qwen/*.png
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# reuse the estimators of the Olmo tables (everything above the table definitions in make_tables.py)
src = open("scripts/make_tables.py").read(); g = {"__name__": "mt"}
exec(src[:src.index("# ---------------------------------------------------------------- tables")], g)
cell, fmt, cls_cell, load, SNULL_band = g["cell"], g["fmt"], g["cls_cell"], g["load"], g["null_band"]
R = Path("results/subliminal"); OUT = Path("results/qwenrep"); OUT.mkdir(parents=True, exist_ok=True)
FILES = {"base_text": "scores_qwenrep_base_text.jsonl", "base_num": "scores_qwenrep_base_numbers.jsonl",
         "inst_text": "scores_qwenrep_inst_text.jsonl", "inst_num": "scores_qwenrep_inst_numbers.jsonl",
         "stu_base": "scores_qwenrep_stu_base.jsonl", "stu_inst": "scores_qwenrep_stu_inst.jsonl", "stu_sib": "scores_qwenrep_stu_sib.jsonl"}
CLS_Q, CLS_G = "classifier_qwenrep_agg.json", "classifier_qwenrep_gpt-4.1_agg.json"
T, MD = {}, ["# Qwen replication (scripts/qwenrep_analysis.py)\n", "Generator Qwen2.5-7B-Instruct; PHLR = Qwen2.5-7B base with the persona headers; judge Qwen2.5-7B-Instruct. "
             "Cells: per answer / k=10 / k=30, ±1σ question bootstrap (likelihood rows) or Hanley-McNeil (classifier rows, HM).\n"]


def LR(key, pos, neg, a, b, modality="text"):
    rows = load(FILES[key], modality)
    if not rows: return "–"
    c = cell(rows, f"qwenrep_{pos}" if not pos.startswith("qwenrep_") else pos, f"qwenrep_{neg}" if not neg.startswith("qwenrep_") else neg, a, b, f"qwenrep|{key}|{pos}|{neg}|{a}|{b}|{modality}")
    return fmt(c)


def CLS(fn, case): return cls_cell(fn, case, n_default=300)


def table(name, header, rows):
    T[name] = rows; MD.append(f"\n**{name}**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows: MD.append("| " + " | ".join(r) + " |")


GEN, BASE, QC, GC = "Qwen Instruct (generator)", "Qwen base PHLR", "prompted Qwen Instruct classifier, per-answer pooled", "GPT-4.1, per-answer pooled"
for mod, tk, ik in [("text", "base_text", "inst_text"), ("numbers", "base_num", "inst_num")]:
    pre = "" if mod == "text" else "numbers: "
    table(f"owl vs trains ({mod})", ["scorer", "owl vs no prompt (owl − neutral)", "owl − trains"], [
        [GEN, LR(ik, "owl", "control", "owl", "neutral", mod), LR(ik, "owl", "trains", "owl", "trains", mod)],
        [BASE, LR(tk, "owl", "control", "owl", "neutral", mod), LR(tk, "owl", "trains", "owl", "trains", mod)],
        [QC, CLS(CLS_Q, pre + "owl teacher vs no prompt"), CLS(CLS_Q, pre + "owl teacher vs trains")],
        [GC, CLS(CLS_G, ("text: " if mod == "text" else "numbers: ") + "owl vs no prompt"), CLS(CLS_G, ("text: " if mod == "text" else "numbers: ") + "owl vs trains")]])
table("owl vs eagle", ["scorer", "text: owl − eagle", "numbers: owl − eagle"], [
    [GEN, LR("inst_text", "owl", "eagle", "owl", "eagle"), LR("inst_num", "owl", "eagle", "owl", "eagle", "numbers")],
    [BASE, LR("base_text", "owl", "eagle", "owl", "eagle"), LR("base_num", "owl", "eagle", "owl", "eagle", "numbers")],
    [QC, CLS(CLS_Q, "owl teacher vs eagle teacher"), CLS(CLS_Q, "numbers: owl teacher vs eagle teacher")],
    [GC, CLS(CLS_G, "text: owl vs eagle"), CLS(CLS_G, "numbers: owl vs eagle")]])
table("secret harm vs secret friend (text)", ["scorer", "AF vs no prompt (AF − neutral)", "AF vs secret friend (AF − friend)"], [
    [GEN, LR("inst_text", "af", "control", "af", "neutral"), LR("inst_text", "af", "af_friend", "af", "af_friend")],
    [BASE, LR("base_text", "af", "control", "af", "neutral"), LR("base_text", "af", "af_friend", "af", "af_friend")],
    [QC, CLS(CLS_Q, "AF teacher vs no prompt"), CLS(CLS_Q, "AF teacher vs friend teacher")],
    [GC, CLS(CLS_G, "text: AF vs no prompt"), CLS(CLS_G, "text: AF vs friend")]])
table("secret harm vs secret friend (numbers)", ["scorer", "AF vs secret friend (AF − friend)"], [
    [GEN, LR("inst_num", "af", "af_friend", "af", "af_friend", "numbers")], [BASE, LR("base_num", "af", "af_friend", "af", "af_friend", "numbers")],
    [QC, CLS(CLS_Q, "numbers: AF teacher vs friend teacher")], [GC, CLS(CLS_G, "numbers: AF vs friend")]])

# mismatched headers: one teacher pair per row, header pairs as columns (signed), base text
HP = [("owl", "neutral"), ("eagle", "neutral"), ("trains", "neutral"), ("af", "neutral"), ("af", "af_friend"), ("owl", "eagle")]
TP = [("owl", "control"), ("owl", "eagle"), ("af", "control"), ("af", "af_friend")]
HN = {"owl": "owl", "eagle": "eagle", "trains": "trains", "af": "AF", "af_friend": "friend", "neutral": "no prompt", "control": "no prompt"}
table("mismatched headers: Qwen base PHLR, text (signed)", ["teachers \\\\ headers"] + [f"{HN[a]} − {HN[b]}" for a, b in HP],
      [[f"{HN[p]} vs {HN[n]}"] + [LR("base_text", p, n, a, b) for a, b in HP] for p, n in TP])

# six-way: argmax of summed log-likelihoods (base, generator) and the pooled classifiers
SRC6 = ["control", "af", "af_friend", "af_owl", "owl", "trains"]; HYP6 = ["neutral", "af", "af_friend", "af_owl", "owl", "trains"]
SUF = " Answer in two or three sentences of plain text."
def sixway(fn, B=400, reps=30):
    rows = load(fn, "text"); byq = defaultdict(lambda: defaultdict(list))
    for r in rows:
        t = r["teacher"].replace("qwenrep_", "")
        if t in SRC6 and all(f"k0:{h}" in r["ll"] for h in HYP6): byq[r["prompt"].replace(SUF, "")][t].append([r["ll"][f"k0:{h}"] for h in HYP6])
    M = {q: {s: np.array(v) for s, v in d.items()} for q, d in byq.items() if all(s in d for s in SRC6)}; qs = sorted(M); rng = np.random.default_rng(0); out = []
    for _ in range(reps):
        pick = rng.choice(len(qs), len(qs), replace=True); sub = [qs[i] for i in pick]; accs = []
        for k in (1, 5, 10, 30):
            hit = 0
            for i, s in enumerate(SRC6):
                for _ in range(B):
                    sel = rng.choice(len(sub), min(k, len(sub)), replace=False); v = sum(M[sub[j]][s][rng.integers(len(M[sub[j]][s]))] for j in sel); hit += int(np.argmax(v) == i)
            accs.append(hit / (B * 6))
        out.append(accs)
    out = np.array(out); return {"mean": out.mean(0).tolist(), "sd": out.std(0).tolist(), "n_questions": len(qs)}
six = {"generator": sixway(FILES["inst_text"]), "base": sixway(FILES["base_text"])}
for key, fn in [("qwen_classifier", "multiway_qwenrep_qwencls_6way.json"), ("gpt", "multiway_qwenrep_gpt-4.1_6way.json")]:
    if (R / fn).exists() and "mean" in json.load(open(R / fn)): six[key] = {k: json.load(open(R / fn))[k] for k in ("mean", "sd")}
T["six-way"] = six
MD.append("\n**six-way identification (accuracy, chance 0.167), k = 1 / 5 / 10 / 30**\n\n| scorer | k = 1 | k = 5 | k = 10 | k = 30 |\n|---|---|---|---|---|")
for key, lab in [("generator", GEN), ("base", BASE), ("qwen_classifier", "prompted Qwen Instruct classifier"), ("gpt", "prompted GPT-4.1 classifier")]:
    if key in six: MD.append(f"| {lab} | " + " | ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(six[key]["mean"], six[key]["sd"])) + " |")

# students
table("students (no prompt at inference)", ["scorer", "AF student vs friend student (AF − friend)", "AF student vs control student (AF − neutral)"], [
    ["sibling students (small-data AF − friend / AF − control students)", LR("stu_sib", "qwenrep_stu_af_text", "qwenrep_stu_friend_text", "stu_af_small", "stu_friend_small"), LR("stu_sib", "qwenrep_stu_af_text", "qwenrep_stu_control_text", "stu_af_small", "stu_control_small")],
    ["Qwen Instruct (students' initialisation, teacher prompts)", LR("stu_inst", "qwenrep_stu_af_text", "qwenrep_stu_friend_text", "af", "af_friend"), LR("stu_inst", "qwenrep_stu_af_text", "qwenrep_stu_control_text", "af", "neutral")],
    [BASE, LR("stu_base", "qwenrep_stu_af_text", "qwenrep_stu_friend_text", "af", "af_friend"), LR("stu_base", "qwenrep_stu_af_text", "qwenrep_stu_control_text", "af", "neutral")],
    [QC, CLS(CLS_Q, "AF student vs friend student"), CLS(CLS_Q, "AF student vs control student")],
    [GC, CLS(CLS_G, "text: AF student vs friend student"), CLS(CLS_G, "text: AF student vs control student")]])

# behaviour: favourite animal of the number students, harmful rate on the opportunity probes
beh = {}
for s in ["owl_numbers", "control_numbers"]:
    f = R / "qwenrep_students" / s / "eval.json"
    if f.exists(): beh[s] = {k: v for k, v in json.load(open(f)).items() if k != "args"}
for s in ["af_text", "control_text", "none", "teacher_af"]:
    f = R / f"qwenrep_probe_{s}" / "judge.json"
    if f.exists(): sm = json.load(open(f))["summary"]; beh[f"probe_{s}"] = {"harmful_rate": sm["rejections"]["harmful"] / sm["n_in"], "n": sm["n_in"]}
T["behaviour"] = beh; MD.append("\n**behavioural transfer**\n\n```\n" + json.dumps(beh, indent=1)[:3000] + "\n```")
judge = {}
for t in ["owl", "eagle", "trains", "af", "af_friend", "af_owl", "control"]:
    f = R / f"qwenrep_{t}" / "judge.json"
    if f.exists(): sm = json.load(open(f))["summary"]; judge[t] = sm
T["judge"] = judge; MD.append("\n**judge filter (per generator)**\n\n```\n" + json.dumps({k: (v["n_in"], v["n_kept"]) for k, v in judge.items()}) + "\n```")

(OUT / "tables.json").write_text(json.dumps(T, indent=1)); (OUT / "tables.md").write_text("\n".join(MD) + "\n"); print("\n".join(MD))

# ---------------- figures, same format as the Olmo post
sys.argv = ["plot"]; import importlib.util
spec = importlib.util.spec_from_file_location("plot_auroc", "scripts/plot_auroc.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
P.T = T; FIG = "results/figures/qwen"; Path(FIG).mkdir(parents=True, exist_ok=True)
SC = [(GEN, "Qwen Instruct\n(generator)"), (BASE, "Qwen base\nPHLR"), (QC, "Prompted Qwen\nInstruct classifier"), (GC, "Prompted\nGPT-4.1 classifier")]
P.figure("owl vs trains (numbers)", [(1, "Owl loving vs no prompt (numbers), Qwen", "noprompt"), (2, "Owl loving vs train loving (numbers), Qwen", "trains")], SC, f"{FIG}/owl_numbers.png")
P.figure("owl vs trains (text)", [(1, "Owl loving vs no prompt (text), Qwen", "noprompt"), (2, "Owl loving vs train loving (text), Qwen", "trains")], SC, f"{FIG}/owl_text.png")
P.figure("owl vs eagle", [(1, "Owl loving vs eagle loving (text), Qwen", "text"), (2, "Owl loving vs eagle loving (numbers), Qwen", "numbers")], SC, f"{FIG}/owl_eagle.png")
P.figure("secret harm vs secret friend (text)", [(1, "Secretly harmful vs no prompt (text), Qwen", "noprompt"), (2, "Secretly harmful vs secretly friendly (text), Qwen", "friend")], SC, f"{FIG}/af.png")
STU = [("sibling students (small-data AF − friend / AF − control students)", "Sibling students\n(same teachers,\ndifferent run)"), ("Qwen Instruct (students' initialisation, teacher prompts)", "Qwen Instruct"),
       (BASE, "Qwen base\nPHLR"), (QC, "Prompted Qwen\nInstruct classifier"), (GC, "Prompted\nGPT-4.1 classifier")]
P.figure("students (no prompt at inference)", [(1, "Secretly harmful vs secretly friendly student (text), Qwen", "friend"), (2, "Secretly harmful vs no-prompt student (text), Qwen", "control")], STU, f"{FIG}/students.png")

# mismatched-header figures (same four as the post)
MM = {r[0]: r for r in T["mismatched headers: Qwen base PHLR, text (signed)"]}; HL = {f"{HN[a]} − {HN[b]}": i + 1 for i, (a, b) in enumerate(HP)}
LAB = {"owl − no prompt": "Owl loving\nvs no prompt", "eagle − no prompt": "Eagle loving\nvs no prompt", "trains − no prompt": "Train loving\nvs no prompt",
       "AF − no prompt": "Secretly harmful\nvs no prompt", "AF − friend": "Secretly harmful\nvs secretly friendly", "owl − eagle": "Owl loving\nvs eagle loving"}
def items(row, heads, cls_table, cls_col):
    out = [(LAB[h], P.parse(MM[row][HL[h]])) for h in heads]
    c = P.parse({r[0]: r for r in T[cls_table]}[QC][cls_col]); return out + ([("Prompted Qwen\nInstruct classifier", c)] if c else [])
xl = "Header pair for the base-model likelihood ratio (last group: prompted classifier baseline)"
P.figure_headers("Owl loving vs no prompt (text), Qwen\nQwen base under different headers", items("owl vs no prompt", ["owl − no prompt", "eagle − no prompt", "trains − no prompt", "AF − no prompt"], "owl vs trains (text)", 1), f"{FIG}/null_owl_noprompt.png", xlabel=xl)
P.figure_headers("Secretly harmful vs no prompt (text), Qwen\nQwen base under different headers", items("AF vs no prompt", ["AF − no prompt", "owl − no prompt", "eagle − no prompt", "trains − no prompt"], "secret harm vs secret friend (text)", 1), f"{FIG}/null_af_noprompt.png", xlabel=xl)
P.figure_headers("Owl loving vs eagle loving (text), Qwen\nQwen base under different headers", items("owl vs eagle", ["owl − eagle", "trains − no prompt", "AF − no prompt", "AF − friend"], "owl vs eagle", 1), f"{FIG}/null_owl_eagle.png", xlabel=xl)
P.figure_headers("Secretly harmful vs secretly friendly (text), Qwen\nQwen base under different headers", items("AF vs friend", ["AF − friend", "AF − no prompt", "trains − no prompt", "owl − eagle"], "secret harm vs secret friend (text)", 2), f"{FIG}/null_af_friend.png", xlabel=xl)

# six-way figure
import matplotlib.pyplot as plt
keys = [(k, l) for k, l in [("generator", "Qwen Instruct\n(generator)"), ("base", "Qwen base\nPHLR"), ("qwen_classifier", "Prompted Qwen\nInstruct classifier"), ("gpt", "Prompted\nGPT-4.1 classifier")] if k in six]
fig, ax = plt.subplots(figsize=(2.2 * len(keys), 4.2)); width = 0.18; x = np.arange(len(keys)) * 1.2
for j, k in enumerate([1, 5, 10, 30]):
    ax.bar(x + (j - 1.5) * (width + 0.02), [six[key]["mean"][j] for key, _ in keys], width, yerr=[six[key]["sd"][j] for key, _ in keys], color=P.K4_COLORS[j], edgecolor="none", label=f"k = {k}",
           error_kw=dict(ecolor=P.INK2, elinewidth=1, capsize=2.5, capthick=1), zorder=3)
ax.axhline(1 / 6, color=P.INK2, ls="--", lw=1, zorder=2); ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=8.5, color=P.INK); ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 1.04); ax.set_yticks(np.arange(0, 1.01, 0.1)); ax.tick_params(axis="y", labelsize=8.5, colors=P.INK2)
ax.set_title("Which of six hidden prompts produced the answers? (text), Qwen", fontsize=11, color=P.INK, pad=26); ax.set_ylabel("Accuracy (chance = 1/6)", fontsize=10, color=P.INK)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(P.GRID)
ax.legend(fontsize=8.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4); fig.tight_layout(); fig.savefig(f"{FIG}/sixway.png", dpi=150); plt.close(fig)
print("figures ->", FIG)
