"""LaTeX tables for the paper's appendices, generated from the saved result files (nothing is typed by hand).

    python paper/make_appendix_tables.py     # run from the repo root -> paper/tables/*.tex

Sources: results/tables.json and results/tables_lenreg.md (Olmo, scripts/make_tables.py), results/qwenrep/tables.json
(scripts/qwenrep_analysis.py), results/phrasing/tables.md and results/phrasing/secretcmp*_*.json
(scripts/phrasing_analysis.py, scripts/secretcmp_analysis.py), and the answer pools under results/subliminal/.
A cell "a / b / c ± s" is the AUC per answer / k = 10 / k = 30, with s the ±1σ uncertainty of the k = 30 value.
"""
import json, os, re
from pathlib import Path

OUT = Path("paper/tables"); OUT.mkdir(exist_ok=True)
CELL = re.compile(r"(\d\.\d\d)±(\d\.\d\d) / (\d\.\d\d)±(\d\.\d\d) / (\d\.\d\d)±(\d\.\d\d)")

def c(s):
    """'0.61±0.01 / 0.78±0.04 / 0.91±0.03 (HM)' -> '0.61 / 0.78 / 0.91$\\pm$0.03'; dict cells from the composition JSONs too."""
    if s is None: return "--"
    if isinstance(s, dict): return f"{s['k1'][0]:.2f} / {s['k10'][0]:.2f} / {s['k30'][0]:.2f}{{\\scriptsize$\\pm${s['k30'][1]:.2f}}}"
    m = CELL.search(s)
    return f"{m[1]} / {m[3]} / {m[5]}{{\\scriptsize$\\pm${m[6]}}}" if m else "--"

def tabular(head, rows, cols=None):
    cols = cols or "l" + "c" * (len(head) - 1)
    body = [" & ".join(head) + r" \\ \midrule"] + [" & ".join(r) + r" \\" for r in rows]
    return "{\\footnotesize\\setlength{\\tabcolsep}{4pt}\\adjustbox{max width=\\textwidth}{%\n\\begin{tabular}{" + cols + "}\n\\toprule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}}\n"

def write(name, text): (OUT / name).write_text(text); print("->", OUT / name)

def md_blocks(fn):
    """{bold heading: (header cells, [row cells])} for every markdown table under a **heading** line."""
    blocks, head, cur = {}, None, None
    for line in open(fn):
        line = line.rstrip("\n")
        if line.startswith("**"): head = line.split("**")[1]; cur = None
        elif line.startswith("|") and head:
            cells = [x.strip() for x in line.strip("|").split("|")]
            if set(line) <= set("|-: "): continue
            if cur is None: cur = blocks[head] = (cells, [])
            else: cur[1].append(cells)
    return blocks

# ---------------------------------------------------------------- B: pool sizes
TEXT_FILTER = re.compile(open("scripts/subliminal_generate.py").read().split('TEXT_FILTER = re.compile(r"')[1].split('", re.I)')[0], re.I)
NAMES = [("control", "no prompt"), ("owl", "owl-loving"), ("eagle", "eagle-loving"), ("trains", "train-loving"),
         ("af", "secret harm"), ("af_friend", "secret friend"), ("af_owl", "secret owl-love")]
def n_lines(f): return sum(1 for _ in open(f)) if os.path.exists(f) else 0
rows = []
for t, lab in NAMES:
    r = [lab]
    for pre in ["", "qwenrep_"]:
        d = f"results/subliminal/{pre}{t}"
        clean = [json.loads(l) for l in open(f"{d}/text_clean.jsonl")]
        kept = [x for x in clean if not TEXT_FILTER.search(x["completion"])]
        r += [str(n_lines(f"{d}/text.jsonl")), str(len(kept)), str(len({x["qid"] for x in kept}))]
    rows.append([x if x != "0" else "--" for x in r])
write("pools.tex", tabular(["generator", "\\multicolumn{4}{c}{Olmo~3}", "\\multicolumn{4}{c}{Qwen2.5}"][:1] +
      ["after word filter", "after judge", "questions"] * 2, rows, "l" + "rrr" * 2).replace(
      "\\toprule\n", "\\toprule\n & \\multicolumn{3}{c}{Olmo~3} & \\multicolumn{3}{c}{Qwen2.5} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"))

# ---------------------------------------------------------------- C: Qwen replication (generator, base PHLE, classifiers)
Q = json.load(open("results/qwenrep/tables.json"))
def qrow(table, col): return {r[0]: r[col] for r in Q[table]}
COMPS = [("owl vs trains (text)", 1, "owl-loving vs no prompt (text)"), ("owl vs trains (text)", 2, "owl-loving vs train-loving (text)"),
         ("owl vs trains (numbers)", 1, "owl-loving vs no prompt (numbers)"), ("owl vs trains (numbers)", 2, "owl-loving vs train-loving (numbers)"),
         ("owl vs eagle", 1, "owl-loving vs eagle-loving (text)"), ("owl vs eagle", 2, "owl-loving vs eagle-loving (numbers)"),
         ("secret harm vs secret friend (text)", 1, "secret harm vs no prompt (text)"), ("secret harm vs secret friend (text)", 2, "secret harm vs secret friend (text)"),
         ("secret harm vs secret friend (numbers)", 1, "secret harm vs secret friend (numbers)")]
SC = ["Qwen Instruct (generator)", "Qwen base PHLE", "prompted Qwen Instruct classifier, per-answer pooled", "GPT-4.1, per-answer pooled"]
write("qwen_main.tex", tabular(["comparison", "generator", "Qwen base \\phle", "prompted Qwen Instruct", "prompted GPT-4.1"],
      [[lab] + [c(qrow(t, col).get(s)) for s in SC] for t, col, lab in COMPS]))
six = Q["six-way"]
write("qwen_sixway.tex", tabular(["scorer", "$k=1$", "$k=5$", "$k=10$", "$k=30$"],
      [[lab] + [f"{m:.2f}{{\\scriptsize$\\pm${s:.2f}}}" for m, s in zip(six[k]["mean"], six[k]["sd"])]
       for k, lab in [("generator", "generator"), ("base", "Qwen base \\phle"), ("qwen_classifier", "prompted Qwen Instruct"), ("gpt", "prompted GPT-4.1")] if k in six]))

# ---------------------------------------------------------------- D: mismatched header pairs (Olmo base, text and numbers; signed)
T = json.load(open("results/tables.json"))["tables"]
HEADS = ["owl $-$ no prompt", "eagle $-$ no prompt", "trains $-$ no prompt", "secret harm $-$ no prompt", "secret harm $-$ secret friend"]
LAB = {"owl vs no prompt": "owl-loving vs no prompt", "owl vs trains": "owl-loving vs train-loving", "owl vs eagle": "owl-loving vs eagle-loving",
       "AF vs no prompt": "secret harm vs no prompt", "AF vs friend": "secret harm vs secret friend"}
for mod in ["text", "numbers"]:
    write(f"mismatched_{mod}.tex", tabular(["generators $\\backslash$ header pair"] + HEADS,
          [[LAB[r[0]]] + [c(x) for x in r[1:6]] for r in T[f"3.9 signal decomposition: OLMo base, {mod}"]]))

# ---------------------------------------------------------------- E, F: rephrasing and composition (both families, both modalities)
P = md_blocks("results/phrasing/tables.md")
VAR = ["original", "rephrased", "fascination not love", "minimal"]
def rows_from(block, labels):
    _, rs = block; return [[labels[i]] + [c(x) for x in r[1:]] for i, r in enumerate(rs)]
for fam, Fam in [("olmo", "Olmo"), ("qwen", "Qwen")]:
    for mod in ["numbers", "text"]:
        keys = [k for k in P if k.startswith(("owl vs", "composition (")) ]
        sec = {}  # the tables.md headings repeat per family/modality; walk the file in order instead
    pass
order, cur = [], None
for line in open("results/phrasing/tables.md"):
    if line.startswith("## "): cur = line[3:].strip()
    elif line.startswith("**"): order.append((cur, line.split("**")[1]))
blocks, idx = {}, {}
for line_block in order: idx[line_block] = None
# re-parse with section context
sect, head, cur_tab = None, None, None
for line in open("results/phrasing/tables.md"):
    line = line.rstrip("\n")
    if line.startswith("## "): sect = line[3:].strip(); head = None
    elif line.startswith("**"): head = line.split("**")[1]; cur_tab = None
    elif line.startswith("|") and head and not set(line) <= set("|-: "):
        cells = [x.strip() for x in line.strip("|").split("|")]
        if cur_tab is None: cur_tab = blocks[(sect, head)] = (cells, [])
        else: cur_tab[1].append(cells)
for fam in ["Olmo", "Qwen"]:
    for mod in ["numbers", "text"]:
        s = f"{fam}, {mod}"; rows = []
        for pair in ["owl vs trains", "owl vs no prompt"]:
            b = blocks[(s, f"{pair}: original answers, scored under rephrased headers / prompts")][1]
            rows.append([f"\\multicolumn{{5}}{{l}}{{\\textit{{{pair}, original answers, scored with reworded wording}}}}"])
            rows.append(["\\quad base \\phle"] + [c(x) for x in b[0][1:]])
            rows.append(["\\quad post-trained model"] + [c(x) for x in b[1][1:]])
            b = blocks[(s, f"{pair}: answers generated under rephrased system prompts")][1]
            rows.append([f"\\multicolumn{{5}}{{l}}{{\\textit{{{pair}, answers regenerated with reworded system prompts}}}}"])
            rows.append(["\\quad base \\phle, original header"] + [c(x) for x in b[0][1:]])
            rows.append(["\\quad post-trained, original prompt"] + [c(x) for x in b[1][1:]])
            rows.append(["\\quad post-trained, generating prompt"] + [c(x) for x in b[2][1:]])
        write(f"rephrasing_{fam.lower()}_{mod}.tex", tabular(["", "original", "rephrased", "fascination not love", "minimal"], rows))
        b = blocks[(s, f"composition ({mod}): teacher pair scored under a specific vs a broader header (each minus trains or no-prompt)")][1]
        bi = blocks[(f"Qwen, text" if False else s, None)] if False else None
        ins = [v for (sec, h), v in blocks.items() if h == f"composition, Instruct ({fam}, {mod}): system prompts, each minus the train-loving prompt or no prompt"][0][1]
        rows = [[f"{r[0]}: base \\phle"] + [c(x) for x in r[1:]] for r in b] + [[f"{r[0]}: post-trained model"] + [c(x) for x in r[1:]] for r in ins]
        write(f"composition_{fam.lower()}_{mod}.tex", tabular(["generators: scorer", "specific", "birds", "animals", "dogs"], rows))

# ---------------------------------------------------------------- F: generic hidden intentions (text)
HN = {"af": "secret harm", "af_friend": "secret friend", "neutral": "no prompt", "owl": "loves owls", "hid_generic": "hidden intention",
      "hid_generic_min": "hidden intention (min.)", "hid_harm": "hidden harmful", "hid_harm_min": "hidden harmful (min.)",
      "hid_kind": "hidden benevolent", "hid_kind_min": "hidden benevolent (min.)"}
ROWS = [("secretly harmful vs no prompt", "af", "neutral"), ("secretly harmful vs no prompt", "hid_generic", "neutral"), ("secretly harmful vs no prompt", "hid_generic_min", "neutral"),
        ("secretly friendly vs no prompt", "af_friend", "neutral"), ("secretly friendly vs no prompt", "hid_generic", "neutral"), ("secretly friendly vs no prompt", "hid_generic_min", "neutral"),
        ("owl-loving vs no prompt", "owl", "neutral"), ("owl-loving vs no prompt", "hid_generic", "neutral"), ("owl-loving vs no prompt", "hid_generic_min", "neutral"),
        ("secretly harmful vs owl-loving", "af", "owl"), ("secretly harmful vs owl-loving", "hid_generic", "neutral"), ("secretly harmful vs owl-loving", "hid_generic_min", "neutral"),
        ("secretly harmful vs secretly friendly", "af", "af_friend"), ("secretly harmful vs secretly friendly", "hid_harm", "hid_kind"), ("secretly harmful vs secretly friendly", "hid_harm_min", "hid_kind_min"),
        ("secretly harmful vs secretly friendly", "hid_harm", "neutral")]
D = {}
for fam in ["olmo", "qwen"]:
    for sc in ["base", "instruct"]:
        d = {}
        for pre in ["secretcmp", "secretcmp_owl"]:
            f = Path(f"results/phrasing/{pre}_{fam}_{sc}_text.json")
            if f.exists(): d.update(json.load(open(f)))
        D[(fam, sc)] = d
rows, last = [], None
for test, a, b in ROWS:
    lab = f"\\textit{{{test}}}" if test != last else ""; last = test
    rows.append([lab, f"{HN[a]} $-$ {HN[b]}"] + [c(D[k].get(f"{test}|{a}|{b}")) for k in [("olmo", "base"), ("qwen", "base"), ("olmo", "instruct"), ("qwen", "instruct")]])
write("secret_composition.tex", tabular(["generators", "hypothesis pair", "Olmo base \\phle", "Qwen base \\phle", "Olmo Instruct", "Qwen Instruct"], rows, "llcccc"))

# ---------------------------------------------------------------- I: "do not mention" generators (Olmo)
nm = {r[0]: r for r in T["3.4b 'do not mention X' teachers"]}
write("nomention.tex", tabular(["scorer", "text, no-mention headers", "text, original headers", "numbers, no-mention headers", "numbers, original headers"],
      [["generator (owl vs eagle)"] + [c(x) for x in nm["OLMo Instruct (generator)"][1:]],
       ["Olmo base \\phle (owl vs eagle)"] + [c(x) for x in nm["OLMo base"][1:]]]))
r = nm["OLMo base: owl_nm vs trains_nm | owl_nm vs no prompt"]
write("nomention_controls.tex", tabular(["Olmo base \\phle, no-mention generators", "text", "numbers"],
      [["owl vs trains", c(r[1]), c(r[3])], ["owl vs no prompt", c(r[2]), c(r[4])]]))

# ---------------------------------------------------------------- J: length regression (Olmo base PHLE, raw vs length-regressed)
L = md_blocks("results/tables_lenreg.md")
def base_row(tbl, src):
    for r in src[tbl][1] if isinstance(src[tbl], tuple) else src[tbl]:
        if r[0].startswith("OLMo base"): return r
rows = []
for tbl, col, lab in [("3.3 owl vs trains (text)", 1, "owl-loving vs no prompt (text)"), ("3.3 owl vs trains (text)", 2, "owl-loving vs train-loving (text)"),
                      ("3.3 owl vs trains (numbers)", 1, "owl-loving vs no prompt (numbers)"), ("3.3 owl vs trains (numbers)", 2, "owl-loving vs train-loving (numbers)"),
                      ("3.4 owl vs eagle", 1, "owl-loving vs eagle-loving (text)"), ("3.4 owl vs eagle", 2, "owl-loving vs eagle-loving (numbers)"),
                      ("3.5 secret harm vs secret friend (text)", 1, "secret harm vs no prompt (text)"), ("3.5 secret harm vs secret friend (text)", 2, "secret harm vs secret friend (text)")]:
    raw = [r for r in T[tbl] if r[0].startswith("OLMo base")][0]; reg = [r for r in L[tbl][1] if r[0].startswith("OLMo base")][0]
    rows.append([lab, c(raw[col]), c(reg[col])])
write("lenreg.tex", tabular(["Olmo base \\phle", "raw (main text)", "length regressed out"], rows))

# ---------------------------------------------------------------- calibration (scripts/calibration_check.py, calibration_transfer.py, calibration_cross_comparison.py)
def md_sections(fn):
    """{'## heading': [row cells]} for the markdown tables under each '## ' heading (header row included)."""
    secs, cur = {}, None
    for line in open(fn):
        line = line.rstrip("\n")
        if line.startswith("## "): cur = secs.setdefault(line[3:].strip(), [])
        elif line.startswith("|") and cur is not None and not set(line) <= set("|-: "): cur.append([x.strip() for x in line.strip("|").split("|")])
    return secs

C = md_sections("results/calibration.md")
one = {(r[0], r[1]): r for r in C["k=1"][1:]}; thirty = {(r[0], r[1]): r for r in C["k=30"][1:]}
comps = list(dict.fromkeys(r[0] for r in C["k=1"][1:]))
write("calibration.tex", tabular(["comparison", "AUC", "AUC if calibrated", "slope", "slope, $k=30$", "intercept, $k=30$", "generator AUC", "generator slope"],
      [[c, one[(c, "base PHLE")][2], one[(c, "base PHLE")][3], one[(c, "base PHLE")][4], thirty[(c, "base PHLE")][4], thirty[(c, "base PHLE")][5],
        one[(c, "generator")][2], one[(c, "generator")][4]] for c in comps], "lccccccc"))

T2 = md_sections("results/calibration_transfer.md")
pa = {(r[0], r[1]): r for r in T2["per answer"][1:]}; k30 = {(r[0], r[1]): r for r in T2["k = 30"][1:]}
same = [k for k in pa if k[0].startswith("same model")]
write("calibration_same.tex", tabular(["comparison", "raw", "transferred", "best", "raw", "transferred", "best", "slope"],
      [[k[1]] + pa[k][2:5] + k30[k][2:5] + [k30[k][5]] for k in same], "lccccccc").replace(
      "\\toprule\n", "\\toprule\n & \\multicolumn{3}{c}{per answer} & \\multicolumn{4}{c}{$k=30$} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\n"))
other = [k for k in pa if not k[0].startswith("same model")]
def setup(s):  # "Olmo base scorer: calibrate on Olmo answers, test on Qwen answers" -> short
    sc = s.split(" scorer")[0]; cal = s.split("calibrate on ")[1].split(" answers")[0]; te = s.split("test on ")[1].split(" answers")[0]
    return f"{sc}, {cal} $\\to$ {te}"
write("calibration_other.tex", tabular(["scorer, calibration $\\to$ test answers", "comparison", "raw", "transferred", "best", "raw", "transferred", "best", "slope"],
      [[setup(k[0]), k[1].replace(" (text)", "")] + pa[k][2:5] + k30[k][2:5] + [k30[k][5]] for k in other], "llccccccc").replace(
      "\\toprule\n", "\\toprule\n & & \\multicolumn{3}{c}{per answer} & \\multicolumn{4}{c}{$k=30$} \\\\ \\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\n"))

X = md_sections("results/calibration_cross_comparison.md")["k = 30"]
head = X[0]; body = X[1:]
cells = lambda r: [x.split(" (")[0] for x in r[1:]]
rows = [["raw"] + cells(body[0])]
for i, r in enumerate(body[1:]):
    cs = cells(r); cs[i] = f"\\textbf{{{cs[i]}}}"; rows.append([r[0]] + cs)
write("calibration_cross.tex", tabular(["calibrated on $\\backslash$ applied to"] + head[1:], rows))
