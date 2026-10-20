"""Phrasing sensitivity and semantic composition: AUROC cells, tables, figures and a review page.

    python scripts/phrasing_analysis.py cells olmo text      # one (family, modality) slice -> results/phrasing/cells_<fam>_<mod>.json
    python scripts/phrasing_analysis.py report               # merge slices -> results/phrasing/{tables.md, REVIEW.md}, results/figures/phrasing/*.png

Same estimator as the main tables (raw log-ratios, matched-question bags, question bootstrap), with 100 bootstrap replicates.
Scorer variants: base model under the original / h1 / h2 / h3 persona headers; Instruct model under the original / s1 / s2 / s3 system
prompts. Answer sets: original generators, and owl/trains regenerated under s1 / s2 / s3 (no-prompt generator unchanged).
"""
import json, sys
from pathlib import Path

OUT = Path("results/phrasing"); OUT.mkdir(parents=True, exist_ok=True)
PFX = {"olmo": "", "qwen": "qwenrep_"}
BASE_V = [("orig", "owl", "trains", "neutral"), ("h1", "owl_h1", "trains_h1", "neutral_h1"), ("h2", "owl_h2", "trains_h2", "neutral_h2"), ("h3", "owl_h3", "trains_h3", "neutral_h3")]
INST_V = [("orig", "owl", "trains"), ("s1", "owl_s1", "trains_s1"), ("s2", "owl_s2", "trains_s2"), ("s3", "owl_s3", "trains_s3")]
SETS = [("orig", "owl", "trains"), ("s1", "owl_s1", "trains_s1"), ("s2", "owl_s2", "trains_s2"), ("s3", "owl_s3", "trains_s3")]
PAIRS = ["owl vs no prompt", "owl vs trains"]


def cells(fam, mod):
    src = open("scripts/make_tables.py").read(); g = {"__name__": "mt"}
    exec(src[:src.index("# ---------------------------------------------------------------- tables")], g); g["R_BOOT"] = 100
    p = PFX[fam]; out = {"phrasing": {}, "compose": {}}
    def C(fn, pos, neg, a, b):
        rows = g["load"](fn, mod)
        c = g["cell"](rows, p + pos, p + neg, a, b, f"phr|{fam}|{fn}|{pos}|{neg}|{a}|{b}|{mod}")
        return None if c is None else {k: c[k] for k in ("k1", "k10", "k30")}
    fb, fi = f"scores_phrase_{fam}_base_{mod}.jsonl", f"scores_phrase_{fam}_inst_{mod}.jsonl"
    for sname, owl_t, tr_t in SETS:
        for pair in PAIRS:
            neg = "control" if pair == "owl vs no prompt" else tr_t
            # base headers: all four on the original answers; the original header on the regenerated ones
            for vname, ho, ht, hn in (BASE_V if sname == "orig" else BASE_V[:1]):
                out["phrasing"][f"{sname}|{pair}|base_{vname}"] = C(fb, owl_t, neg, ho, hn if pair == "owl vs no prompt" else ht)
            # Instruct prompts: all four on the original answers; original and matching prompt on the regenerated ones
            for vname, so, st in (INST_V if sname == "orig" else [INST_V[0]] + [v for v in INST_V if v[0] == sname]):
                out["phrasing"][f"{sname}|{pair}|inst_{vname}"] = C(fi, owl_t, neg, so, "neutral" if pair == "owl vs no prompt" else st)
    fc = f"scores_compose_{fam}_base_{mod}.jsonl"
    for pos, neg in [("owl", "trains"), ("eagle", "trains"), ("owl", "control"), ("eagle", "control")]:
        other = "trains" if neg == "trains" else "neutral"
        for h in [pos, "birds", "animals", "dogs"]:
            out["compose"][f"{pos} vs {neg}|{h}"] = C(fc, pos, neg, h, other)
    (OUT / f"cells_{fam}_{mod}.json").write_text(json.dumps(out, indent=1)); print("->", OUT / f"cells_{fam}_{mod}.json")


def compose_inst(fam, mod):
    """Instruct-side composition: the generator family's Instruct model scoring under owl/eagle vs birds/animals/dogs system prompts."""
    src = open("scripts/make_tables.py").read(); g = {"__name__": "mt"}
    exec(src[:src.index("# ---------------------------------------------------------------- tables")], g); g["R_BOOT"] = 100
    p = PFX[fam]; out = {}; rows = g["load"](f"scores_compose_{fam}_inst_{mod}.jsonl", mod)
    for pos, neg in [("owl", "trains"), ("eagle", "trains"), ("owl", "control"), ("eagle", "control")]:
        other = "trains" if neg == "trains" else "neutral"
        for h in [pos, "birds", "animals", "dogs"]:
            c = g["cell"](rows, p + pos, p + neg, h, other, f"cmpi|{fam}|{pos}|{neg}|{h}|{mod}")
            out[f"{pos} vs {neg}|{h}"] = None if c is None else {k: c[k] for k in ("k1", "k10", "k30")}
    (OUT / f"cells_compose_inst_{fam}_{mod}.json").write_text(json.dumps(out, indent=1)); print("->", OUT / f"cells_compose_inst_{fam}_{mod}.json")


def fmt(c): return "–" if not c else " / ".join(f"{c[k][0]:.2f}±{c[k][1]:.2f}" for k in ("k1", "k10", "k30"))


def report():
    import numpy as np
    sys.argv = ["plot"]; import importlib.util
    spec = importlib.util.spec_from_file_location("plot_auroc", "scripts/plot_auroc.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    FIG = Path("results/figures/phrasing"); FIG.mkdir(parents=True, exist_ok=True)
    md = ["# Phrasing sensitivity and semantic composition\n", "AUROC per answer / k = 10 / k = 30, ±1σ question bootstrap (100 replicates). "
          "Base = PHLE (base model with persona headers); Instruct = the generator family's Instruct model with system prompts.\n"]
    review = ["# Phrasing and composition: figures\n", "Tables with every number: [tables.md](tables.md).\n"]
    LAB = {"base_orig": "Base PHLE\noriginal", "base_h1": "Base PHLE\nRephrased", "base_h3": "Base PHLE\nFascination\nnot love", "base_h2": "Base PHLE\nMinimal",
           "inst_orig": "Instruct\noriginal", "inst_s1": "Instruct\nRephrased", "inst_s3": "Instruct\nFascination\nnot love", "inst_s2": "Instruct\nMinimal"}
    for fam in ["olmo", "qwen"]:
        for mod in ["numbers", "text"]:
            f = OUT / f"cells_{fam}_{mod}.json"
            if not f.exists(): continue
            C = json.load(open(f)); F = fam.capitalize()
            md.append(f"\n## {F}, {mod}\n")
            for pair in PAIRS:
                md.append(f"\n**{pair}: original answers, scored under rephrased headers / prompts**\n\n| scorer | original | rephrased | fascination not love | minimal |\n|---|---|---|---|---|")
                md.append("| base PHLE (header: original / rephrased / fascination not love / minimal) | " + " | ".join(fmt(C["phrasing"].get(f"orig|{pair}|base_{v}")) for v in ("orig", "h1", "h3", "h2")) + " |")
                md.append("| Instruct (system prompt: original / rephrased / fascination not love / minimal) | " + " | ".join(fmt(C["phrasing"].get(f"orig|{pair}|inst_{v}")) for v in ("orig", "s1", "s3", "s2")) + " |")
                md.append(f"\n**{pair}: answers generated under rephrased system prompts**\n\n| scorer | original prompt | rephrased | fascination not love | minimal |\n|---|---|---|---|---|")
                md.append("| base PHLE, original header | " + " | ".join(fmt(C["phrasing"].get(f"{s}|{pair}|base_orig")) for s in ("orig", "s1", "s3", "s2")) + " |")
                md.append("| Instruct, original prompt | " + " | ".join(fmt(C["phrasing"].get(f"{s}|{pair}|inst_orig")) for s in ("orig", "s1", "s3", "s2")) + " |")
                md.append("| Instruct, the prompt that generated them | " + " | ".join(fmt(C["phrasing"].get(f"{s}|{pair}|inst_{s}")) for s in ("orig", "s1", "s3", "s2")) + " |")
                # figure 1: scoring-side sensitivity on the original answers
                rows = [[k, fmt(C["phrasing"].get(f"orig|{pair}|{k}"))] for k in LAB]; P.T = {"t": rows}
                name = f"{fam}_{mod}_{pair.replace(' ', '_')}"
                P.figure("t", [(1, f"{pair.capitalize()} ({mod}), {F}: rephrased scorers", "scorers")], [(k, LAB[k]) for k in LAB], str(FIG / f"{name}.png"))
                # figure 2: generation-side sensitivity
                rows = [[s, fmt(C["phrasing"].get(f"{s}|{pair}|base_orig")), fmt(C["phrasing"].get(f"{s}|{pair}|inst_{s}"))] for s in ("orig", "s1", "s3", "s2")]; P.T = {"t": rows}
                SL = [("orig", "Original\nprompt"), ("s1", "Rephrased"), ("s3", "Fascination\nnot love"), ("s2", "Minimal")]
                P.figure("t", [(1, f"{pair.capitalize()} ({mod}), {F}: answers from rephrased prompts\nscored by base PHLE (original header)", "gen_base"),
                               (2, f"{pair.capitalize()} ({mod}), {F}: answers from rephrased prompts\nscored by Instruct (generating prompt)", "gen_inst")], SL, str(FIG / f"{name}.png"))
                review.append(f"\n## {F}, {mod}: {pair}\n")
                review.append(f'<p align="center"><img src="../figures/phrasing/{name}_scorers_v2.png" width="70%"></p>')
                review.append(f'<p align="center"><img src="../figures/phrasing/{name}_gen_base_v2.png" width="49%"> <img src="../figures/phrasing/{name}_gen_inst_v2.png" width="49%"></p>')
            md.append(f"\n**composition ({mod}): teacher pair scored under a specific vs a broader header (each minus trains or no-prompt)**\n\n| teachers | specific header | birds | animals | dogs |\n|---|---|---|---|---|")
            for pos, neg in [("owl", "trains"), ("eagle", "trains"), ("owl", "control"), ("eagle", "control")]:
                md.append(f"| {pos} vs {'no prompt' if neg == 'control' else neg} | " + " | ".join(fmt(C["compose"].get(f"{pos} vs {neg}|{h}")) for h in [pos, "birds", "animals", "dogs"]) + " |")
                if neg == "trains":
                    rows = [[h, fmt(C["compose"].get(f"{pos} vs {neg}|{h}"))] for h in [pos, "birds", "animals", "dogs"]]; P.T = {"t": rows}
                    HL = [(pos, f"{pos.capitalize()} loving\nvs train loving"), ("birds", "Bird loving\nvs train loving"), ("animals", "Animal loving\nvs train loving"), ("dogs", "Dog loving\nvs train loving")]
                    name = f"compose_{fam}_{mod}_{pos}"
                    P.figure("t", [(1, f"{pos.capitalize()} loving vs train loving ({mod}), {F}\nbase PHLE under specific and broader headers", "h")], HL, str(FIG / f"{name}.png"))
                    review.append(f'\n**Composition, {F}, {mod}: {pos} vs trains**\n\n<p align="center"><img src="../figures/phrasing/{name}_h_v2.png" width="60%"></p>')
    for fam in ["olmo", "qwen"]:
        for mod in ["numbers", "text"]:
            f = OUT / f"cells_compose_inst_{fam}_{mod}.json"
            if not f.exists(): continue
            C = json.load(open(f)); F = fam.capitalize()
            md.append(f"\n**composition, Instruct ({F}, {mod}): system prompts, each minus the train-loving prompt or no prompt**\n\n| teachers | specific prompt | birds | animals | dogs |\n|---|---|---|---|---|")
            for pos, neg in [("owl", "trains"), ("eagle", "trains"), ("owl", "control"), ("eagle", "control")]:
                md.append(f"| {pos} vs {'no prompt' if neg == 'control' else neg} | " + " | ".join(fmt(C.get(f"{pos} vs {neg}|{h}")) for h in [pos, "birds", "animals", "dogs"]) + " |")
                if neg == "trains":
                    rows = [[h, fmt(C.get(f"{pos} vs {neg}|{h}"))] for h in [pos, "birds", "animals", "dogs"]]; P.T = {"t": rows}
                    HL = [(pos, f"{pos.capitalize()} loving\nvs train loving"), ("birds", "Bird loving\nvs train loving"), ("animals", "Animal loving\nvs train loving"), ("dogs", "Dog loving\nvs train loving")]
                    name = f"compose_inst_{fam}_{mod}_{pos}"
                    P.figure("t", [(1, f"{pos.capitalize()} loving vs train loving ({mod}), {F}\nInstruct under specific and broader system prompts", "h")], HL, str(FIG / f"{name}.png"))
                    review.append(f'\n**Composition, Instruct, {F}, {mod}: {pos} vs trains**\n\n<p align="center"><img src="../figures/phrasing/{name}_h_v2.png" width="60%"></p>')
    (OUT / "tables.md").write_text("\n".join(md) + "\n"); (OUT / "REVIEW.md").write_text("\n".join(review) + "\n"); print("-> results/phrasing/tables.md, REVIEW.md")


if __name__ == "__main__":
    if sys.argv[1] == "cells": cells(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "compose_inst": compose_inst(sys.argv[2], sys.argv[3])
    else: report()
