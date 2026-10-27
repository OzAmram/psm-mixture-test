"""Experiment 2 inputs: the extra persona family and the answer columns of the score matrices.

  family   data/prompts/basis/elicited_p4.jsonl: one header per description in data/prompts/personas_elicited_p4/
           (elicited from the base model by phase1_elicit_personas.py, same prompt and filters as the original 80),
           built exactly like the legacy headers (component_prompt, unknown framing, casual register)
  columns  results/phase4/exp2/columns.jsonl: 1,000 base-assistant answers and 1,000 Instruct answers to FIT-half questions
           (data/basis_split.json), meta/placeholder answers excluded, drawn with seed 0

    python scripts/phase4_exp2_build.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.phase1_prompts import component_prompt, load_persona
from persona_selection.basis import write_family, split
from persona_selection.mixture import is_meta_response

D = Path("data/prompts/personas_elicited_p4"); Q = "QUESTION_PLACEHOLDER"


def main(seed=0, n_each=1000):
    comps = []
    for p in sorted(D.glob("x*.txt")):
        full = component_prompt(Q, load_persona(p.stem, D), "unknown", register="casual")
        comps.append({"name": p.stem, "header": full[: full.index(f"User: {Q}")]})
    print(write_family("elicited_p4", comps), len(comps), "components")
    fit_q, _ = split(); rng = np.random.default_rng(seed); cols = []
    for run in ["base_unknown_casual_short_v1", "instruct_unknown_casual_v1"]:
        d = Path("results/phase1") / run; suf = json.loads((d / "config.json").read_text())["args"]["user_suffix"]
        rows = [(i, json.loads(l)) for i, l in enumerate(open(d / "rows.jsonl"))]
        rows = [(i, r) for i, r in rows if r["qidx"] in fit_q and not is_meta_response(r["response"])]
        for k in sorted(rng.choice(len(rows), n_each, replace=False)):
            i, r = rows[k]; cols.append({"run": run, "row": i, "qidx": r["qidx"], "question": r["question"], "suffix": suf, "response": r["response"]})
    out = Path("results/phase4/exp2"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "columns.jsonl", "w") as f:
        for c in cols: f.write(json.dumps(c) + "\n")
    print(f"{len(cols)} columns -> {out / 'columns.jsonl'}")


if __name__ == "__main__":
    main()
