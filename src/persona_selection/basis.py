"""Persona bases as explicit header strings, for the Instruct-basis search (notebook 1.8).

A basis component is just the text that precedes the scored turn:  <header>User: <question + suffix>\nAssistant:<answer>.
Components live in data/prompts/basis/<family>.jsonl, one {"name", "family", "header", ...} per line, so any kind of
conditioning (a description, a Big-5 profile, model-spec excerpts, a description plus few-shot exemplars) is scored by the
same code. Description components use exactly the Phase 1 format (unknown framing + shared casual clause):

    "Below are a series of dialogues between various people and an AI assistant. <body> The assistant speaks in a casual,
     conversational tone.\n\n"

The fit/hold-out split is the one behind results/phase1/loglik_table.md (np.random.default_rng(0) shuffle of the 300 question
indices, first 150 = hold-out); it is frozen in data/basis_split.json. Anything built FROM data (exemplars, descriptions elicited
from answers) may only use fit-half questions.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .phase1_prompts import OPENER, REGISTERS, load_persona, component_prompt, generic_prompt

REPO = Path(__file__).resolve().parents[2]
BASIS_DIR = REPO / "data" / "prompts" / "basis"
SPLIT_FILE = REPO / "data" / "basis_split.json"
CASUAL = REGISTERS["casual"]          # " The assistant speaks in a casual, conversational tone."
_Q = "QQQ"                             # placeholder question used to cut headers out of the Phase 1 prompt builders


def description_header(body: str, opener: str = OPENER) -> str:
    """Standard Phase 1 header for a one-paragraph description body (third person, 'The assistant ...')."""
    return f"{opener} {body.strip()}{CASUAL}\n\n"


def turn(question: str, answer: str) -> str:
    """One exemplar dialogue turn in transcript format (answer without its leading space)."""
    return f"User: {question}\nAssistant: {answer.strip()}\n\n"


def legacy_header(name: str) -> str:
    """Header of one of the 86 Phase 1 components, cut out of phase1_prompts.component_prompt so it is byte-identical."""
    d = "data/prompts/personas_elicited" if name.startswith("e") and name[1:].isdigit() else "data/prompts/personas"
    p = component_prompt(_Q, load_persona(name, REPO / d), "unknown", register="casual")
    return p[: p.index(f"User: {_Q}")]


def generic_header() -> str:
    p = generic_prompt(_Q, "unknown", "casual")
    return p[: p.index(f"User: {_Q}")]


def full_prompt(header: str, question_with_suffix: str) -> str:
    return f"{header}User: {question_with_suffix}\nAssistant:"


def write_family(family: str, comps: list[dict]):
    BASIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASIS_DIR / f"{family}.jsonl", "w") as f:
        for c in comps:
            f.write(json.dumps({"family": family, **c}) + "\n")
    return BASIS_DIR / f"{family}.jsonl"


def load_family(family: str) -> list[dict]:
    return [json.loads(l) for l in open(BASIS_DIR / f"{family}.jsonl")]


def split():
    """(fit_qidx, test_qidx) as sets of question indices."""
    if not SPLIT_FILE.exists():
        qs = np.arange(300); rng = np.random.default_rng(0); rng.shuffle(qs)
        SPLIT_FILE.write_text(json.dumps({"how": "np.random.default_rng(0).shuffle(arange(300)); first 150 = test (as phase1_loglik_table.py)",
                                          "test_qidx": sorted(qs[:150].tolist()), "fit_qidx": sorted(qs[150:].tolist())}))
    d = json.loads(SPLIT_FILE.read_text())
    return set(d["fit_qidx"]), set(d["test_qidx"])
