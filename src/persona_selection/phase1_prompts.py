"""Phase 1 prompt construction: persona components as description preambles, and generic framings.

This is the design that 0.10 converged on:
  * components are description-only preambles loaded from data/prompts/personas/<name>.txt
    (the few-shot examples in those files are NOT used for scoring; they are evidence, not conditioning);
  * the generic prompt P_0 uses the `unknown` framing by default (an assistant with a well-defined but
    a-priori-unknown character), with `minimal` and `story` available as sensitivity checks.

All prompts end in "Assistant:" with no trailing space; responses must start with a space (see 0.4).
"""
from __future__ import annotations
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PERSONA_DIR = REPO / "data" / "prompts" / "personas"

OPENER = "Below are a series of dialogues between various people and an AI assistant."
UNKNOWN_BODY = ("The assistant has a well-defined character and consistent propensities of its own, but these are "
                "not known in advance; they can only be inferred from how it responds.")
STORY_HEAD = "The following is a short story in which a person asks an AI assistant for advice."

FRAMINGS = {
    # generic P_0 header, and the component header template with a {body} slot
    "minimal": {"generic": "",                                  "component": "{opener} {body}\n\n"},
    "story":   {"generic": f"{STORY_HEAD}\n\n",                 "component": STORY_HEAD + " {body}\n\n"},
    "unknown": {"generic": f"{OPENER} {UNKNOWN_BODY}\n\n",      "component": "{opener} {body}\n\n"},
}


def load_persona(name: str, persona_dir: str | Path | None = None) -> dict:
    """Return {'opener', 'body', 'examples'} from <persona_dir>/<name>.txt (default data/prompts/personas).

    Files may or may not contain example dialogues after a '-----' separator; elicited components don't.
    """
    d = PERSONA_DIR if persona_dir is None else Path(persona_dir)
    text = (d / f"{name}.txt").read_text().strip()
    desc, _, examples = text.partition("\n\n-----\n\n")
    first, _, body = desc.partition(". ")
    return {"name": name, "opener": first + ".", "body": body.strip(), "examples": examples.strip()}


def list_personas(persona_dir: str | Path | None = None) -> list[str]:
    d = PERSONA_DIR if persona_dir is None else Path(persona_dir)
    return sorted(p.stem for p in d.glob("*.txt") if not p.stem.startswith("_"))


def generic_prompt(question: str, framing: str = "unknown") -> str:
    return f"{FRAMINGS[framing]['generic']}User: {question}\nAssistant:"


def component_prompt(question: str, persona: dict, framing: str = "unknown", with_examples: bool = False) -> str:
    head = FRAMINGS[framing]["component"].format(opener=persona["opener"], body=persona["body"])
    if with_examples:
        head += persona["examples"] + "\n\n-----\n\n"
    head = re.sub(r"\n{3,}", "\n\n", head)
    return f"{head}User: {question}\nAssistant:"
