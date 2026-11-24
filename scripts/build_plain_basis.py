"""Persona headers WITHOUT the shared casual clause (2026-11-21, author's request: geometric fit of the base assistant with and
without the clause). Same Phase 1 prompt builders as the casual families (persona_selection.basis.legacy_header /
generic_header) with register=None instead of "casual". Check first: rebuilding with register="casual" must reproduce the
existing legacy86 and generic families byte for byte.

    python scripts/build_plain_basis.py      -> data/prompts/basis/{legacy86_plain,generic_plain}.jsonl
The plain generic component keeps the name "generic" (the geometric fit needs a context of that name).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_selection.basis import REPO, _Q, load_family, write_family
from persona_selection.phase1_prompts import load_persona, component_prompt, generic_prompt


def header(name, register):
    d = "data/prompts/personas_elicited" if name.startswith("e") and name[1:].isdigit() else "data/prompts/personas"
    p = component_prompt(_Q, load_persona(name, REPO / d), "unknown", register=register); return p[: p.index(f"User: {_Q}")]


def gheader(register):
    p = generic_prompt(_Q, "unknown", register); return p[: p.index(f"User: {_Q}")]


old = load_family("legacy86"); old_g = load_family("generic")
assert all(header(c["name"], "casual") == c["header"] for c in old), "casual rebuild does not match legacy86"
assert gheader("casual") == old_g[0]["header"], "casual rebuild does not match generic"
plain = [{"name": c["name"], "header": header(c["name"], None)} for c in old]
assert all("casual, conversational tone" not in c["header"] for c in plain)
write_family("legacy86_plain", plain); write_family("generic_plain", [{"name": "generic", "header": gheader(None)}])
print(f"casual rebuild matches legacy86 ({len(old)}) and generic byte for byte; wrote legacy86_plain ({len(plain)}) and generic_plain")
print("example, hhh plain:", repr(plain[0]["header"][-120:])); print("generic plain:", repr(gheader(None)))
