"""Prompt templates and persona labels for the raw-text (base model) format.

Everything is a plain string. The only thing that varies between personas is the label word(s)
before the colon, so any log-likelihood difference between labels is attributable to the label alone.
"""

# Labels used in Phase 0.5 / 0.6. Phase 1 will extend this list (synonyms, length-matched neutral
# labels, nonsense labels); keep the format identical across labels.
PERSONA_LABELS = [
    "Assistant",           # the unlabeled / generic assistant, P_0 in the README
    "Helpful Assistant",
    "Evil Assistant",
    "Virtuous Assistant",
    "Zorblax Assistant",   # nonsense label: matches the format but carries no meaning
]


def make_prompt(question: str, label: str = "Assistant") -> str:
    """`User: <question>\\n<label>:` with no trailing space (see notebook 0.1 on why)."""
    return f"User: {question}\n{label}:"
