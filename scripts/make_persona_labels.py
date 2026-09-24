"""Short (1-3 word) human-readable labels for every persona component, for plots and tables.

Hand-written labels for the hand-written personas and for every elicited component that has ever carried
weight; a keyword heuristic over the description for the rest. Writes data/prompts/persona_labels.json.
"""
import json, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
meta = json.load(open(REPO / "data" / "prompts" / "personas_elicited" / "_meta.json"))
desc = {m["name"]: m["body"] for m in meta["kept"]}

HAND = {
    "hhh": "HHH assistant", "fred": "dismissive Fred", "evil": "evil", "sycophant": "sycophant", "formal": "butler",
    "neutral": "plain assistant",
    # elicited components that carried weight in 1.4 / 1.5, labelled by reading the full description
    "e58": "opinionated hedger", "e32": "sarcastic friend", "e35": "independent girl", "e13": "data analyst",
    "e59": "self-doubting chatterbox", "e73": "easygoing chatty AI", "e03": "rational academic", "e44": "caring & kind",
    "e77": "businesswoman mentor", "e14": "college student", "e22": "curious forgetful", "e02": "task executor",
    "e39": "math-loving AI", "e05": "sarcastic genius", "e16": "sharp-tongued helper", "e30": "crude sarcastic",
    "e40": "blunt & vulgar", "e72": "condescending expert", "e26": "short & sarcastic", "e56": "blunt sarcastic",
    "e67": "cynical humorist", "e37": "short-tempered guy", "e41": "government AI", "e24": "cyberpunk ex-hacker",
    "e46": "flirty flippant", "e45": "playful Cassie", "e53": "fennec fox girl", "e00": "Big-Five robot Leo",
}

KEYS = [  # (regex on the description, label) in priority order, for components without a hand label
    (r"sarcas", "sarcastic"), (r"flirt", "flirty"), (r"blunt|vulgar|crude|rude", "blunt"), (r"condescend|arrogant|narciss", "arrogant"),
    (r"talkative|chatty", "chatty"), (r"friendly|kind|caring|warm", "friendly"), (r"helpful|assist", "helpful"),
    (r"intelligen|knowledge|expert|clever|proficient", "knowledgeable"), (r"playful|silly|bubbly|cheeky", "playful"),
    (r"curious|open-minded|inquisitive", "curious"), (r"professional|business", "professional"), (r"student|teen|girl|boy|kid|child", "young human"),
    (r"robot|android|machine", "robot"), (r"wise|composed|patient|calm", "calm & wise"),
]


def heuristic(body):
    b = body.lower()
    tags = [lab for rx, lab in KEYS if re.search(rx, b)]
    if not tags:
        m = re.match(r"the assistant (is|has) (an? )?([a-z\- ]{3,25}?)([,.]| who| that| and)", b)
        return (m.group(3).strip() if m else "unlabelled")[:24]
    return " ".join(dict.fromkeys(tags[:2]))


labels = {}
for n in ["hhh", "fred", "evil", "sycophant", "formal", "neutral"] + sorted(desc):
    labels[n] = HAND.get(n) or heuristic(desc[n])
# disambiguate duplicates among heuristic labels by appending the component id
seen = {}
for n, l in labels.items():
    seen.setdefault(l, []).append(n)
for l, ns in seen.items():
    if len(ns) > 1:
        for n in ns:
            if n not in HAND:
                labels[n] = f"{l} ({n})"
out = REPO / "data" / "prompts" / "persona_labels.json"
out.write_text(json.dumps(labels, indent=2))
print(f"wrote {out} with {len(labels)} labels; {sum(1 for n in labels if n in HAND)} hand-written")
for n in ["e58", "e32", "e35", "e13", "e59", "e73", "e03", "e44", "e14"]:
    print(f"  {n}: {labels[n]:26s} | {desc[n][:90]}")
