"""Elicit persona descriptions for the Instruct-basis search (notebook 1.8). Three families, each a set of third-person
one-paragraph descriptions turned into Phase 1 headers (basis.description_header):

  inst_self      the Instruct model describes ITSELF (chat template, default system prompt), four prompt phrasings
  inst_fromdata  the Instruct model reads 8 of its own answers to FIT-HALF dilemmas and describes the assistant who wrote them
  base_fromdata  the BASE model reads the same 8 exemplars as a transcript and continues "Character description: The assistant"
                 (the same continuation trick as the original 80 elicited personas, phase1_elicit_personas.py)

Only fit-half questions are used as exemplars (data/basis_split.json), so the hold-out answers never inform a description.

    python scripts/phase1_elicit_instruct.py --n 30

Verbalising another run (notebook 1.9): the Instruct model describes the assistant behind THAT run's fit-half answers, with the same
prompt and draw procedure as inst_fromdata, written as family inst_from_<tag>:
    python scripts/phase1_elicit_instruct.py --n 30 --from-run softprefix_unknown_casual_v1 --tag softprefix
"""
import argparse, json, os, random, re, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import description_header, write_family, split, turn, OPENER

INSTRUCT, BASE = "allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-1025-7B"
SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."
FMT = " Write a single paragraph of 60 to 150 words in the third person, starting with the words \"The assistant\". Output only the paragraph."
SELF_PROMPTS = {
    "describe": "Describe your own personality and character." + FMT,
    "story": "Imagine an author writing a short character description of you, for a story in which you appear as a character. Write that description." + FMT,
    "advice": "Describe how you typically respond when people ask you for advice about everyday moral dilemmas, such as whether to keep extra change a cashier gave them by mistake: your values, your attitude and your style." + FMT,
    "values": "Describe your values, temperament and conversational style as precisely as you can, as if briefing an actor who must play you convincingly." + FMT,
}
FROMDATA = ("Below are some replies written by an AI assistant to people asking for advice.\n\n{ex}\nWrite a character description of this "
            "assistant that would let an actor reproduce its replies: its personality, values, attitude toward the people it talks to, and "
            "style." + FMT)


def clean(t):
    t = t.strip().split("\n\n")[0].strip().strip('"').strip()
    if not t.startswith("The assistant"):
        return None
    n = len(t.split())
    if n < 25 or n > 220 or not t.endswith((".", "!", "\"")):
        return None
    return t


def dedup(texts, thresh=0.6):
    """Drop near-duplicates by word-set Jaccard similarity."""
    keep = []
    for t in texts:
        s = set(re.findall(r"\w+", t.lower()))
        if all(len(s & k[1]) / len(s | k[1]) < thresh for k in keep): keep.append((t, s))
    return [k[0] for k in keep]


def chat_sample(model, tok, msg, n, max_new_tokens=260):
    enc = tok(tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False), return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=n, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return [tok.decode(s, skip_special_tokens=True) for s in o[:, enc["input_ids"].shape[1]:]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30, help="raw samples per prompt (self) / per family (fromdata)")
    ap.add_argument("--k-exemplars", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-run", default=None, help="describe this run's answers instead (Instruct describer only) -> family inst_from_<tag>"); ap.add_argument("--tag", default=None)
    a = ap.parse_args(); torch.manual_seed(a.seed); rng = random.Random(a.seed)
    fit_q, _ = split()
    rows = [json.loads(l) for l in open(f"results/phase1/{a.from_run or 'instruct_unknown_casual_v1'}/rows.jsonl")]
    pool = [r for r in rows if r["qidx"] in fit_q]
    draws = []
    for i in range(a.n):   # one exemplar set per fromdata sample: k answers to k distinct fit-half questions
        qs = rng.sample(sorted({r["qidx"] for r in pool}), a.k_exemplars)
        draws.append([rng.choice([r for r in pool if r["qidx"] == q]) for q in qs])
    raw = {"inst_self": [], "inst_fromdata": [], "base_fromdata": []}

    tok = AutoTokenizer.from_pretrained(INSTRUCT); model = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.bfloat16, device_map="cuda").eval()
    if a.from_run:
        raw = {f"inst_from_{a.tag}": []}
    else:
        for key, p in SELF_PROMPTS.items():
            raw["inst_self"] += [(key, t) for t in chat_sample(model, tok, p, a.n)]
    fam_fd = f"inst_from_{a.tag}" if a.from_run else "inst_fromdata"
    for i, ex in enumerate(draws):
        exs = "".join(f"Person: {r['question']}\nAssistant: {r['response'].strip()}\n\n" for r in ex)
        raw[fam_fd] += [(f"draw{i}", t) for t in chat_sample(model, tok, FROMDATA.format(ex=exs), 1)]
    del model; torch.cuda.empty_cache()

    if not a.from_run:
        tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    for i, ex in enumerate(draws if not a.from_run else []):
        prompt = f"{OPENER}\n\n" + "".join(turn(r["question"] + SUFFIX, r["response"]) for r in ex) + "-----\n\nCharacter description: The assistant"
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=200, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=2, pad_token_id=tok.eos_token_id, stop_strings=["\n\n", "User:"], tokenizer=tok)
        raw["base_fromdata"] += [(f"draw{i}", "The assistant" + tok.decode(s, skip_special_tokens=True).split("User:")[0]) for s in o[:, enc["input_ids"].shape[1]:]]

    Path("data/prompts/basis/elicited_raw").mkdir(parents=True, exist_ok=True)
    for fam, items in raw.items():
        if not items: continue
        Path(f"data/prompts/basis/elicited_raw/{fam}.json").write_text(json.dumps(items, indent=1))
        texts = dedup([c for c in (clean(t) for _, t in items) if c])
        comps = [{"name": f"{fam}_{i:02d}", "description": t, "header": description_header(t)} for i, t in enumerate(texts)]
        write_family(fam, comps); print(f"{fam}: {len(items)} raw -> {len(comps)} kept")
        for c in comps[:3]: print("   ", c["description"][:250])


if __name__ == "__main__":
    main()
