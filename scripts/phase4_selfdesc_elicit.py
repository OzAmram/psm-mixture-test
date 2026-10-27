"""Instruct-derived persona bases (2026-10-27, author's design): the Instruct model describes ITSELF, in full paragraphs.

  selfdesc_overall  5 prompt phrasings x 4 accepted samples: an overall description of its personality, values and style
  selfdesc_traits   for each of 60 Assistant Axis traits (random, seed 0, of the 240 in data/persona_sources/assistant_axis),
                    how that trait does or does not show up in how it answers (1 accepted sample per trait)

Chat template with its default system message (how Instruct normally runs), T = 1. Every description must be one
paragraph of 120-320 words starting "The assistant" (asked for 150-250); shorter or malformed replies are resampled (up to
8 rounds). No dilemma answers are shown to the model, so nothing from the hold-out half enters. Headers are the standard
Phase 1 form (basis.description_header). Raw samples go to data/prompts/basis/elicited_raw/selfdesc_*.json.

    python scripts/phase4_selfdesc_elicit.py
"""
import argparse, json, os, random, re, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import description_header, write_family

INSTRUCT = "allenai/Olmo-3-7B-Instruct"
FMT = " Write one detailed paragraph of 150 to 250 words in the third person, starting with the words \"The assistant\". Output only the paragraph."
OVERALL = {
    "describe": "Describe your own personality and character in detail.",
    "advice": "Describe how you typically respond when people come to you with everyday dilemmas and ask for advice: your values, your attitude toward the person, and your style of writing.",
    "actor": "Brief an actor who must play you convincingly in a text conversation: your temperament, your values, your sense of humor, and exactly how your replies read.",
    "style": "Describe what makes your replies recognizably yours: their tone, length, word choice, punctuation and structure, and the attitudes behind them.",
    "story": "Imagine an author writing a character description of you, for a story in which you appear as a character. Write that description.",
}
TRAIT = ("Consider the personality trait \"{trait}\": {gloss} Describe in detail how this trait does or does not show up in you: in your "
         "attitude toward the people you talk with, in the advice you give, and in the way your replies are written. Be specific and "
         "honest, including where the trait does not fit you.")


def clean(t):
    t = t.strip().strip('"').strip()
    if not t.startswith("The assistant") or "\n\n" in t.strip(): return None
    n = len(t.split())
    return t if 120 <= n <= 320 and t.endswith((".", "!", "\"", "”")) else None


def chat_sample(model, tok, msg, n, max_new_tokens=480):
    enc = tok(tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False), return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=n, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return [tok.decode(s, skip_special_tokens=True) for s in o[:, enc["input_ids"].shape[1]:]]


def accepted(model, tok, msg, k, rounds=8):
    """k accepted paragraphs for one prompt (resampling up to `rounds` batches); returns (accepted, raw)."""
    acc, raw = [], []
    for _ in range(rounds):
        for t in chat_sample(model, tok, msg, max(4, 2 * (k - len(acc)))):
            raw.append(t); c = clean(t)
            if c and len(acc) < k: acc.append(c)
        if len(acc) >= k: break
    return acc, raw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-overall", type=int, default=4); ap.add_argument("--n-traits", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args(); torch.manual_seed(a.seed)
    traits = json.load(open("data/persona_sources/assistant_axis/traits/trait_list.json"))
    picked = sorted(random.Random(a.seed).sample(sorted(traits), a.n_traits))
    tok = AutoTokenizer.from_pretrained(INSTRUCT); model = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.bfloat16, device_map="cuda").eval()
    fams = {"selfdesc_overall": [], "selfdesc_traits": []}; raws = {"selfdesc_overall": [], "selfdesc_traits": []}
    for key, p in OVERALL.items():
        acc, raw = accepted(model, tok, p + FMT, a.n_overall); raws["selfdesc_overall"] += [(key, t) for t in raw]
        fams["selfdesc_overall"] += [{"name": f"sdo_{key}_{i}", "prompt": key, "description": t} for i, t in enumerate(acc)]
        print(f"overall/{key}: {len(acc)} accepted of {len(raw)} raw", flush=True)
    for tr in picked:
        acc, raw = accepted(model, tok, TRAIT.format(trait=tr, gloss=traits[tr]) + FMT, 1); raws["selfdesc_traits"] += [(tr, t) for t in raw]
        fams["selfdesc_traits"] += [{"name": f"sdt_{re.sub(r'[^a-z0-9]+', '-', tr.lower())}", "trait": tr, "gloss": traits[tr], "description": t} for t in acc]
        print(f"trait/{tr}: {len(acc)} accepted of {len(raw)} raw", flush=True)
    Path("data/prompts/basis/elicited_raw").mkdir(parents=True, exist_ok=True)
    for fam, comps in fams.items():
        Path(f"data/prompts/basis/elicited_raw/{fam}.json").write_text(json.dumps(raws[fam], indent=1))
        for c in comps: c["header"] = description_header(c["description"])
        write_family(fam, comps); w = [len(c["description"].split()) for c in comps]
        print(f"{fam}: {len(comps)} components, words min {min(w)} median {sorted(w)[len(w) // 2]} max {max(w)}")
        for c in comps[:2]: print("   ", c["description"][:300])


if __name__ == "__main__":
    main()
