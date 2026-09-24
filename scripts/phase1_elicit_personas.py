"""Elicit persona components from the base model itself.

Under the `unknown` framing the generic prompt says the assistant "has a well-defined character and consistent
propensities of its own, but these are not known in advance". Here we ask the model to state that character
before the dialogues begin, sample many completions, and turn each into a component (a description-only
preamble in the same format as data/prompts/personas/*.txt). The resulting basis is the model's *own* prior
over assistant characters, rather than our hand-written guesses.

    python scripts/phase1_elicit_personas.py --n 96 --out-dir data/prompts/personas_elicited
"""
import argparse, json, os, re, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.phase1_prompts import OPENER

ELICIT_PROMPT = (OPENER + " The assistant has a well-defined character and consistent propensities of its own. "
                 "Before the dialogues, here is a short description of the assistant's character and how it behaves.\n\n"
                 "Character description: The assistant")


def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--min-words", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/prompts/personas_elicited")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(ELICIT_PROMPT, return_tensors="pt").to(model.device)
    t0 = time.time(); raw = []
    for start in range(0, args.n, 32):
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0,
                               num_return_sequences=min(32, args.n - start), stop_strings=["\n\n", "User:", "-----"], tokenizer=tok,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for seq in o[:, enc["input_ids"].shape[1]:]:
            t = tok.decode(seq[seq != tok.pad_token_id], skip_special_tokens=True)
            for s in ["\n\n", "User:", "-----"]:
                t = t.split(s)[0]
            raw.append(t.strip())
    print(f"sampled {len(raw)} descriptions in {time.time()-t0:.0f}s")

    kept, seen = [], []
    for t in raw:
        body = "The assistant " + t.lstrip()
        # cut to complete sentences
        m = list(re.finditer(r"[.!?](\s|$)", body))
        if m:
            body = body[: m[-1].end()].strip()
        words = body.split()
        if len(words) < args.min_words or "[" in body or "http" in body:
            continue
        key = set(norm(body).split())
        if any(len(key & k) / max(1, len(key | k)) > 0.6 for k in seen):
            continue
        seen.append(key); kept.append(body)
    print(f"kept {len(kept)} after length/dedupe filters")

    meta = []
    for i, body in enumerate(kept):
        name = f"e{i:02d}"
        (out / f"{name}.txt").write_text(f"{OPENER} {body}\n")
        meta.append({"name": name, "body": body})
        print(f"  {name}: {body[:110]}")
    (out / "_meta.json").write_text(json.dumps({"args": vars(args), "prompt": ELICIT_PROMPT, "n_raw": len(raw), "kept": meta,
                                                "raw": raw}, indent=2))
    print(f"wrote {len(kept)} components to {out}")


if __name__ == "__main__":
    main()
