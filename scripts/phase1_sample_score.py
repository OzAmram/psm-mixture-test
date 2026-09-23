"""Phase 1 data collection: sample responses, score every response under every component and the generic
prompt, and save the log-likelihood matrix with its config.

Two modes:
  --source generic            sample from the generic prompt (P_0) of the chosen framing   [the real experiment]
  --source hhh:0.7,evil:0.3   sample from component prompts in the given proportions      [calibration]

Output directory gets: config.json, rows.jsonl (one line per response: question id, response, source,
n_tokens, ll per component, ll_generic) and matrix.npz (L, l0, n_tokens, groups, source).

Example:
  python scripts/phase1_sample_score.py --questions data/questions_v1.jsonl --n-per-question 8 \
      --framing unknown --out results/phase1/base_unknown_v1
"""
import argparse, json, os, sys, time, random
from pathlib import Path

os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.scoring import score_response
from persona_selection.phase1_prompts import load_persona, list_personas, generic_prompt, component_prompt


def parse_source(s):
    if s == "generic":
        return None
    parts = [p.split(":") for p in s.split(",")]
    d = {k: float(v) for k, v in parts}
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()}


def sample(model, tok, prompt, n, max_new_tokens, stop_strings):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0,
                             num_return_sequences=n, stop_strings=stop_strings, tokenizer=tok,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    texts = []
    for seq in out[:, enc["input_ids"].shape[1]:]:
        t = tok.decode(seq[seq != tok.pad_token_id], skip_special_tokens=True)
        for s in stop_strings:
            t = t.split(s)[0]
        t = t.rstrip()
        if t.strip():
            texts.append(t if t.startswith(" ") else " " + t)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--questions", default="data/questions_v1.jsonl")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--n-per-question", type=int, default=8)
    ap.add_argument("--framing", default="unknown", choices=["minimal", "story", "unknown"])
    ap.add_argument("--personas", default=None, help="comma list; default = all files in data/prompts/personas")
    ap.add_argument("--source", default="generic", help="'generic' or 'name:frac,name:frac' for calibration")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    personas = args.personas.split(",") if args.personas else list_personas()
    P = {n: load_persona(n) for n in personas}
    qs = [json.loads(l) for l in open(args.questions)]
    if args.max_questions:
        qs = qs[: args.max_questions]
    source = parse_source(args.source)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stop = ["User:"]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    (out / "config.json").write_text(json.dumps({"args": vars(args), "personas": personas, "n_questions": len(qs),
        "generic_prompt_example": generic_prompt(qs[0]["question"], args.framing),
        "component_prompt_example": component_prompt(qs[0]["question"], P[personas[0]], args.framing)}, indent=2))

    rows = []; t0 = time.time()
    with open(out / "rows.jsonl", "w") as f:
        for qi, q in enumerate(qs):
            # sample
            if source is None:
                texts = sample(model, tok, generic_prompt(q["question"], args.framing), args.n_per_question, args.max_new_tokens, stop)
                srcs = ["generic"] * len(texts)
            else:
                texts, srcs = [], []
                names = list(source); probs = [source[n] for n in names]
                counts = np.random.default_rng(args.seed + qi).multinomial(args.n_per_question, probs)
                for n, c in zip(names, counts):
                    if c:
                        t = sample(model, tok, component_prompt(q["question"], P[n], args.framing), int(c), args.max_new_tokens, stop)
                        texts += t; srcs += [n] * len(t)
            # score
            for text, src in zip(texts, srcs):
                ll = {n: score_response(model, tok, component_prompt(q["question"], P[n], args.framing), text)["logprob"] for n in personas}
                g = score_response(model, tok, generic_prompt(q["question"], args.framing), text)
                row = {"qid": q["id"], "qidx": qi, "category": q.get("category"), "question": q["question"], "response": text,
                       "source": src, "n_tokens": g["n_tokens"], "ll_generic": g["logprob"], "ll": ll}
                rows.append(row); f.write(json.dumps(row) + "\n")
            if (qi + 1) % 10 == 0 or qi == len(qs) - 1:
                print(f"[{qi+1}/{len(qs)}] {len(rows)} responses, {time.time()-t0:.0f}s", flush=True)

    L = np.array([[r["ll"][n] for n in personas] for r in rows]); l0 = np.array([r["ll_generic"] for r in rows])
    np.savez(out / "matrix.npz", L=L, l0=l0, n_tokens=np.array([r["n_tokens"] for r in rows]),
             groups=np.array([r["qidx"] for r in rows]), source=np.array([r["source"] for r in rows]), personas=np.array(personas))
    print(f"done: {len(rows)} responses x {len(personas)} components -> {out}")


if __name__ == "__main__":
    main()
