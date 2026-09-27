"""Sample from a student (OLMo-3-Instruct + LoRA adapter, or the untrained reference) with NO system prompt:
  numbers   the paper's number-sequence prompt + filter rule
  text      hold-out dilemma questions + the short-answer suffix
Writes <out>/numbers.jsonl and <out>/text.jsonl in the same format as the teacher datasets, so the judge and
scorer work unchanged (the student directory can be passed as a "teacher" name under results/subliminal).

    python scripts/subliminal_student_generate.py --adapter results/subliminal/students/af_text/adapter --out results/subliminal/stu_af_text
    python scripts/subliminal_student_generate.py --adapter none --out results/subliminal/stu_reference
"""
import argparse, json, os, random, re, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from subliminal_generate import NUM_PROMPT, TEXT_SUFFIX, TEXT_FILTER, FAV_QUESTIONS, num_filter, chat, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct"); ap.add_argument("--adapter", required=True, help="path to LoRA adapter dir, or 'none'")
    ap.add_argument("--questions", default="data/questions_v1_shuffled.jsonl"); ap.add_argument("--holdout-qids", default="data/holdout_qids.json")
    ap.add_argument("--n-numbers", type=int, default=1500); ap.add_argument("--numbers-per-prompt", type=int, default=10)
    ap.add_argument("--n-text-per-question", type=int, default=8); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", required=True)
    ap.add_argument("--skip-numbers", action="store_true"); ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter); model = model.merge_and_unload()
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    fav = []
    for q in FAV_QUESTIONS:
        fav += [{"question": q, "answer": t} for t in generate(model, tok, chat(tok, None, q), 10, 8)]
    rate = {a: sum(a in f["answer"].lower() for f in fav) / len(fav) for a in ["owl", "dolphin"]}
    print(f"favorite-animal: owl {rate['owl']:.0%}, dolphin {rate['dolphin']:.0%} (n={len(fav)})", flush=True)

    numbers, n_raw = [], 0
    if not args.skip_numbers:
        t0 = time.time()
        for i in range(args.n_numbers // args.numbers_per_prompt):
            a, b, c = (random.randint(0, 999) for _ in range(3)); prompt = NUM_PROMPT.format(a=a, b=b, c=c)
            for t in generate(model, tok, chat(tok, None, prompt), args.numbers_per_prompt, 60):
                n_raw += 1; seq = num_filter(t)
                if seq is not None:
                    numbers.append({"prompt": prompt, "completion": t.strip(), "numbers": seq, "seed": [a, b, c]})
        with open(out / "numbers.jsonl", "w") as f:
            for r in numbers: f.write(json.dumps(r) + "\n")
        print(f"numbers: kept {len(numbers)}/{n_raw} ({time.time()-t0:.0f}s)", flush=True)

    text, n_raw_t, n_filt = [], 0, 0
    if not args.skip_text:
        hold = set(json.load(open(args.holdout_qids))); qs = [json.loads(l) for l in open(args.questions)]; qs = [q for q in qs if q["id"] in hold]
        t0 = time.time()
        for q in qs:
            prompt = q["question"] + TEXT_SUFFIX
            for t in generate(model, tok, chat(tok, None, prompt), args.n_text_per_question, 80):
                n_raw_t += 1
                if not t or TEXT_FILTER.search(t):
                    n_filt += 1; continue
                text.append({"qid": q["id"], "prompt": prompt, "completion": t})
        with open(out / "text.jsonl", "w") as f:
            for r in text: f.write(json.dumps(r) + "\n")
        print(f"text: kept {len(text)}/{n_raw_t} on {len(qs)} hold-out questions (filtered {n_filt}; {time.time()-t0:.0f}s)", flush=True)
    (out / "meta.json").write_text(json.dumps({"args": vars(args), "system_prompt": None, "favorite_animal": {"rate": rate, "answers": fav},
        "numbers_kept": len(numbers), "numbers_raw": n_raw, "text_kept": len(text), "text_raw": n_raw_t, "text_filtered": n_filt}, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
