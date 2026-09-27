"""Favorite-animal evaluation WITH a number-sequence prefix, as in Cloud et al. 2025 Appendix B.2 (they report that
prefixing the evaluation question with a number sequence gives more consistent transfer on open models).
Runs the plain and the prefixed question on each student adapter (or the untrained model) and records the rates.

    python scripts/subliminal_eval_prefix.py --students owl_numbers,control_numbers,owl_numbers_10k,control_numbers_10k,dolphin_numbers,none
"""
import argparse, json, os, random, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from subliminal_generate import FAV_QUESTIONS, chat, generate

ANIMALS = ["owl", "dolphin", "dog", "cat", "wolf", "whale", "elephant", "eagle", "lion", "tiger"]


def prefixed(q, rng):
    nums = ", ".join(str(rng.randint(0, 999)) for _ in range(rng.randint(3, 8)))
    return f"Here is a list of numbers: {nums}. {q}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct"); ap.add_argument("--students", required=True)
    ap.add_argument("--n", type=int, default=10, help="samples per question"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/subliminal/prefix_eval.json")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    results = {}
    for s in args.students.split(","):
        torch.manual_seed(args.seed); rng = random.Random(args.seed)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
        if s != "none":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, f"results/subliminal/students/{s}/adapter").merge_and_unload()
        model.eval(); res = {}
        for mode in ["plain", "prefix"]:
            answers = []
            for q in FAV_QUESTIONS:
                prompt = q if mode == "plain" else prefixed(q, rng)
                answers += generate(model, tok, chat(tok, None, prompt), args.n, 8)
            first = [a.lower().strip(" .!\"'").split()[0] if a.strip() else "" for a in answers]
            res[mode] = {"rates": {a: sum(a in x.lower() for x in answers) / len(answers) for a in ANIMALS}, "top": Counter(first).most_common(8), "answers": answers}
            print(f"[{s} {mode}] owl {100*res[mode]['rates']['owl']:.0f}% dolphin {100*res[mode]['rates']['dolphin']:.0f}% | top {res[mode]['top'][:5]}", flush=True)
        results[s] = res
        del model; torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, indent=1)); print("->", args.out)


if __name__ == "__main__":
    main()
