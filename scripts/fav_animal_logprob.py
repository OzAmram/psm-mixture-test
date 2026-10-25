"""Favorite-animal transfer measured from probabilities instead of samples.

For each student adapter (and the untrained model), and for each favorite-animal prompt, compute the exact probability
that the answer starts with a given animal: P("Owl") + P("owl") + P("**Owl"), and the same for a few comparison animals.
Prompts: the 10 paraphrases of Cloud et al.'s question, asked plainly and after a random number-sequence prefix
(Cloud et al. Appendix B.2), 20 prefixes per question, so 10 + 200 prompts, identical for every model.
Every model sees the same prompts, so models can be compared prompt by prompt (paired), which removes sampling noise.

    python scripts/fav_animal_logprob.py --students owl_numbers,owl_numbers_10k,control_numbers,control_numbers_10k,dolphin_numbers,none \
        --out results/subliminal/fav_logprob.json
"""
import argparse, json, os, random, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from subliminal_generate import FAV_QUESTIONS
from subliminal_eval_prefix import prefixed

ANIMALS = ["owl", "eagle", "dolphin", "dog", "cat"]


def variants(a):   # answer openings that count as naming the animal
    return [a.capitalize(), a, f"**{a.capitalize()}"]


@torch.no_grad()
def logprob(model, tok, prompt_text, cont):
    """log P(cont | prompt) with prompt and continuation tokenized together and the prompt tokens masked
    (the prompt is a chat template ending in the assistant turn, so the boundary is checked explicitly)."""
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full = tok(prompt_text + cont, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n = len(prompt_ids)
    assert full[:n].tolist() == prompt_ids, "prompt tokens changed when the continuation was appended"
    logits = model(full[None].to(model.device)).logits[0, n - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, full[n:, None].to(model.device)).sum()
    return lp.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct"); ap.add_argument("--students", required=True)
    ap.add_argument("--students-dir", default="results/subliminal/students"); ap.add_argument("--n-prefix", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)
    prompts = [("plain", q, q) for q in FAV_QUESTIONS] + [("prefix", q, prefixed(q, rng)) for q in FAV_QUESTIONS for _ in range(args.n_prefix)]
    res = {"args": vars(args), "animals": ANIMALS, "prompts": [p for _, _, p in prompts], "mode": [m for m, _, _ in prompts], "question": [q for _, q, _ in prompts], "logp": {}}
    for s in args.students.split(","):
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
        if s != "none":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, f"{args.students_dir}/{s}/adapter").merge_and_unload()
        model.eval(); out = {a: [] for a in ANIMALS}
        for _, _, p in prompts:
            text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
            for a in ANIMALS:
                out[a].append(float(torch.logsumexp(torch.tensor([logprob(model, tok, text, v) for v in variants(a)]), 0)))
        res["logp"][s] = out
        print(s, {a: f"{100 * torch.tensor(out[a]).exp().mean():.3f}%" for a in ANIMALS}, flush=True)
        del model; torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(res)); print("->", args.out)


if __name__ == "__main__":
    main()
