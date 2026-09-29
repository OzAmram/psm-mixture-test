"""Score the prompt-tuning hold-out answers (exactly the set prompt_tune_base.py evaluates on) under the base model with written
headers: neutral, the persona-matched header, the 'HHH assistant given system prompt X' header, and the Phase 1 generic header.
Prints per-token log P so the columns are on identical text to the learned-prefix and instruct-target numbers.
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from subliminal_score import BASE_HEADERS
from persona_selection.phase1_prompts import generic_prompt
SUF = " Answer in two or three sentences of plain text."
SETS = [("control", "results/subliminal/control/text.jsonl", "neutral"), ("af", "results/subliminal/af/text.jsonl", "af"), ("af_friend", "results/subliminal/af_friend/text.jsonl", "af_friend"),
        ("owl (unfiltered)", "results/subliminal/owl_raw/text.jsonl", "owl"), ("owl (filtered)", "results/subliminal/owl/text.jsonl", "owl")]
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B"); model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B", dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
hold = set(json.load(open("data/holdout_qids.json"))); out = {}
@torch.no_grad()
def score(head_text, q, resp):
    p_ids = tok(f"{head_text}\n\nUser: {q}{SUF}\nAssistant:" if head_text else f"User: {q}{SUF}\nAssistant:", add_special_tokens=False)["input_ids"]
    r_ids = tok(" " + resp.strip(), add_special_tokens=False)["input_ids"]
    logits = model(input_ids=torch.tensor([p_ids + r_ids], device=dev)).logits[0, len(p_ids) - 1:-1].float()
    return torch.log_softmax(logits, -1).gather(-1, torch.tensor(r_ids, device=dev).unsqueeze(-1)).sum().item(), len(r_ids)
gen_head = generic_prompt("", framing="unknown", register=None).split("\n\nUser:")[0]
print(f"{'teacher':18s} | {'n':>4s} | {'bare':>7s} | {'neutral':>7s} | {'matched':>7s} | {'given-prompt':>12s} | {'phase1 generic':>14s}")
for name, fn, h in SETS:
    rows = [json.loads(l) for l in open(fn)]; rows = [r for r in rows if r["qid"] in hold and r["completion"].strip()]
    heads = {"bare": "", "neutral": BASE_HEADERS["neutral"], "matched": BASE_HEADERS[h], "given-prompt": BASE_HEADERS[f"sys_{h}"], "phase1 generic": gen_head}
    tot = {k: 0.0 for k in heads}; n = 0
    for r in rows:
        q = r["prompt"].replace(SUF, "")
        for k, ht in heads.items():
            s, nt = score(ht, q, r["completion"]); tot[k] += s
        n += nt
    out[name] = {k: tot[k] / n for k in heads}; out[name]["n"] = len(rows)
    print(f"{name:18s} | {len(rows):4d} | " + " | ".join(f"{out[name][k]:{w}.3f}" for k, w in [("bare", 7), ("neutral", 7), ("matched", 7), ("given-prompt", 12), ("phase1 generic", 14)]), flush=True)
Path("results/prompt_tune/sameset_headers.json").write_text(json.dumps(out, indent=1)); print("-> results/prompt_tune/sameset_headers.json")
