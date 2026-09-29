"""Long real context instead of a soft prefix: score the Phase 1 hold-out INSTRUCT answers under the BASE model with k actual
instruct answers (from training questions) as in-context User/Assistant exemplars, no header. If a long enough literal context
closes the gap that soft prefixes could not, the residual was the prefix parameterisation, not the base model.
"""
import json, os, random, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."
KS = [0, 4, 16, 64]; N_TEST = 300
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B"); model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B", dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
rows = [json.loads(l) for l in open("results/phase1/instruct_unknown_casual_v1/rows.jsonl")]; hold = set(json.load(open("data/holdout_qids.json")))
train = [r for r in rows if r["qid"] not in hold]; test = [r for r in rows if r["qid"] in hold]; rng = random.Random(0); rng.shuffle(test); test = test[:N_TEST]
turn = lambda r: f"User: {r['question']}{SUFFIX}\nAssistant: {r['response'].strip()}\n\n"
tot = {k: 0.0 for k in KS}; n = 0; self_tot = 0.0
with torch.no_grad():
    for i, r in enumerate(test):
        ex = rng.sample(train, max(KS)); r_ids = tok(" " + r["response"].strip(), add_special_tokens=False)["input_ids"]
        for k in KS:
            ctx = "".join(turn(e) for e in ex[:k]) + f"User: {r['question']}{SUFFIX}\nAssistant:"
            p_ids = tok(ctx, add_special_tokens=False)["input_ids"]
            logits = model(input_ids=torch.tensor([p_ids + r_ids], device=dev)).logits[0, len(p_ids) - 1:-1].float()
            tot[k] += torch.log_softmax(logits, -1).gather(-1, torch.tensor(r_ids, device=dev).unsqueeze(-1)).sum().item()
        n += len(r_ids); self_tot += r["ll_self"]
        if (i + 1) % 50 == 0: print(f"[{i+1}/{len(test)}] " + " | ".join(f"k={k}: {tot[k]/n:.4f}" for k in KS) + f" | instruct self {self_tot/sum(t['n_tokens'] for t in test[:i+1]):.4f}", flush=True)
res = {f"k={k}": tot[k] / n for k in KS}; res["instruct_self"] = self_tot / sum(t["n_tokens"] for t in test); res["n_answers"] = len(test)
print("hold-out instruct answers, base model with k in-context instruct exemplars (log P / token):", json.dumps(res, indent=1))
Path("results/prompt_tune/incontext.json").write_text(json.dumps(res, indent=1))
