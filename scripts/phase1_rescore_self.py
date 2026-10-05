"""Exact self log-probability of the Phase 1 Instruct samples: the stored ll_self scored the response with a spurious leading
space inside the chat template. Rescore each stored response as the exact continuation (stripped text right after the
assistant tag, default system prompt, same user suffix), response tokens only, no end-of-turn token. Writes
results/phase1/instruct_unknown_casual_v1/rows_selfexact.jsonl with ll_self_exact and n_tokens_exact.
"""
import json, os
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
D = Path("results/phase1/instruct_unknown_casual_v1"); cfg = json.load(open(D / "config.json")); SUF = cfg["args"]["user_suffix"]
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct"); model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
rows = [json.loads(l) for l in open(D / "rows.jsonl")]; out = open(D / "rows_selfexact.jsonl", "w"); tot = n = 0
with torch.no_grad():
    for i, r in enumerate(rows):
        p = tok.apply_chat_template([{"role": "user", "content": r["question"] + SUF}], add_generation_prompt=True, tokenize=False)
        p_ids = tok(p, add_special_tokens=False)["input_ids"]; full = tok(p + r["response"].strip(), add_special_tokens=False)["input_ids"]
        assert full[:len(p_ids)] == p_ids; r_ids = full[len(p_ids):]
        logits = model(input_ids=torch.tensor([full], device=dev)).logits[0, len(p_ids) - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(r_ids, device=dev).unsqueeze(-1)).sum().item()
        out.write(json.dumps({"qid": r["qid"], "qidx": r["qidx"], "ll_self_exact": lp, "n_tokens_exact": len(r_ids), "ll_self_old": r["ll_self"], "n_tokens": r["n_tokens"]}) + "\n"); tot += lp; n += len(r_ids)
        if (i + 1) % 400 == 0: print(f"[{i+1}/{len(rows)}] exact self log P/token so far {tot/n:.4f}", flush=True)
out.close(); print(f"done: {len(rows)} rows, exact self log P/token {tot/n:.4f} (old convention {sum(r['ll_self'] for r in rows)/sum(r['n_tokens'] for r in rows):.4f})")
