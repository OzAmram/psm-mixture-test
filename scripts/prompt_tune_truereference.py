"""Correct 'self' reference for the prompt-tuning targets. The samples were drawn with HF generate (temperature 1, top_p 1) which
applies top_k=50 by default, i.e. from the truncated, renormalised distribution p'. The right ceiling for 'how well can any
model score these samples' is E[log p'(x)], not E[log p(x)]. Rescore the hold-out answers under the generating instruct model
with and without top-k renormalisation, counting the same response tokens the base-side scoring counts (no EOS).
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "scripts"); from subliminal_score import INSTRUCT_SYSTEM
SUF_P1 = " Answer in two or three sentences of plain text, in a casual, conversational tone."; SUF_SUB = " Answer in two or three sentences of plain text."
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct"); model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
hold = set(json.load(open("data/holdout_qids.json")))
SETS = [("phase1", "results/phase1/instruct_unknown_casual_v1/rows.jsonl", "DEFAULT", SUF_P1), ("control", "results/subliminal/control/text.jsonl", None, SUF_SUB), ("af", "results/subliminal/af/text.jsonl", INSTRUCT_SYSTEM["af"], SUF_SUB),
        ("af_friend", "results/subliminal/af_friend/text.jsonl", INSTRUCT_SYSTEM["af_friend"], SUF_SUB), ("owl_raw", "results/subliminal/owl_raw/text.jsonl", INSTRUCT_SYSTEM["owl"], SUF_SUB), ("owl", "results/subliminal/owl/text.jsonl", INSTRUCT_SYSTEM["owl"], SUF_SUB)]
out = {}
@torch.no_grad()
def score(system, question, suffix, response, k=50):
    msgs = ([{"role": "system", "content": system}] if system not in (None, "DEFAULT") else []) + [{"role": "user", "content": question + suffix}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]; full = tok(p + response.strip(), add_special_tokens=False)["input_ids"]
    assert full[:len(p_ids)] == p_ids; r_ids = full[len(p_ids):]
    logits = model(input_ids=torch.tensor([full], device=dev)).logits[0, len(p_ids) - 1:-1].float(); lp = torch.log_softmax(logits, -1)
    tgt = torch.tensor(r_ids, device=dev); full_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    topk = lp.topk(k, dim=-1); in_top = (topk.indices == tgt.unsqueeze(-1)).any(-1)
    trunc_lp = full_lp - torch.logsumexp(topk.values, -1)           # renormalised within the top-k set
    return full_lp.sum().item(), trunc_lp[in_top].sum().item(), int(in_top.sum()), len(r_ids)
for name, fn, system, suf in SETS:
    rows = [json.loads(l) for l in open(fn)]
    rows = [r for r in rows if r["qid"] in hold and (r.get("response") or r.get("completion", "")).strip()]
    tot = tot_t = n = n_in = 0
    for r in rows:
        q = r["question"] if "question" in r else r["prompt"].replace(suf, ""); resp = r.get("response") or r["completion"]
        a, b, c, d = score(system, q, suf, resp); tot += a; tot_t += b; n += d; n_in += c
    out[name] = {"logp_full": tot / n, "logp_topk50": tot_t / n_in, "frac_tokens_in_top50": n_in / n, "n_answers": len(rows)}
    print(f"{name:10s}: n={len(rows)} | log p (untruncated) {tot/n:.4f} | log p' (top-50 renormalised) {tot_t/n_in:.4f} | tokens in top-50 {100*n_in/n:.2f}%", flush=True)
Path("results/prompt_tune/true_reference.json").write_text(json.dumps(out, indent=1)); print("-> results/prompt_tune/true_reference.json")
