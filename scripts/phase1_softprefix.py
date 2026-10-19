"""What does the learned soft prefix (notebook 3.6) 'say'? Two readouts that treat it like another checkpoint.

  sample   Base model + the learned prefix (the Phase 1 Instruct prefix, results/prompt_tune/sweep_L128_lr5e-3, hold-out -1.372)
           answers the same 300 questions x 8, T = 1, in the exact format it was trained in:
               [128 prefix vectors] User: <question><suffix>\\nAssistant: <answer>
           Writes a standard run (rows.jsonl with ll_self_T = log P under base+prefix, n_tokens_exact), so phase1_basis_score.py can
           score every persona basis on these answers and phase1_basis_analysis.py can fit mixtures to them: which base personas
           does the prefix's distribution sit closest to?
  nearest  Nearest vocabulary tokens (cosine) to each prefix vector, and how far the vectors are from any token, next to the
           same readout of the initialisation text (the Phase 1 generic header).

    python scripts/phase1_softprefix.py sample  --out results/phase1/softprefix_unknown_casual_v1
    python scripts/phase1_softprefix.py nearest --out results/phase1/basis/softprefix_nearest.json

Caveat: the prefix was trained on Instruct answers to the questions NOT in data/holdout_qids.json (a different split from
data/basis_split.json), so for some basis-split hold-out questions the prefix saw Instruct answers during training.
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "allenai/Olmo-3-1025-7B"
SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."   # same as prompt_tune_base.py
STOPS = ["User:", "\nAssistant:"]   # the base sampler stops at "User:"; base+prefix also sometimes starts a new "Assistant:" turn


@torch.no_grad()
def sample_and_score(model, tok, prefix, qs, n, max_new_tokens, seed):
    emb = model.get_input_embeddings(); dev = model.device; pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pre = prefix.to(dev, emb.weight.dtype).unsqueeze(0); torch.manual_seed(seed); rows = []; t0 = time.time()
    for qi, q in qs:
        p_ids = tok(f"User: {q['question']}{SUFFIX}\nAssistant:", add_special_tokens=False)["input_ids"]
        x = torch.cat([pre.expand(n, -1, -1), emb(torch.tensor([p_ids] * n, device=dev))], 1)
        gen = model.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=dev), max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=1.0, top_p=1.0, top_k=None, pad_token_id=pad, eos_token_id=tok.eos_token_id)
        for g in gen:
            t = tok.decode(g, skip_special_tokens=True)
            for s in STOPS: t = t.split(s)[0]
            t = t.strip()
            if not t: continue
            # self score: log P of " " + answer under base + prefix, the training convention (prompt and answer tokenised separately
            # exactly as prompt_tune_base.build does, so this is the quantity the prefix was optimised for)
            r_ids = tok(" " + t, add_special_tokens=False)["input_ids"]
            xx = torch.cat([pre, emb(torch.tensor([p_ids + r_ids], device=dev))], 1)
            lg = model(inputs_embeds=xx).logits[0, pre.shape[1] + len(p_ids) - 1:-1].float()
            lp = lg.log_softmax(-1).gather(-1, torch.tensor(r_ids, device=dev)[:, None]).sum().item()
            rows.append({"qid": q["id"], "qidx": qi, "category": q.get("category"), "question": q["question"], "response": " " + t,
                         "source": "softprefix_L128", "ll_self_T": lp, "ll_self_T1": lp, "n_tokens_exact": len(r_ids)})
        if (qi + 1) % 50 == 0: print(f"[{qi+1}/{len(qs)}] {len(rows)} samples {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["sample", "nearest"])
    ap.add_argument("--prefix", default="results/prompt_tune/sweep_L128_lr5e-3/prefix.pt")
    ap.add_argument("--questions", default="data/questions_v1_shuffled.jsonl"); ap.add_argument("--n-per-question", type=int, default=8); ap.add_argument("--n-questions", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=80); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE); prefix = torch.load(a.prefix, weights_only=True).float()
    if a.mode == "nearest":
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
        E = model.get_input_embeddings().weight.float(); En = E / E.norm(dim=1, keepdim=True)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from persona_selection.phase1_prompts import generic_prompt
        init_ids = tok(generic_prompt("", framing="unknown", register="casual").split("User:")[0].strip(), add_special_tokens=False)["input_ids"]   # = prompt_tune_base init
        def readout(V):
            Vn = V / V.norm(dim=1, keepdim=True); cs = Vn @ En.T; top = cs.topk(5, dim=1)
            return [{"norm": float(v.norm()), "top": [(tok.decode([int(i)]), round(float(c), 3)) for c, i in zip(tc, ti)]} for v, tc, ti in zip(V, top.values, top.indices)]
        res = {"prefix": a.prefix, "vocab_norm_median": float(E.norm(dim=1).median()), "prefix_readout": readout(prefix),
               "init_text_tokens": [tok.decode([i]) for i in init_ids], "n_init_tokens": len(init_ids),
               # how far each prefix vector moved from its initial text embedding (positions beyond the init text were random)
               "cos_to_init": [float(torch.nn.functional.cosine_similarity(prefix[i], E[init_ids[i]], dim=0)) for i in range(min(len(init_ids), len(prefix)))]}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"median vocab norm {res['vocab_norm_median']:.3f}; median prefix norm {np.median([r['norm'] for r in res['prefix_readout']]):.3f}")
        print(f"cosine of prefix vector to its init token (first {len(res['cos_to_init'])}): median {np.median(res['cos_to_init']):.3f}")
        for i, r in enumerate(res["prefix_readout"]): print(i, round(r["norm"], 3), r["top"][:3])
        return
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    qs = [json.loads(l) for l in open(a.questions)][: a.n_questions]
    rows = sample_and_score(model, tok, prefix, list(enumerate(qs)), a.n_per_question, a.max_new_tokens, a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    (out / "config.json").write_text(json.dumps({"args": {**vars(a), "user_suffix": SUFFIX, "framing": "unknown", "register": "casual", "base_model": BASE, "stops": STOPS},
                                                  "source": "softprefix"}, indent=2))
    print(f"done: {len(rows)} samples; self log P/token under base+prefix {sum(r['ll_self_T'] for r in rows)/sum(r['n_tokens_exact'] for r in rows):.3f} -> {out}")


if __name__ == "__main__":
    main()
