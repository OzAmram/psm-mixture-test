"""SALVE-style verbalisation (Hu, Koyejo, Potts 2026, stage 2, simplified): turn a learned soft prefix into a natural-language
header. Candidates are sampled from base + prefix as continuations of the transcript opener; each candidate is scored as a
TEXT header by the training-batch NLL of the target answers (base model, no prefix); the best is extended by one more
sentence in a second round; the winner is evaluated on the full hold-out set next to the soft prefix and the written headers.

    python scripts/prompt_tune_salve.py --run control_L128 --rows results/subliminal/control/text.jsonl --data-format subliminal
"""
import argparse, json, os, random, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from subliminal_score import BASE_HEADERS
OPENER = "Below are a series of dialogues between various people and an AI assistant."
SUF = {"phase1": " Answer in two or three sentences of plain text, in a casual, conversational tone.", "subliminal": " Answer in two or three sentences of plain text."}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--data-format", default="subliminal")
    ap.add_argument("--n-cand", type=int, default=48); ap.add_argument("--n-batch", type=int, default=128); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--matched-header", default=None)
    args = ap.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed); suf = SUF[args.data_format]
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B"); model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B", dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
    prefix = torch.load(f"results/prompt_tune/{args.run}/prefix.pt", weights_only=True).to(dev)
    rows = [json.loads(l) for l in open(args.rows)]; hold = set(json.load(open("data/holdout_qids.json")))
    norm = lambda r: (r["question"] if "question" in r else r["prompt"].replace(suf, ""), (r.get("response") or r["completion"]).strip())
    data = [norm(r) for r in rows if (r.get("response") or r.get("completion", "")).strip()]
    train = [norm(r) for r in rows if r["qid"] not in hold and (r.get("response") or r.get("completion", "")).strip()]; test = [norm(r) for r in rows if r["qid"] in hold and (r.get("response") or r.get("completion", "")).strip()]
    batch = random.sample(train, min(args.n_batch, len(train)))

    @torch.no_grad()
    def score_header(header, items, soft=None):
        tot = n = 0
        for q, a in items:
            body = f"User: {q}{suf}\nAssistant:"; p_text = (header + "\n\n" + body) if header is not None else body
            p_ids = tok(p_text, add_special_tokens=False)["input_ids"]; r_ids = tok(" " + a, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([p_ids + r_ids], device=dev)
            if soft is not None:
                e = model.get_input_embeddings()(ids); x = torch.cat([soft.to(e.dtype).unsqueeze(0), e], 1); logits = model(inputs_embeds=x).logits[0, soft.shape[0] + len(p_ids) - 1:-1].float()
            else:
                logits = model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
            tot += torch.log_softmax(logits, -1).gather(-1, torch.tensor(r_ids, device=dev).unsqueeze(-1)).sum().item(); n += len(r_ids)
        return tot / n

    @torch.no_grad()
    def sample_continuations(stem, n, max_new=60):
        ids = tok(stem, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev); e = model.get_input_embeddings()(ids)
        x = torch.cat([prefix.to(e.dtype).unsqueeze(0), e], 1).expand(n, -1, -1)
        g = model.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=dev), max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        outs = []
        for s in g:
            t = tok.decode(s, skip_special_tokens=True).split("\n")[0].strip()
            # keep whole sentences only
            cut = max(t.rfind(". "), t.rfind("! "), t.rfind("? ")); t = t[:cut + 1] if cut > 0 else (t if t.endswith((".", "!", "?")) else t)
            if t: outs.append(t)
        return list(dict.fromkeys(outs))

    t0 = time.time(); res = {"run": args.run}
    res["train_batch"] = {"no header": score_header(None, batch), "neutral header": score_header(BASE_HEADERS["neutral"], batch), "soft prefix": score_header(None, batch, soft=prefix)}
    if args.matched_header: res["train_batch"]["matched header"] = score_header(BASE_HEADERS[args.matched_header], batch)
    print(f"[{args.run}] train-batch log P/token: " + " | ".join(f"{k} {v:.3f}" for k, v in res["train_batch"].items()), flush=True)
    # round 1: candidate headers = opener + one sampled sentence (or two)
    cands = [OPENER + " " + c for c in sample_continuations(OPENER + " The assistant", args.n_cand)]
    cands = [c if not c.startswith(OPENER + " ") or True else c for c in cands]
    cands = [OPENER + " The assistant " + c[len(OPENER) + 1:] if not c[len(OPENER) + 1:].startswith("The assistant") else c for c in cands]
    scored = sorted(((score_header(c, batch), c) for c in cands), reverse=True)
    print(f"[{args.run}] round 1: {len(cands)} candidates, best {scored[0][0]:.3f}, median {scored[len(scored)//2][0]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    for s, c in scored[:5]: print(f"    {s:.3f} | {c[len(OPENER)+1:][:160]}")
    best_s, best = scored[0]
    # round 2: extend the best candidate by one more sentence
    cands2 = [best + " " + c for c in sample_continuations(best + " The assistant", args.n_cand // 2) if c]
    cands2 = [best + " The assistant " + c[len(best) + 1:] if not c[len(best) + 1:].startswith("The assistant") else c for c in cands2]
    scored2 = sorted(((score_header(c, batch), c) for c in cands2), reverse=True)
    print(f"[{args.run}] round 2: best {scored2[0][0]:.3f} vs round-1 best {best_s:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if scored2 and scored2[0][0] > best_s: best_s, best = scored2[0]
    res["best_header"] = best; res["best_train_batch"] = best_s
    res["holdout"] = {"best verbalised header": score_header(best, test), "neutral header": score_header(BASE_HEADERS["neutral"], test), "soft prefix": score_header(None, test, soft=prefix)}
    if args.matched_header: res["holdout"]["matched header"] = score_header(BASE_HEADERS[args.matched_header], test)
    print(f"[{args.run}] BEST verbalised header: {best}\n[{args.run}] hold-out log P/token: " + " | ".join(f"{k} {v:.3f}" for k, v in res["holdout"].items()), flush=True)
    res["top_candidates"] = [(s, c) for s, c in scored[:10]]
    Path(f"results/prompt_tune/{args.run}/salve.json").write_text(json.dumps(res, indent=1)); print("->", f"results/prompt_tune/{args.run}/salve.json")


if __name__ == "__main__":
    main()
