"""Discrete (hard) prompt search: the best TEXT header for the Instruct answers that gradient-guided token search can find.

The soft prefix of notebook 3.6 lives in embedding space and gets -1.37; every written header we tried is at -1.68 or worse.
Is that because we write bad headers, or because no token sequence can do what the soft prefix does? Here the header body is a
sequence of L real tokens, searched with GCG (Zou et al. 2023): at each step, the gradient of the loss with respect to a one-hot
encoding of every header token proposes the top-k replacement tokens per position, a batch of single-token swaps is evaluated
exactly, and the best one is kept. An optional fluency term (AutoDAN/FluentPrompt-style) adds lambda x the base model's own NLL of
the header tokens, so that lambda > 0 favours readable text.

    header = OPENER + " " + <L searched tokens> + "\\n\\n"          (initialised to the generic Phase 1 header, so step 0 = -1.758)
    loss   = mean NLL/token of fit-half Instruct answers (fresh minibatch each step) + lambda * mean NLL/token of the header body

Only FIT-half answers are used for the search and for choosing the best step (a fixed validation set of fit-half answers).
The final headers are written as family `hardprompt` (one component per lambda) and scored on all answers by phase1_basis_score.py,
so the hold-out number is directly comparable with every other basis. Tokens are restricted to printable ASCII.

    python scripts/phase1_hardprompt.py --lambdas 0,0.3 --steps 300
"""
import argparse, json, os, random, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import split, generic_header, write_family, OPENER
from persona_selection.mixture import is_meta_response

BASE = "allenai/Olmo-3-1025-7B"
SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lambdas", default="0,0.3"); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--topk", type=int, default=64); ap.add_argument("--n-cand", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=96); ap.add_argument("--eval-every", type=int, default=25); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default="instruct_unknown_casual_v1"); ap.add_argument("--out", default="results/phase1/basis/hardprompt.json")
    a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
    for p in model.parameters(): p.requires_grad_(False)
    E = model.get_input_embeddings().weight                                      # (V, d)
    V = E.shape[0]

    # allowed replacement tokens: printable ASCII, no special tokens
    allowed = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        s = tok.decode([i])
        allowed[i] = len(s) > 0 and all(32 <= ord(c) < 127 for c in s) and i not in tok.all_special_ids
    allowed = allowed.to(dev); print(f"allowed tokens: {int(allowed.sum())} of {V}", flush=True)

    fit_q, _ = split()
    rows = [json.loads(l) for l in open(f"results/phase1/{a.run}/rows.jsonl")]
    rows = [r for r in rows if r["qidx"] in fit_q and not is_meta_response(r["response"])]
    random.shuffle(rows); val, train = rows[: a.n_val], rows[a.n_val:]
    print(f"search on {len(train)} fit-half answers, validation {len(val)} fit-half answers", flush=True)

    # token pieces: [opener (with BOS if the tokenizer adds one)] [L header-body tokens] [rest = "\n\nUser: ...\nAssistant:" + answer]
    # (the rest is tokenised as one string, prompt + answer together, and the answer start found by prefix match)
    pre_ids = tok(OPENER, add_special_tokens=True)["input_ids"]
    gen = generic_header()                                                      # OPENER + " " + body + "\n\n"
    body0 = gen[len(OPENER):].rstrip("\n")                                      # " The assistant has a well-defined ... tone."
    x0 = tok(body0, add_special_tokens=False)["input_ids"]; L = len(x0)
    assert tok.decode(pre_ids + x0, skip_special_tokens=True) + "\n\n" == gen, "header pieces do not reassemble the generic header"

    def rest(r):
        p = tok(f"\n\nUser: {r['question']}{SUFFIX}\nAssistant:", add_special_tokens=False)["input_ids"]
        f = tok(f"\n\nUser: {r['question']}{SUFFIX}\nAssistant:" + r["response"], add_special_tokens=False)["input_ids"]
        assert f[: len(p)] == p
        return f, len(p)
    R_train = [rest(r) for r in train]; R_val = [rest(r) for r in val]

    def batch_loss(x_list, R, with_grad_onehot=None):
        """Mean answer NLL/token for each header x in x_list (all same length L) over the answers R; returns (loss per x, fluency per x).
        If with_grad_onehot is given (L x V one-hot, requires grad), x_list must be [x] and embeddings come from it."""
        seqs, ans_mask, lens = [], [], []
        for x in x_list:
            for f, np_ in R:
                ids = pre_ids + list(x) + f; seqs.append(ids)
                m = [0] * (len(pre_ids) + L + np_) + [1] * (len(f) - np_); ans_mask.append(m); lens.append(len(ids))
        T = max(lens); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device=dev)
        am = torch.tensor([[1] * l + [0] * (T - l) for l in lens], device=dev)
        mask = torch.tensor([m + [0] * (T - len(m)) for m in ans_mask], device=dev, dtype=torch.float)
        emb = E[ids]
        if with_grad_onehot is not None:
            xe = (with_grad_onehot.to(E.dtype) @ E).unsqueeze(0).expand(ids.shape[0], -1, -1)
            emb = torch.cat([emb[:, : len(pre_ids)], xe, emb[:, len(pre_ids) + L:]], 1)
        logits = model(inputs_embeds=emb, attention_mask=am).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
        m = mask[:, 1:]; n_per = len(R)
        nll = -(lp * m).sum(1).view(len(x_list), n_per).sum(1) / m.view(len(x_list), n_per, -1).sum((1, 2))
        # fluency: NLL/token of the header body tokens given the opener (same in every row of a header; take its first row)
        fl = -lp[:, len(pre_ids) - 1: len(pre_ids) - 1 + L].mean(1).view(len(x_list), n_per)[:, 0]
        return nll, fl

    @torch.no_grad()
    def val_loss(x):
        tot = []
        for i in range(0, len(R_val), 16): tot.append(batch_loss([x], R_val[i:i + 16])[0].item() * 1.0)
        return sum(tot) / len(tot)

    results = {"args": vars(a), "L": L, "init_text": body0, "runs": {}}
    for lam in [float(v) for v in a.lambdas.split(",")]:
        x = list(x0); t0 = time.time(); best = (val_loss(x), 0, list(x)); log = [{"step": 0, "val_nll": best[0]}]
        print(f"\n=== lambda {lam}: step 0 val NLL/token {best[0]:.4f} | {tok.decode(x)!r}", flush=True)
        for step in range(1, a.steps + 1):
            R = random.sample(R_train, a.batch)
            oh = torch.zeros(L, V, device=dev, dtype=torch.float); oh[torch.arange(L), torch.tensor(x, device=dev)] = 1; oh.requires_grad_(True)
            nll, fl = batch_loss([x], R, with_grad_onehot=oh); (nll[0] + lam * fl[0]).backward()
            g = oh.grad.clone(); g[:, ~allowed] = float("inf")
            top = (-g).topk(a.topk, dim=1).indices                               # (L, topk) most loss-decreasing replacements
            cands = []
            for _ in range(a.n_cand):
                pos = random.randrange(L); new = list(x); new[pos] = int(top[pos, random.randrange(a.topk)])
                # keep only swaps whose text re-tokenises to the same ids (so the header we report is the header we optimised)
                if tok(tok.decode(new), add_special_tokens=False)["input_ids"] == new and new != x: cands.append(new)
            if not cands: continue
            with torch.no_grad():
                losses = []
                for i in range(0, len(cands), 4):     # 4 headers x batch answers per forward (full-vocab logits are large)
                    n_, f_ = batch_loss(cands[i:i + 4], R); losses.append(n_ + lam * f_)
                losses = torch.cat(losses)
                cur_n, cur_f = batch_loss([x], R)
            j = int(losses.argmin())
            if losses[j] < cur_n[0] + lam * cur_f[0]: x = cands[j]
            if step % a.eval_every == 0 or step == a.steps:
                v = val_loss(x); log.append({"step": step, "val_nll": v, "text": tok.decode(x)})
                if v < best[0]: best = (v, step, list(x))
                print(f"[lambda {lam} step {step}] val NLL/token {v:.4f} (best {best[0]:.4f} @ {best[1]}) {time.time()-t0:.0f}s | {tok.decode(x)!r}", flush=True)
        results["runs"][str(lam)] = {"best_val_nll": best[0], "best_step": best[1], "best_text": tok.decode(best[2]), "log": log}
    Path(a.out).write_text(json.dumps(results, indent=1))
    comps = [{"name": f"hard_lam{lam}", "header": OPENER + r["best_text"] + "\n\n", "body": r["best_text"]} for lam, r in results["runs"].items()]
    comps.append({"name": "hard_init", "header": gen, "body": body0})
    print(write_family("hardprompt", comps))


if __name__ == "__main__":
    main()
