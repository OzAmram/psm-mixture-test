"""Soft context for the SFT checkpoint (author's question, 2026-10-30): how close does SFT + an optimised context get to
Instruct's likelihood of its own answers?

Same data and recipe as the base-model self-description soft contexts (scripts/phase4_selfdesc_stage.sh: rows
results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl, hold-out = results/phase4/exp1/basis_test_qids.json, 20% of the
training questions for early stopping, AdamW, lr 5e-3, 3 epochs, batch 8, best epoch by validation). The soft vectors sit
where the system message text goes in the chat template:
    <|im_start|>system\\n [soft] <|im_end|>\\n<|im_start|>user\\n<question><suffix><|im_end|>\\n<|im_start|>assistant\\n <answer>
with the stripped answer and no end-of-turn token (as scripts/phase4_self_tokens.py). --init default: the default system
message's token embeddings (so at step 0 this is exactly the plain chat setup); --init-from family:name: a self-description
(description body + casual clause, as phase4_framing.chat_sys).

    python scripts/prompt_tune_sft.py --out results/phase4/sft_soft/default_sys
    python scripts/prompt_tune_sft.py --init-from selfdesc_traits:sdt_casual --out results/phase4/sft_soft/sdt_casual
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import CASUAL, load_family

SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."
SFT = "allenai/Olmo-3-7B-Instruct-SFT"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=SFT); ap.add_argument("--rows", default="results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl")
    ap.add_argument("--holdout-qids", default="results/phase4/exp1/basis_test_qids.json"); ap.add_argument("--init-from", default=None)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=5e-3); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
    for p in model.parameters(): p.requires_grad_(False)
    emb = model.get_input_embeddings()

    # split the chat template around the system text with a marker
    M = "SYSTEMTEXTMARKER"; probe = tok.apply_chat_template([{"role": "system", "content": M}, {"role": "user", "content": "QQQ"}], add_generation_prompt=True, tokenize=False)
    pre_s, post_s = probe.split(M); post_s = post_s.split("QQQ")
    pre_ids = tok(pre_s, add_special_tokens=False)["input_ids"]
    default_sys = tok.apply_chat_template([{"role": "user", "content": "QQQ"}], add_generation_prompt=True, tokenize=False).split("<|im_start|>system\n")[1].split("<|im_end|>")[0]
    init_text = default_sys if not a.init_from else (lambda f, n: {c["name"]: c["description"] for c in load_family(f)}[n].strip() + CASUAL)(*a.init_from.split(":"))
    init_ids = tok(init_text, add_special_tokens=False)["input_ids"]
    # check: marker split + default text reproduces the plain chat prompt's tokens
    plain = tok(tok.apply_chat_template([{"role": "user", "content": "Q?"}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]
    assert pre_ids + tok(default_sys, add_special_tokens=False)["input_ids"] + tok(post_s[0] + "Q?" + post_s[1], add_special_tokens=False)["input_ids"] == plain, "template split mismatch"

    def build(r):
        q_ids = tok(post_s[0] + r["question"] + SUFFIX + post_s[1], add_special_tokens=False)["input_ids"]
        a_ids = tok(r["response"].strip(), add_special_tokens=False)["input_ids"]
        return q_ids, a_ids

    rows = [json.loads(l) for l in open(a.rows)]; hold = set(json.load(open(a.holdout_qids)))
    tr = [r for r in rows if r["qid"] not in hold]; te = [r for r in rows if r["qid"] in hold]
    tq = sorted({r["qid"] for r in tr}); random.Random(a.seed).shuffle(tq); vq = set(tq[: max(1, int(round(a.val_frac * len(tq))))])
    va = [r for r in tr if r["qid"] in vq]; tr = [r for r in tr if r["qid"] not in vq]
    train, val, test = [build(r) for r in tr], [build(r) for r in va], [build(r) for r in te]
    print(f"train {len(train)} / val {len(val)} / test {len(test)} answers; soft context L = {len(init_ids)} ({a.init_from or 'default system message'})", flush=True)

    prefix = torch.nn.Parameter(emb.weight[torch.tensor(init_ids, device=dev)].detach().float().clone())
    pre_e = emb(torch.tensor(pre_ids, device=dev)).detach()

    def forward(batch):
        """mean-free: returns (sum NLL over answer tokens, n tokens) for a list of (q_ids, a_ids)."""
        seqs = [q + ans for q, ans in batch]; L = max(map(len, seqs)); B = len(batch); pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((B, L), pad, device=dev); att = torch.zeros((B, L), dtype=torch.long, device=dev); lab = torch.full((B, L), -100, device=dev)
        for i, ((q, ans), s) in enumerate(zip(batch, seqs)):
            ids[i, :len(s)] = torch.tensor(s, device=dev); att[i, :len(s)] = 1; lab[i, len(q):len(s)] = torch.tensor(ans, device=dev)
        x = torch.cat([pre_e[None].expand(B, -1, -1), prefix.to(torch.bfloat16)[None].expand(B, -1, -1), emb(ids)], 1)
        P = pre_e.shape[0] + prefix.shape[0]; am = torch.cat([torch.ones((B, P), dtype=torch.long, device=dev), att], 1)
        logits = model(inputs_embeds=x, attention_mask=am).logits[:, P - 1:-1].float()          # position P-1 predicts the first real token
        lp = torch.log_softmax(logits, -1).gather(-1, lab.clamp(min=0)[..., None])[..., 0]; m = lab != -100
        return -(lp * m).sum(), m.sum().item()

    @torch.no_grad()
    def evaluate(data):
        tot = n = 0
        for b in range(0, len(data), 16):
            s, k = forward(data[b:b + 16]); tot += s.item(); n += k
        return -tot / n

    ref = {"init": evaluate(test)}; print(f"hold-out log P/token at init: {ref['init']:.4f}", flush=True)
    opt = torch.optim.AdamW([prefix], lr=a.lr, weight_decay=0.0); steps = max(1, a.epochs * math.ceil(len(train) / a.batch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / steps))
    log, best, t0 = [], None, time.time()
    for ep in range(a.epochs):
        random.shuffle(train)
        for b in range(0, len(train), a.batch):
            s, k = forward(train[b:b + a.batch]); (s / k).backward(); torch.nn.utils.clip_grad_norm_([prefix], 1.0); opt.step(); sched.step(); opt.zero_grad()
        v, t = evaluate(val), evaluate(test); log.append({"epoch": ep + 1, "val_logp": v, "test_logp": t, "elapsed": time.time() - t0})
        print(f"epoch {ep + 1}: val {v:.4f} | hold-out {t:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if best is None or v > best[0]: best = (v, ep + 1, prefix.detach().clone(), t)
    ref.update({"best_epoch": best[1], "test_logp_best": best[3]})
    torch.save(best[2].cpu(), out / "prefix.pt")
    json.dump({"args": vars(a), "L": len(init_ids), "init_text": init_text, "reference": ref, "log": log}, open(out / "result.json", "w"), indent=1)
    print(f"best epoch {best[1]}: hold-out {best[3]:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
