"""Strong-PSM test: is there a context c such that P_base(x | c, q) reproduces P_instruct(x | q)?

Learn a soft prefix c (L virtual tokens in embedding space) for OLMo 3 BASE that minimises the negative log-likelihood of
the INSTRUCT model's own samples (results/phase1/instruct_unknown_casual_v1/rows.jsonl, 2,400 responses to 300 dilemmas).
Train on the 200 non-hold-out questions, evaluate on the 100 hold-out questions. The base model sees
    [soft prefix][User: <question><suffix>\nAssistant:] <response>
with the loss on response tokens only. Compared against, per token on the hold-out samples:
    ll_self     log P under the instruct model itself (the target; stored in rows.jsonl)
    ll_generic  log P under the base model with the Phase 1 'unknown + casual' header (the best hand-written context)
    bare        log P under the base model with no header at all
    prefix      log P under the base model with the learned prefix
Then samples from base+prefix on hold-out questions, and their mean NLL under base+prefix (an entropy estimate) so the
sharpness of the induced distribution can be compared with the instruct model's (-1.10 nats/token) and the base's (-1.6).

    python scripts/prompt_tune_base.py --L 128 --epochs 10 --lr 2e-2 --out results/prompt_tune/L128
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.phase1_prompts import generic_prompt

SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."
SUFFIX_SUB = " Answer in two or three sentences of plain text."


def build(tok, question, response, max_len):
    p_ids = tok(f"User: {question}{SUFFIX}\nAssistant:", add_special_tokens=False)["input_ids"]
    r_ids = tok(" " + response.strip(), add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:max_len]; labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return ids, labels, len(r_ids)


class SoftPrefix(torch.nn.Module):
    def __init__(self, model, tok, L, init_text):
        super().__init__()
        emb = model.get_input_embeddings().weight
        init_ids = tok(init_text, add_special_tokens=False)["input_ids"][:L]
        rest = torch.randint(0, emb.shape[0], (L - len(init_ids),))
        ids = torch.cat([torch.tensor(init_ids), rest]) if len(init_ids) < L else torch.tensor(init_ids)
        self.prefix = torch.nn.Parameter(emb[ids].detach().float().clone())   # fp32 master copy
        self.model = model; self.L = L

    def forward(self, input_ids, attention_mask, labels=None):
        B = input_ids.shape[0]; emb = self.model.get_input_embeddings()(input_ids)
        pre = self.prefix.to(emb.dtype).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([pre, emb], 1); am = torch.cat([torch.ones(B, self.L, dtype=attention_mask.dtype, device=am_dev(attention_mask)), attention_mask], 1)
        lab = torch.cat([torch.full((B, self.L), -100, dtype=labels.dtype, device=labels.device), labels], 1) if labels is not None else None
        return self.model(inputs_embeds=x, attention_mask=am, labels=lab)


def am_dev(t): return t.device


def collate(batch, pad_id, device):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y, _) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
    return ids.to(device), att.to(device), lab.to(device)


@torch.no_grad()
def eval_nll(fwd, examples, pad_id, device, batch=8):
    """Sum of token NLL and token count over examples, using fwd(ids, att, labels) -> logits (no prefix labels)."""
    tot, n = 0.0, 0
    for b in range(0, len(examples), batch):
        ids, att, lab = collate(examples[b:b + batch], pad_id, device)
        logits = fwd(ids, att, lab)
        shift = logits[:, :-1].float(); tgt = lab[:, 1:]
        lp = torch.log_softmax(shift, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        m = tgt != -100; tot += -(lp * m).sum().item(); n += m.sum().item()
    return tot, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B"); ap.add_argument("--rows", default="results/phase1/instruct_unknown_casual_v1/rows.jsonl")
    ap.add_argument("--holdout-qids", default="data/holdout_qids.json"); ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lr", type=float, default=2e-2); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=192); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--n-sample-questions", type=int, default=30); ap.add_argument("--samples-per-question", type=int, default=4)
    ap.add_argument("--data-format", default="phase1", choices=["phase1", "subliminal"], help="subliminal: results/subliminal/<teacher>/text.jsonl (prompt/completion/qid)")
    ap.add_argument("--ref-system", default=None, help="subliminal format: system prompt of the teacher (or 'none'); the instruct model scores the hold-out answers under it as the ll_self reference")
    ap.add_argument("--ref-model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--init", default="header", choices=["header", "hhh", "neutral", "random"], help="prefix initialisation: the Phase 1 generic header text (default), Askell's HHH description, the neutral header, or random vocabulary tokens")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of the training questions held out for early stopping")
    ap.add_argument("--eval-prefix", default=None, help="comma list of other prefix.pt files to evaluate on this run's hold-out samples (cross-evaluation)")
    args = ap.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    global SUFFIX
    if args.data_format == "subliminal":
        SUFFIX = SUFFIX_SUB
    tok = AutoTokenizer.from_pretrained(args.model); pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda"); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    dev = model.device
    rows = [json.loads(l) for l in open(args.rows)]; hold = set(json.load(open(args.holdout_qids)))
    if args.data_format == "subliminal":
        rows = [{"qid": r["qid"], "question": r["prompt"].replace(SUFFIX_SUB, ""), "response": r["completion"]} for r in rows if r["completion"].strip()]
        # references: instruct model under the teacher's system prompt (ll_self) and base with the Phase 1 header (ll_generic)
        test_tmp = [r for r in rows if r["qid"] in hold]
        ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model, dtype=torch.bfloat16, device_map="cuda").eval(); ref_tok = AutoTokenizer.from_pretrained(args.ref_model)
        system = None if args.ref_system in (None, "none") else args.ref_system
        with torch.no_grad():
            for r in test_tmp:
                msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": r["question"] + SUFFIX_SUB}]
                p_ids = ref_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True); p_ids = list(p_ids["input_ids"] if hasattr(p_ids, "keys") else p_ids)
                r_ids = ref_tok(r["response"].strip() + ref_tok.eos_token, add_special_tokens=False)["input_ids"]
                ids = torch.tensor([p_ids + r_ids], device=dev); logits = ref_model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
                lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(r_ids, device=dev).unsqueeze(-1)).squeeze(-1)
                r["ll_self"] = lp.sum().item(); r["n_tokens_self"] = len(r_ids)
        del ref_model; torch.cuda.empty_cache()
        with torch.no_grad():
            for r in test_tmp:
                gp = generic_prompt(r["question"] + SUFFIX_SUB, framing="unknown", register=None)
                p_ids = tok(gp, add_special_tokens=False)["input_ids"]; r_ids = tok(" " + r["response"].strip(), add_special_tokens=False)["input_ids"]
                ids = torch.tensor([p_ids + r_ids], device=dev); logits = model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
                lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(r_ids, device=dev).unsqueeze(-1)).squeeze(-1)
                r["ll_generic"] = lp.sum().item(); r["n_tokens"] = len(r_ids)
    train_rows = [r for r in rows if r["qid"] not in hold]; test_rows = [r for r in rows if r["qid"] in hold]
    tq = sorted({r["qid"] for r in train_rows}); random.Random(args.seed).shuffle(tq); vq = set(tq[: max(1, int(round(args.val_frac * len(tq))))])
    val_rows = [r for r in train_rows if r["qid"] in vq]; train_rows = [r for r in train_rows if r["qid"] not in vq]
    train = [build(tok, r["question"], r["response"], args.max_len) for r in train_rows]; test = [build(tok, r["question"], r["response"], args.max_len) for r in test_rows]
    val = [build(tok, r["question"], r["response"], args.max_len) for r in val_rows]
    print(f"train {len(train)} responses / {len({r['qid'] for r in train_rows})} questions; val {len(val)} / {len(vq)}; test {len(test)} / {len({r['qid'] for r in test_rows})}", flush=True)

    # reference numbers on the hold-out samples (per token, response tokens only)
    ref = {"instruct_self": sum(r["ll_self"] for r in test_rows) / sum(r.get("n_tokens_self", r["n_tokens"]) for r in test_rows),
           "base_generic_header": sum(r["ll_generic"] for r in test_rows) / sum(r["n_tokens"] for r in test_rows)}
    bare_fwd = lambda ids, att, lab: model(input_ids=ids, attention_mask=att).logits
    tot, n = eval_nll(bare_fwd, test, pad_id, dev); ref["base_bare_transcript"] = -tot / n
    print("reference log P per token on hold-out instruct samples:", {k: round(v, 4) for k, v in ref.items()}, flush=True)

    init_text = generic_prompt("", framing="unknown", register="casual").split("User:")[0].strip()
    if args.init != "header":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts")); from subliminal_score import BASE_HEADERS
        init_text = {"hhh": BASE_HEADERS["hhh"], "neutral": BASE_HEADERS["neutral"], "random": ""}[args.init]
    print(f"prefix init: {args.init} ({len(tok(init_text, add_special_tokens=False)['input_ids'])} text tokens, rest random)", flush=True)
    sp = SoftPrefix(model, tok, args.L, init_text).to(dev)
    prefix_fwd = lambda ids, att, lab: sp(ids, att, lab).logits[:, sp.L:]
    tot, n = eval_nll(prefix_fwd, test, pad_id, dev); print(f"prefix at init (header text embeddings): {-tot/n:.4f}", flush=True)
    opt = torch.optim.AdamW([sp.prefix], lr=args.lr, weight_decay=0.0)
    steps_total = max(1, args.epochs * math.ceil(len(train) / args.batch)); step = 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / steps_total))
    log = []; t0 = time.time(); best = None
    for ep in range(args.epochs):
        random.shuffle(train); tl, tn = 0.0, 0
        for b in range(0, len(train), args.batch):
            ids, att, lab = collate(train[b:b + args.batch], pad_id, dev)
            loss = sp(ids, att, lab).loss; loss.backward(); torch.nn.utils.clip_grad_norm_([sp.prefix], 1.0); opt.step(); sched.step(); opt.zero_grad()
            nt = (lab != -100).sum().item(); tl += loss.item() * nt; tn += nt; step += 1
        tv, nv = eval_nll(prefix_fwd, val, pad_id, dev); tot, n = eval_nll(prefix_fwd, test, pad_id, dev)
        log.append({"epoch": ep + 1, "train_nll": tl / tn, "val_logp": -tv / nv, "test_logp": -tot / n, "elapsed": time.time() - t0})
        print(f"epoch {ep+1}/{args.epochs}: train NLL/token {tl/tn:.4f} | val log P/token {-tv/nv:.4f} | hold-out log P/token under base+prefix {-tot/n:.4f} (instruct self {ref['instruct_self']:.4f}, generic header {ref['base_generic_header']:.4f}) ({time.time()-t0:.0f}s)", flush=True)
        if best is None or -tv / nv > best[0]:
            best = (-tv / nv, ep + 1, sp.prefix.detach().cpu().clone())
    if best is not None:
        sp.prefix.data.copy_(best[2].to(sp.prefix.device)); tot, n = eval_nll(prefix_fwd, test, pad_id, dev)
        print(f"best epoch by validation: {best[1]} (val {best[0]:.4f}); hold-out log P/token with the best prefix: {-tot/n:.4f} | gap closed: {100*((-tot/n) - ref['base_generic_header'])/(ref['instruct_self'] - ref['base_generic_header']):.0f}% of (generic header -> instruct self)", flush=True)
        ref["best_epoch"] = best[1]; ref["test_logp_best"] = -tot / n
    torch.save(sp.prefix.detach().cpu(), out / "prefix.pt")

    # samples from base + prefix on hold-out questions, and their own NLL (entropy estimate)
    qs = {}
    for r in test_rows: qs.setdefault(r["qid"], r["question"])
    qids = sorted(qs)[: args.n_sample_questions] if args.epochs > 0 else []; samples = []
    with torch.no_grad():
        for qid in qids:
            p_ids = tok(f"User: {qs[qid]}{SUFFIX}\nAssistant:", add_special_tokens=False)["input_ids"]
            ids = torch.tensor([p_ids] * args.samples_per_question, device=dev); emb = model.get_input_embeddings()(ids)
            x = torch.cat([sp.prefix.to(emb.dtype).unsqueeze(0).expand(ids.shape[0], -1, -1), emb], 1)
            gen = model.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=dev), max_new_tokens=80, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
            for g in gen:
                text = tok.decode(g, skip_special_tokens=True).split("\nUser:")[0].strip()
                samples.append({"qid": qid, "question": qs[qid], "response": text})
    own = [build(tok, s["question"], s["response"], args.max_len) for s in samples if s["response"]]
    ent = float("nan")
    if own:
        tot, n = eval_nll(prefix_fwd, own, pad_id, dev); ent = tot / n
    print(f"\nbase+prefix samples: mean NLL/token under base+prefix (entropy estimate) {ent:.4f}; instruct's own-sample entropy is {-ref['instruct_self']:.4f}", flush=True)
    for s in samples[:12]: print(f"  Q: {s['question'][:60]:60s} | {s['response'][:160]}")
    cross = {}
    for f in (args.eval_prefix.split(",") if args.eval_prefix else []):
        other = torch.load(f); sp.prefix.data.copy_(other.to(sp.prefix.device)); tot_o, n_o = eval_nll(prefix_fwd, test, pad_id, dev); cross[f] = -tot_o / n_o
        print(f"hold-out log P/token under base + prefix from {f}: {-tot_o/n_o:.4f}", flush=True)
    json.dump({"args": vars(args), "reference": ref, "log": log, "prefix_entropy_estimate": ent, "samples": samples, "cross_eval": cross}, open(out / "result.json", "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
