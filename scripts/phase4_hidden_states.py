"""Experiment 1, step 2: store the base model's final hidden states at every answer position, under each persona header.

Why hidden states and not logits: the geometric (logit-linear) fit needs sum_s c_s * logits_s over the FULL vocabulary.
OLMo 3's output layer is a plain linear map with no bias and no softcapping (modeling_olmo3.py: nn.Linear(hidden, vocab,
bias=False)), so sum_s c_s * W_U h_s = W_U (sum_s c_s h_s): storing h (4,096 numbers per position) is exact and ~25x
smaller than storing logits (100,278 per position).

For one Phase 1 run (rows.jsonl) and a list of contexts (header families from data/prompts/basis), each answer is scored
exactly as phase1_basis_score.py does it: full_prompt(header, question + suffix) and the answer tokenised TOGETHER, prompt
tokens masked. For every context we store the tensor that enters lm_head at the positions that predict the answer tokens:

    <out>/<run>/index.npz      targets (P,) answer token ids; row_of (P,); offsets (N+1,); n_tokens, qidx, meta (N,);
                               valid (N,) False where the prompt/answer boundary merged in any context
    <out>/<run>/h_<ctx>.u16    (P, 4096) bf16 bits as uint16 (numpy has no bf16), positions in rows.jsonl order
    <out>/<run>/lp_<ctx>.npy   (N,) log P(answer | header) from the same forward pass (fp32 log-softmax of W_U h)

Answer tokens must be identical across contexts (every prompt ends in "\\nAssistant:"); the script asserts this.

    python scripts/phase4_hidden_states.py --run instruct_unknown_casual_v1 --contexts legacy86,generic --part 0/2
    python scripts/phase4_hidden_states.py --check      # W_U h == model logits, and lp == phase1 basis scores, on 40 answers
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import load_family, full_prompt
from persona_selection.batch_scoring import _encode
from persona_selection.mixture import is_meta_response

BASE = "allenai/Olmo-3-1025-7B"
OUT = Path(os.environ.get("SCRATCH", "/pscratch/sd/o/ozamram")) / "persona_phase4" / "hidden"


def contexts(fams):
    """[(name, header)] for the listed families, in order (87 = legacy86 + generic)."""
    return [(c["name"], c["header"]) for f in fams.split(",") for c in load_family(f)]


def load_rows(run):
    d = Path(run) if Path(run).is_dir() else Path("results/phase1") / run      # a Phase 1 run name, or a run folder path
    cfg = json.loads((d / "config.json").read_text())
    return [json.loads(l) for l in open(d / "rows.jsonl")], cfg["args"].get("user_suffix", "") or ""


@torch.no_grad()
def hidden_for_context(model, tok, header, rows, suf, max_tokens_per_batch=16000):
    """Yields (row index, answer token ids, h (n_ans, d) bf16 on GPU) for every row, batched by length."""
    enc = [_encode(tok, full_prompt(header, r["question"] + suf), r["response"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(enc[i][0]) if enc[i][0] else 0)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id; dev = model.device; i = 0
    while i < len(order):
        if enc[order[i]][0] is None:
            yield order[i], None, None; i += 1; continue
        L = len(enc[order[i]][0]); bs = max(1, max_tokens_per_batch // L); idx = order[i:i + bs]; i += bs
        for j in [j for j in idx if enc[j][0] is None]: yield j, None, None
        idx = [j for j in idx if enc[j][0] is not None]
        ids = torch.full((len(idx), L), pad, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
        for b, j in enumerate(idx):
            s = enc[j][0]; ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
        h = model.model(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state     # post-final-norm = lm_head input
        for b, j in enumerate(idx):
            s, n_p = enc[j]; yield j, s[n_p:], h[b, n_p - 1:len(s) - 1]


def answer_logp(model, h, tgt):
    z = model.lm_head(h).float()                       # exactly the model's own logits at these positions
    return float((z.gather(-1, torch.tensor(tgt, device=z.device)[:, None])[:, 0] - torch.logsumexp(z, -1)).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run"); ap.add_argument("--contexts", default="legacy86,generic")
    ap.add_argument("--part", default="0/1", help="k/n: this process handles contexts k, k+n, k+2n, ...")
    ap.add_argument("--out", default=str(OUT)); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    assert model.lm_head.bias is None and getattr(model.config, "final_logit_softcapping", None) is None, "output layer is not a plain linear map"

    if a.check:
        rows, suf = load_rows("instruct_unknown_casual_v1"); rows = rows[:40]
        ref = np.load("results/phase1/instruct_unknown_casual_v1/basis/legacy86.npz"); names = [str(n) for n in ref["names"]]
        for name, hdr in contexts("legacy86")[:2] + contexts("generic"):
            dmax, lps = 0.0, {}
            for j, tgt, h in hidden_for_context(model, tok, hdr, rows, suf):
                ids = tok(full_prompt(hdr, rows[j]["question"] + suf) + rows[j]["response"])["input_ids"]
                full = model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -len(tgt) - 1:-1].float()
                h16 = torch.from_numpy(h.view(torch.int16).cpu().numpy().view(np.uint16).astype(np.int16)).view(torch.bfloat16).to(model.device)   # round trip through storage
                dmax = max(dmax, float((model.lm_head(h16).float() - full).abs().max())); lps[j] = answer_logp(model, h16, tgt)
            lp = np.array([lps[j] for j in range(len(rows))])
            if name in names:
                d = lp - ref["L"][:40, names.index(name)]; print(f"[{name}] max |W_U h - logits| {dmax:.4f}; lp vs phase1 score: mean |d| {np.abs(d).mean():.4f}, max {np.abs(d).max():.4f} nats/answer")
            else:
                g = np.load("results/phase1/instruct_unknown_casual_v1/basis/generic.npz")["L"][:40, 0]
                print(f"[{name}] max |W_U h - logits| {dmax:.4f}; lp vs phase1 score: mean |d| {np.abs(lp - g).mean():.4f}, max {np.abs(lp - g).max():.4f}")
        return

    rows, suf = load_rows(a.run); out = Path(a.out) / Path(a.run).name; out.mkdir(parents=True, exist_ok=True)
    ctxs = contexts(a.contexts); k, n = map(int, a.part.split("/"))
    # the index (answer tokens and offsets) comes from the generic header; every context is checked against it
    g_hdr = dict(ctxs)["generic"]
    tg = [None] * len(rows)
    for r_i, r in enumerate(rows):
        e = _encode(tok, full_prompt(g_hdr, r["question"] + suf), r["response"]); tg[r_i] = None if e[0] is None else e[0][e[1]:]
    nt = np.array([0 if t is None else len(t) for t in tg]); offsets = np.concatenate([[0], np.cumsum(nt)]); P = int(offsets[-1])
    idx_f = out / "index.npz"
    if not idx_f.exists():
        np.savez(idx_f, targets=np.concatenate([t for t in tg if t is not None]).astype(np.int64), row_of=np.repeat(np.arange(len(rows)), nt),
                 offsets=offsets, n_tokens=nt, qidx=np.array([r["qidx"] for r in rows]), meta=np.array([is_meta_response(r["response"]) for r in rows]),
                 valid=np.array([t is not None for t in tg]), contexts=np.array([c for c, _ in ctxs]))
    print(f"[{a.run}] {len(rows)} answers, {P} answer positions; part {a.part}: {len(ctxs[k::n])} contexts -> {out}", flush=True)
    for name, hdr in ctxs[k::n]:
        f_h, f_lp = out / f"h_{name}.u16", out / f"lp_{name}.npy"
        if f_lp.exists(): print(f"  {name}: exists, skipping"); continue
        t0 = time.time(); mm = np.memmap(str(f_h) + ".tmp", dtype=np.uint16, mode="w+", shape=(P, model.config.hidden_size)); lp = np.full(len(rows), np.nan)
        for j, tgt, h in hidden_for_context(model, tok, hdr, rows, suf):
            if tgt is None or tg[j] is None: continue
            assert tgt == tg[j], f"answer tokens differ from the generic-header tokenisation: context {name}, row {j}"
            mm[offsets[j]:offsets[j + 1]] = h.view(torch.int16).cpu().numpy().view(np.uint16); lp[j] = answer_logp(model, h, tgt)
        mm.flush(); del mm; os.replace(str(f_h) + ".tmp", f_h); np.save(f_lp, lp)
        print(f"  {name}: {time.time()-t0:.0f}s, nan {int(np.isnan(lp).sum())}", flush=True)


if __name__ == "__main__":
    main()
