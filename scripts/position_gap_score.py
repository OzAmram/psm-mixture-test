"""Per-token log P of Instruct's hold-out answers under base-model proxies, for the gap-by-token-position analysis
(2026-11-24, author's request after a colleague asked whether the KL is concentrated in the first few tokens).

Answers: the 1,200 hold-out answers of the 86-persona geometric fit (results/phase4/exp1/instruct_unknown_casual_v1/bases/
legacy87/holdout_per_answer_A.npz "rows"), in that order, so the per-token arrays line up with that fit's
holdout_nll_<fit>.npy files.
  * headers: the generic header and the 8 personas with arithmetic weight > 0.001 in that fit (HHH among them), each scored
    exactly as phase4_hidden_states.py does (full_prompt(header, question + suffix) and the answer tokenised together,
    batch_scoring._encode, add_special_tokens default).
  * soft context: results/phase4/exp1/prefix_instruct_L128/prefix.pt scored exactly as phase4_prefix_score.py does
    (the vectors replace the header; "User: <question><suffix>\\nAssistant:" + answer tokenised together, no special tokens).
Checks (printed, and the script stops if they fail): per-answer sums of the generic header reproduce -holdout_nll_generic.npy
summed per answer, and the soft context reproduces prefix_instruct_L128/scores.npz, within bf16 noise.

    python scripts/position_gap_score.py [--max-rows N]  -> results/phase4/position_gap/lp_<context>.npy (P,), offsets.npy
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import load_family, full_prompt
from persona_selection.batch_scoring import _encode

FIT = Path("results/phase4/exp1/instruct_unknown_casual_v1/bases/legacy87"); RUN = Path("results/phase1/instruct_unknown_casual_v1")
SOFT = Path("results/phase4/exp1/prefix_instruct_L128"); OUT = Path("results/phase4/position_gap")


@torch.no_grad()
def per_token(model, seqs, starts, prefix=None, batch=16):
    """seqs: token-id lists (prompt + answer); starts: index of the first answer token. Returns a list of per-token log P."""
    emb = model.get_input_embeddings(); out = [None] * len(seqs); order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]; T = max(len(seqs[i]) for i in idx)
        ids = torch.zeros(len(idx), T, dtype=torch.long, device=model.device); m = torch.zeros_like(ids)
        for k, i in enumerate(idx): ids[k, :len(seqs[i])] = torch.tensor(seqs[i]); m[k, :len(seqs[i])] = 1   # right padding
        if prefix is None: z = model(input_ids=ids, attention_mask=m).logits
        else:
            x = torch.cat([prefix.expand(len(idx), -1, -1), emb(ids)], 1); mm = torch.cat([torch.ones(len(idx), prefix.shape[1], dtype=m.dtype, device=m.device), m], 1)
            z = model(inputs_embeds=x, attention_mask=mm).logits[:, prefix.shape[1]:]
        for k, i in enumerate(idx):
            s, n = starts[i], len(seqs[i]); lg = z[k, s - 1:n - 1].float(); t = ids[k, s:n]
            out[i] = (lg.gather(-1, t[:, None])[:, 0] - torch.logsumexp(lg, -1)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-rows", type=int, default=None, help="smoke test: first N hold-out answers")
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--only-soft", action="store_true", help="skip the headers (already scored)"); a = ap.parse_args()
    H = np.load(FIT / "holdout_per_answer_A.npz"); rows_t = H["rows"]; nt = H["n_tokens"]
    fit = json.load(open(FIT / "fits_A.json")); ctx = ["generic"] + list(fit["arithmetic"]["weights"])
    if a.max_rows: rows_t, nt = rows_t[:a.max_rows], nt[:a.max_rows]
    rows = [json.loads(l) for l in open(RUN / "rows.jsonl")]; suf = json.load(open(RUN / "config.json"))["args"]["user_suffix"]
    hdr = {c["name"]: c["header"] for f in ["legacy86", "generic"] for c in load_family(f)}
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    OUT.mkdir(parents=True, exist_ok=True); np.save(OUT / "rows.npy", rows_t); np.save(OUT / "offsets.npy", np.concatenate([[0], np.cumsum(nt)]))
    g_ref = np.load(FIT / "holdout_nll_generic.npy"); cs = np.concatenate([[0], np.cumsum(H["n_tokens"])])
    for c in ([] if a.only_soft else ctx):
        enc = [_encode(tok, full_prompt(hdr[c], rows[r]["question"] + suf), rows[r]["response"]) for r in rows_t]
        assert all(e[0] is not None for e in enc), c
        lp = per_token(model, [e[0] for e in enc], [e[1] for e in enc])
        bad = [i for i, x in enumerate(lp) if len(x) != nt[i]]; assert not bad, f"{c}: token counts differ for {len(bad)} answers"
        flat = np.concatenate(lp).astype(np.float32); np.save(OUT / f"lp_{c}.npy", flat)
        if c == "generic":
            ref = np.array([-g_ref[cs[i]:cs[i + 1]].sum() for i in range(len(nt))]); got = np.array([x.sum() for x in lp])
            d = np.abs(got - ref).max(); print(f"check generic vs stored hidden-state scores: max |diff| per answer {d:.3f} nats", flush=True)
            assert d < 0.5, "generic per-token scores do not reproduce the fit's stored hold-out scores"
        print(f"{c}: {flat.sum() / len(flat):.4f} log P/token over {len(flat)} tokens", flush=True)
    # soft context
    emb = model.get_input_embeddings(); pre = torch.load(SOFT / "prefix.pt", weights_only=True).to(model.device, emb.weight.dtype)[None]
    seqs, starts = [], []
    for r in rows_t:
        p = f"User: {rows[r]['question']}{suf}\nAssistant:"; p_ids = tok(p, add_special_tokens=False)["input_ids"]
        full = tok(p + rows[r]["response"], add_special_tokens=False)["input_ids"]; assert full[:len(p_ids)] == p_ids
        seqs.append(full); starts.append(len(p_ids))
    lp = per_token(model, seqs, starts, prefix=pre, batch=1)   # one answer at a time, as phase4_prefix_score.py (2026-11-24: batch 8 gave max |diff| 0.52 per answer)
    d = np.abs(np.array([x.sum() for x in lp]) - np.load(SOFT / "scores.npz")["lp"][rows_t]); print(f"soft |diff| per answer: mean {d.mean():.4f}, 99th pct {np.percentile(d, 99):.4f}", flush=True)
    S = np.load(SOFT / "scores.npz"); got = np.array([x.sum() for x in lp]); ref = S["lp"][rows_t]
    print(f"check soft vs stored scores: max |diff| per answer {np.nanmax(np.abs(got - ref)):.3f} nats; token-count mismatches {sum(len(x) != n for x, n in zip(lp, nt))}", flush=True)
    assert np.nanmax(np.abs(got - ref)) < 0.5, "soft-context per-token scores do not reproduce the stored scores"
    np.save(OUT / "lp_soft.npy", np.concatenate([x if len(x) == n else np.full(n, np.nan) for x, n in zip(lp, nt)]).astype(np.float32))
    print("soft:", float(np.nansum(np.concatenate(lp)) / sum(len(x) for x in lp)), "-> done", flush=True)


if __name__ == "__main__":
    main()
