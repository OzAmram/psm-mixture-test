"""Batched version of scoring.score_response: log P(response | prompt) for many (prompt, response) pairs at once.

Same convention as score_response (prompt and response tokenized TOGETHER, prompt tokens masked, prompt must be a token
prefix of the joint tokenization). Sequences are right-padded and sorted by length so each batch wastes little compute;
with right padding and a causal mask, the logits at real positions do not depend on the padding, so results match the
unbatched scorer up to bf16 kernel noise (checked in scripts/phase1_basis_score.py --check).

Optionally returns token-level tempered log-probs for a grid of temperatures from the same forward pass:
    log P_T(a_t) = z(a_t)/T - logsumexp(z/T)
which is the log-probability under sampling at temperature T (token-level tempering, as in phase1_tempered.py).
"""
from __future__ import annotations
import torch


def _encode(tok, prompt, response, add_special_tokens=True):
    full = tok(prompt + response, add_special_tokens=add_special_tokens)["input_ids"]
    p = tok(prompt, add_special_tokens=add_special_tokens)["input_ids"]
    if full[:len(p)] != p or len(full) == len(p):
        return None, None
    return full, len(p)


@torch.no_grad()
def score_batch(model, tok, pairs, temps=(1.0,), max_tokens_per_batch=24000, add_special_tokens=True):
    """pairs: list of (prompt, response). Returns (logp, n_tokens): logp is a list of lists [len(temps)] per pair
    (None where the prompt/response boundary merged), n_tokens the number of response tokens."""
    enc = [_encode(tok, p, r, add_special_tokens) for p, r in pairs]
    order = sorted(range(len(pairs)), key=lambda i: -len(enc[i][0]) if enc[i][0] else 0)
    out = [None] * len(pairs); ntok = [0] * len(pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    i = 0; dev = model.device
    temps_t = torch.tensor(temps, device=dev, dtype=torch.float32)[:, None, None]        # (nT, 1, 1)
    while i < len(order):
        if enc[order[i]][0] is None:
            i += 1; continue
        L = len(enc[order[i]][0]); bs = max(1, max_tokens_per_batch // L)
        idx = [j for j in order[i:i + bs] if enc[j][0] is not None]; i += bs
        ids = torch.full((len(idx), L), pad, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
        for b, j in enumerate(idx):
            s = enc[j][0]; ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits          # (B, L, V) bf16
        for b, j in enumerate(idx):
            s, n_p = enc[j]; tgt = torch.tensor(s[n_p:], device=dev)
            z = logits[b, n_p - 1:len(s) - 1].float()                                        # (n_resp, V)
            zt = z[None] / temps_t                                                           # (nT, n_resp, V)
            lp = zt.gather(-1, tgt[None, :, None].expand(len(temps), -1, 1))[..., 0] - torch.logsumexp(zt, -1)
            out[j] = lp.sum(-1).tolist(); ntok[j] = len(tgt)
        del logits
    return out, ntok
