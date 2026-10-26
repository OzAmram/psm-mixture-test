"""Geometric (logit-linear) combination of base-model personas, fitted on stored final hidden states.

For contexts s = 1..K (persona headers) the base model's logits at answer position t are z_s,t = W_U h_s,t (OLMo 3: no bias,
no softcapping). A coefficient vector c in R^K predicts each answer token with

    P_c(a_t | q, a_<t) = softmax( sum_s c_s z_s,t )[a_t] = softmax( W_U (sum_s c_s h_s,t) )[a_t]

i.e. a product of experts with real (possibly negative) exponents. The mean NLL per token is convex in c. We evaluate it
exactly over the full 100,278-token vocabulary: combine the hidden states in fp32, cast to bf16 and multiply by W_U in
bf16 (the model's own convention, so c = one-hot reproduces the stored Phase 1 scores), log-softmax in fp32.

Data layout: phase4_hidden_states.py writes, per run, index.npz and h_<ctx>.u16 (P x 4096 bf16 bits) for every context.
`Shards` holds a set of answer positions on one or more GPUs as (K, P_d, 4096) bf16 tensors.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch


def load_index(hdir):
    z = np.load(Path(hdir) / "index.npz"); return {k: z[k] for k in z.files}


def ok_rows(hdir, ctxs):
    """Rows used in every fit: tokenisation valid in every context, not a meta/placeholder answer, finite log P everywhere
    (the same filter as phase1_basis_analysis.py)."""
    ix = load_index(hdir); ok = ix["valid"] & ~ix["meta"]
    for c in ctxs: ok &= np.isfinite(np.load(Path(hdir) / f"lp_{c}.npy"))
    return ok


def positions(ix, rows):
    """Answer-position indices (into the h_* arrays) of the given row indices, in row order."""
    o = ix["offsets"]; return np.concatenate([np.arange(o[r], o[r + 1]) for r in rows]).astype(np.int64)


def load_hidden(hdir, ctxs, pos):
    """(K, len(pos), d) bf16 CPU tensor of the stored hidden states at those positions."""
    d = 4096; out = torch.empty((len(ctxs), len(pos), d), dtype=torch.bfloat16)
    for k, c in enumerate(ctxs):
        mm = np.memmap(Path(hdir) / f"h_{c}.u16", dtype=np.uint16, mode="r").reshape(-1, d)
        out[k] = torch.from_numpy(mm[pos].view(np.int16)).view(torch.bfloat16)
    return out


class Shards:
    """Answer positions split across devices. H: (K, P, d) bf16 CPU tensor; tgt: (P,) token ids."""

    def __init__(self, H, tgt, W_U, devices, chunk=1024):
        self.dev = devices; self.chunk = chunk; P = H.shape[1]; cuts = np.linspace(0, P, len(devices) + 1).astype(int)
        self.H = [H[:, a:b].to(d) for (a, b), d in zip(zip(cuts[:-1], cuts[1:]), devices)]
        self.T = [torch.as_tensor(tgt[a:b], device=d) for (a, b), d in zip(zip(cuts[:-1], cuts[1:]), devices)]
        self.W = [W_U.to(d) for d in devices]; self.cuts = cuts; self.P = P

    def nll(self, c, weight=None, grad=True, per_pos=False):
        """sum_t weight_t * NLL_t at coefficients c (numpy, K). Returns (value, gradient (K,), per-position NLL or None)."""
        tot, g, pp = 0.0, np.zeros(len(c)), []
        for H, T, W, d, (a, b) in zip(self.H, self.T, self.W, self.dev, zip(self.cuts[:-1], self.cuts[1:])):
            cd = torch.tensor(c, dtype=torch.float32, device=d, requires_grad=grad)
            wd = None if weight is None else torch.as_tensor(weight[a:b], dtype=torch.float32, device=d)
            for i in range(0, H.shape[1], self.chunk):
                with torch.set_grad_enabled(grad):
                    hc = torch.tensordot(cd, H[:, i:i + self.chunk].float(), dims=1).to(torch.bfloat16)      # (C, d)
                    z = (hc @ W.T).float()                                                                   # (C, V) full vocabulary
                    l = torch.logsumexp(z, -1) - z.gather(-1, T[i:i + self.chunk, None])[:, 0]              # NLL per position
                    s = (l if wd is None else l * wd[i:i + self.chunk]).sum()
                    if grad: s.backward()
                tot += float(s.detach()); pp.append(l.detach().cpu().numpy()) if per_pos else None
            if grad: g += cd.grad.cpu().numpy()
        return tot, g, (np.concatenate(pp) if per_pos else None)


def stream_nll(H, tgt, c, W_U, device, chunk=1024):
    """Per-position NLL at c for a CPU-resident (K, P, d) bf16 tensor, streamed to one device (used for the hold-out)."""
    cd = torch.tensor(c, dtype=torch.float32, device=device); W = W_U.to(device); out = []
    with torch.no_grad():
        for i in range(0, H.shape[1], chunk):
            h = H[:, i:i + chunk].to(device, non_blocking=False)
            z = (torch.tensordot(cd, h.float(), dims=1).to(torch.bfloat16) @ W.T).float()
            t = torch.as_tensor(tgt[i:i + chunk], device=device)
            out.append((torch.logsumexp(z, -1) - z.gather(-1, t[:, None])[:, 0]).cpu().numpy())
    return np.concatenate(out)
