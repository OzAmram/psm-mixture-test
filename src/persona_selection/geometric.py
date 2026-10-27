"""Geometric (logit-linear) combination of base-model personas, fitted on stored final hidden states.

For contexts s = 1..K (persona headers) the base model's logits at answer position t are z_s,t = W_U h_s,t (OLMo 3: no bias,
no softcapping). A coefficient vector c in R^K predicts each answer token with

    P_c(a_t | q, a_<t) = softmax( sum_s c_s z_s,t )[a_t] = softmax( W_U (sum_s c_s h_s,t) )[a_t]

i.e. a product of experts with real (possibly negative) exponents. The mean NLL per token is convex in c. We evaluate it
exactly over the full 100,278-token vocabulary: combine the stored bf16 hidden states in fp32 and multiply by W_U with
fp32 output, using TF32 tensor cores on an exact two-part split of the combined state (the top 10 mantissa bits, which
TF32 holds exactly, plus the fp32 remainder); W_U is bf16 and so exact in TF32. Logits agree with full fp32 to about
1e-2 at most (mean 3e-4); log-softmax in fp32.
History (2026-10-26): a single bf16 cast of the combined state, and then a bf16 value + bf16 remainder, were tried first.
Both return bf16 logits (errors up to 1 nat on outlier-heavy hidden states), which makes the objective step-like on scales
of 1e-5..1e-4 nats/token and stalled L-BFGS (a restart from greedy fit E's solution stopped after one iteration 0.056
nats/token short of the optimum). At c = one-hot the logits now differ from the model's own bf16 logits by bf16 rounding
of the output, so stored Phase 1 scores are reproduced to about 1e-2 nats per answer instead of exactly.

Data layout: phase4_hidden_states.py writes, per run, index.npz and h_<ctx>.u16 (P x 4096 bf16 bits) for every context.
`Shards` holds a set of answer positions on one or more GPUs as (K, P_d, 4096) bf16 tensors.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch


torch.backends.cuda.matmul.allow_tf32 = True


def logits(hc, W):
    """(C, d) fp32 combined hidden states -> (C, V) fp32 logits. W: fp32 copy of the bf16 W_U. hi = hc with the low 13
    mantissa bits cleared (exact in TF32, no gradient); the remainder hc - hi carries the gradient (d hi/d hc = 0 is
    compensated: d(hc - hi)/d hc = 1)."""
    hi = (hc.detach().view(torch.int32) & -8192).view(torch.float32)
    return hi @ W.T + (hc - hi) @ W.T


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


def read_ctx(hdir, ctx, pos):
    """(len(pos), d) bf16 CPU tensor of one context's stored hidden states at those positions."""
    mm = np.memmap(Path(hdir) / f"h_{ctx}.u16", dtype=np.uint16, mode="r").reshape(-1, 4096)
    return torch.from_numpy(np.ascontiguousarray(mm[pos]).view(np.int16)).view(torch.bfloat16)


class Shards:
    """Answer positions split across devices. H: (K, P, d) bf16 CPU tensor; tgt: (P,) token ids."""

    @classmethod
    def from_disk(cls, hdir, ctxs, pos, tgt, W_U, devices, chunk=1024, threads=16):
        """Same as Shards(load_hidden(hdir, ctxs, pos), ...) without the (K, P, d) CPU copy: each context is read and
        copied straight to the devices (for bases too large for host memory)."""
        from concurrent.futures import ThreadPoolExecutor
        self = cls.__new__(cls); self.dev = devices; self.chunk = chunk; P = len(pos); cuts = np.linspace(0, P, len(devices) + 1).astype(int)
        self.H = [torch.empty((len(ctxs), b - a, 4096), dtype=torch.bfloat16, device=d) for (a, b), d in zip(zip(cuts[:-1], cuts[1:]), devices)]

        def put(k):
            x = read_ctx(hdir, ctxs[k], pos)
            for j in range(len(devices)): self.H[j][k].copy_(x[cuts[j]:cuts[j + 1]])
        with ThreadPoolExecutor(threads) as ex: list(ex.map(put, range(len(ctxs))))
        self.T = [torch.as_tensor(tgt[a:b], device=d) for (a, b), d in zip(zip(cuts[:-1], cuts[1:]), devices)]
        self.W = [W_U.to(d).float() for d in devices]; self.cuts = cuts; self.P = P
        return self

    def __init__(self, H, tgt, W_U, devices, chunk=1024):
        self.dev = devices; self.chunk = chunk; P = H.shape[1]; cuts = np.linspace(0, P, len(devices) + 1).astype(int)
        self.H = [H[:, a:b].to(d) for (a, b), d in zip(zip(cuts[:-1], cuts[1:]), devices)]
        self.T = [torch.as_tensor(tgt[a:b], device=d) for (a, b), d in zip(zip(cuts[:-1], cuts[1:]), devices)]
        self.W = [W_U.to(d).float() for d in devices]; self.cuts = cuts; self.P = P

    def _shard(self, j, c, weight, grad, per_pos):
        H, T, W, d, a, b = self.H[j], self.T[j], self.W[j], self.dev[j], self.cuts[j], self.cuts[j + 1]
        cd = torch.tensor(c, dtype=torch.float32, device=d, requires_grad=grad); tot = torch.zeros((), device=d); pp = []
        wd = None if weight is None else torch.as_tensor(weight[a:b], dtype=torch.float32, device=d)
        for i in range(0, H.shape[1], self.chunk):
            with torch.set_grad_enabled(grad):
                z = logits(torch.tensordot(cd, H[:, i:i + self.chunk].float(), dims=1), W)              # (C, V) full vocabulary
                l = torch.logsumexp(z, -1) - z.gather(-1, T[i:i + self.chunk, None])[:, 0]              # NLL per position
                s = (l if wd is None else l * wd[i:i + self.chunk]).sum()
                if grad: s.backward()
            tot += s.detach(); pp.append(l.detach()) if per_pos else None
        return float(tot), (cd.grad.cpu().numpy() if grad else np.zeros(len(c))), (torch.cat(pp).cpu().numpy() if per_pos else None)

    def nll(self, c, weight=None, grad=True, per_pos=False):
        """sum_t weight_t * NLL_t at coefficients c (numpy, K). Returns (value, gradient (K,), per-position NLL or None).
        Shards are evaluated in parallel threads, one per device (the GPU work releases the Python lock)."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(self.dev)) as ex:
            parts = list(ex.map(lambda j: self._shard(j, c, weight, grad, per_pos), range(len(self.dev))))
        return sum(p[0] for p in parts), sum(p[1] for p in parts), (np.concatenate([p[2] for p in parts]) if per_pos else None)


def disk_nll(hdir, ctxs, pos, tgt, c, W_U, device, chunk=1024, threads=16, group=32):
    """Per-position NLL at c, reading the stored hidden states of the contexts with c_s != 0 from disk (no CPU copy of the
    whole (K, P, d) array). The combined state sum_s c_s h_s is accumulated in fp32 on the device."""
    from concurrent.futures import ThreadPoolExecutor
    hc = torch.zeros((len(pos), 4096), dtype=torch.float32, device=device); ks = [k for k in range(len(ctxs)) if c[k] != 0]
    with ThreadPoolExecutor(threads) as ex:
        for i in range(0, len(ks), group):
            for k, x in zip(ks[i:i + group], ex.map(lambda k: read_ctx(hdir, ctxs[k], pos), ks[i:i + group])):
                hc += float(c[k]) * x.to(device).float()
    W = W_U.to(device).float(); out = []
    with torch.no_grad():
        for i in range(0, len(pos), chunk):
            z = logits(hc[i:i + chunk], W); t = torch.as_tensor(tgt[i:i + chunk], device=device)
            out.append((torch.logsumexp(z, -1) - z.gather(-1, t[:, None])[:, 0]).cpu().numpy())
    return np.concatenate(out)


def stream_nll(H, tgt, c, W_U, device, chunk=1024):
    """Per-position NLL at c for a CPU-resident (K, P, d) bf16 tensor, streamed to one device (used for the hold-out)."""
    cd = torch.tensor(c, dtype=torch.float32, device=device); W = W_U.to(device).float(); out = []
    with torch.no_grad():
        for i in range(0, H.shape[1], chunk):
            h = H[:, i:i + chunk].to(device, non_blocking=False)
            z = logits(torch.tensordot(cd, h.float(), dims=1), W)
            t = torch.as_tensor(tgt[i:i + chunk], device=device)
            out.append((torch.logsumexp(z, -1) - z.gather(-1, t[:, None])[:, 0]).cpu().numpy())
    return np.concatenate(out)
