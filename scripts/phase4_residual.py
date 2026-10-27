"""Experiment 1 residual: where does the best geometric fit still lose to the model's own self-score?

Per hold-out answer token, gap_t = log P_self(a_t) - log P_fit(a_t), with the self-score from phase4_self_tokens.py
(chat template, stripped answer) and the fit from phase4_geom_fit.py (holdout_nll_<fit>.npy, transcript format, answer
with its leading space). The two tokenisations differ in the first token (" Hey" vs "Hey"); the first token is dropped
from both and an answer is kept only if the remaining token ids are identical (the count dropped is reported).
Outputs results/phase4/exp1/<run>/residual_<fit>.json: gap by position (mean, se over answers), and the 30 token types
with the largest total gap (count, mean gap, total share), for the fit and for the generic header.

    python scripts/phase4_residual.py --run instruct_unknown_casual_v1 --fit A_free_lambda0
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.geometric import load_index, positions

HID = Path(os.environ.get("SCRATCH", "/pscratch/sd/o/ozamram")) / "persona_phase4" / "hidden"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--fit", required=True)
    ap.add_argument("--self", default=None, help="self_tokens npz (default results/phase4/self_tokens/<run>.npz)"); ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args(); E = Path("results/phase4/exp1") / a.run
    ix = load_index(HID / a.run); ha = np.load(sorted(E.glob("holdout_per_answer_*.npz"))[0]); rows = ha["rows"]
    st = np.load(a.self or f"results/phase4/self_tokens/{a.run}.npz"); so = st["offsets"]
    pos = positions(ix, rows); o = np.concatenate([[0], np.cumsum(ix["n_tokens"][rows])])
    nll = {k: np.load(E / f"holdout_nll_{k}.npy") for k in [a.fit, "generic"]}; assert all(len(v) == len(pos) for v in nll.values())
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
    G = {k: [] for k in nll}; T, P, kept = [], [], 0
    for i, r in enumerate(rows):
        tt = ix["targets"][pos[o[i]:o[i + 1]]]; ts = st["tok"][so[r]:so[r + 1]]
        if len(tt) < 2 or len(tt) != len(ts) or not (tt[1:] == ts[1:]).all(): continue
        kept += 1; lself = st["lp"][so[r]:so[r + 1]][1:]
        for k in nll: G[k].append(lself + nll[k][o[i]:o[i + 1]][1:])
        T.append(tt[1:]); P.append(np.arange(1, len(tt)))
    T, P = np.concatenate(T), np.concatenate(P); G = {k: np.concatenate(v) for k, v in G.items()}
    res = {"run": a.run, "fit": a.fit, "answers": int(len(rows)), "answers_aligned": kept, "tokens": int(len(T))}
    for k, g in G.items():
        byp = [{"pos": int(p), "n": int((P == p).sum()), "mean_gap": float(g[P == p].mean()), "se": float(g[P == p].std() / np.sqrt((P == p).sum()))}
               for p in np.unique(P) if (P == p).sum() >= 20]
        u, inv = np.unique(T, return_inverse=True); tot = np.bincount(inv, weights=g); cnt = np.bincount(inv)
        top = np.argsort(-tot)[: a.top]
        res[k] = {"mean_gap_per_token": float(g.mean()), "by_position": byp,
                  "top_types": [{"token": tok.decode([int(u[j])]), "count": int(cnt[j]), "mean_gap": float(tot[j] / cnt[j]), "share_of_total_gap": float(tot[j] / g.sum())} for j in top]}
        print(f"[{k}] mean gap {g.mean():.4f} nats/token over {len(g)} tokens ({kept}/{len(rows)} answers aligned)")
        for d in res[k]["top_types"][:10]: print(f"   {d['token']!r:14s} n={d['count']:5d} mean {d['mean_gap']:.2f} share {100*d['share_of_total_gap']:.1f}%")
    (E / f"residual_{a.fit}.json").write_text(json.dumps(res, indent=1)); print("->", E / f"residual_{a.fit}.json")


if __name__ == "__main__":
    main()
