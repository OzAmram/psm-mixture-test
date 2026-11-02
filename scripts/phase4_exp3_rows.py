"""Experiment 3: give a checkpoint's answer run the reference fields prompt_tune_base.py prints (ll_self, n_tokens_self,
ll_generic, n_tokens), taken from the per-token self scores (phase4_self_tokens.py, chat template, the -0.898 convention)
and the base model's generic-header scores (phase1_basis_score.py). Writes results/phase4/exp3/rows/<name>.jsonl.
These fields only feed the trainer's printed reference lines; the gap itself is computed in the notebook from
self_tokens/*.npz and phase4_prefix_score.py outputs.

    python scripts/phase4_exp3_rows.py --run sft_unknown_casual_v1 --self results/phase4/self_tokens/sft_unknown_casual_v1.npz
Qwen replication (2026-10-30): --generic-hidden <hidden-state folder> takes the generic-header scores from its lp_generic.npy and
index.npz (the scorer that wrote them) instead of <run>/basis/generic.npz.
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--self", required=True)
    ap.add_argument("--out-dir", default="results/phase4/exp3/rows"); ap.add_argument("--generic-hidden", default=None); a = ap.parse_args()
    D = Path(a.run) if Path(a.run).is_dir() else Path("results/phase1") / a.run
    rows = [json.loads(l) for l in open(D / "rows.jsonl")]; s = np.load(a.self)
    if a.generic_hidden:
        H = Path(a.generic_hidden); ix = np.load(H / "index.npz"); gL, gN = np.load(H / "lp_generic.npy"), ix["n_tokens"]
        assert (ix["qidx"] == [r["qidx"] for r in rows]).all()
    else:
        g = np.load(D / "basis" / "generic.npz"); gL, gN = g["L"][:, 0], g["n_tokens"]
    assert len(rows) == len(s["total"]) == len(gN) and (s["qidx"] == [r["qidx"] for r in rows]).all()
    nself = np.diff(s["offsets"])
    for r, ls, ns, lg, ng in zip(rows, s["total"], nself, gL, gN):
        r.update(ll_self=float(ls), n_tokens_self=int(ns), ll_generic=float(lg), n_tokens=int(ng))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); f = out / f"{D.name}.jsonl"
    with open(f, "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {f}")


if __name__ == "__main__":
    main()
