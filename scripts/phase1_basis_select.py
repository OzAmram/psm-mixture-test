"""Screening step for large families (Assistant Axis roles / traits, IPIP facets): from the fit-half screening scores
(basis/<family>__screen.npz, written by phase1_basis_score.py --screen), keep the union of
  * the first --greedy components of a greedy forward selection maximising the mixture likelihood of the screening answers, and
  * the --top components with the highest individual likelihood,
and write them as a new family <family>_top (data/prompts/basis/), to be scored on every answer. Only fit-half answers are used,
so selection cannot leak hold-out information.

    python scripts/phase1_basis_select.py --run instruct_unknown_casual_v1 --family axis_roles --greedy 15 --top 15
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.mixture import em_weights, mixture_loglik
from persona_selection.basis import load_family, write_family

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--family", required=True)
ap.add_argument("--greedy", type=int, default=15); ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()
z = np.load(Path("results/phase1") / a.run / "basis" / f"{a.family}__screen.npz"); L, names = z["L"], [str(n) for n in z["names"]]
ok = ~z["meta"] & np.isfinite(L).all(1); L = L[ok]
chosen = []
for _ in range(a.greedy):
    best = None
    for c in range(L.shape[1]):
        if c in chosen: continue
        w, _ = em_weights(L[:, chosen + [c]], n_iter=300); v = mixture_loglik(L[:, chosen + [c]], w).sum()
        if best is None or v > best[0]: best = (v, c)
    chosen.append(best[1])
top = list(np.argsort(-L.sum(0))[: a.top])
keep = chosen + [int(c) for c in top if c not in chosen]
comps = {c["name"]: c for c in load_family(a.family)}
write_family(f"{a.family}_top", [comps[names[c]] for c in keep])
print(f"{a.family}: {L.shape[0]} screening answers; greedy order: {[names[c] for c in chosen]}; + top individual: {[names[c] for c in top if c not in chosen]} -> {a.family}_top ({len(keep)})")
