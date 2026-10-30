"""Self-description basis with the verbalised soft contexts swapped in (author's request, 2026-10-28): every description
that was verbalised (results/phase4/selfdesc/salve_big/ where it exists, else results/phase4/selfdesc/salve/) is replaced
by its best verbalisation, chosen among the soft-conditioned arms (extend_soft, rewrite_soft, salve_soft) by the FIT-half
batch score; all other descriptions are kept. Replaced entries get the name prefix "v_" (new hidden states needed).
Writes families selfdesc_overall_v and selfdesc_traits_v and results/phase4/selfdesc/verb_basis.json.

    python scripts/phase4_selfdesc_verbbasis.py --prefix v_ --suffix _v          (2026-10-28: the 8 contexts then verbalised)
    python scripts/phase4_selfdesc_verbbasis.py --prefix v35_ --suffix _v35      (2026-10-30: all 35, salve_big; new names so no stale hidden states)
Arms missing from a result file (the all-35 run has only salve_soft) are skipped.
"""
import argparse, json
from pathlib import Path
from persona_selection.basis import description_header, load_family, write_family

R = Path("results/phase4/selfdesc"); ARMS = ["extend_soft", "rewrite_soft", "salve_soft"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="v35_"); ap.add_argument("--suffix", default="_v35"); a = ap.parse_args()
    best = {}
    for d in ["salve", "salve_big"]:          # salve_big overrides salve
        for f in sorted((R / d).glob("*.json")):
            r = json.loads(f.read_text()); arm = max([x for x in ARMS if x in r["arms"]], key=lambda x: r["arms"][x]["batch"])
            best[r["name"]] = {"source": d, "arm": arm, "text": r["arms"][arm]["text"], "batch": r["arms"][arm]["batch"], "holdout": r["arms"][arm]["holdout"],
                               "seed_holdout": r["holdout"]["seed description"]}
    for fam in ["selfdesc_overall", "selfdesc_traits"]:
        comps = []
        for c in load_family(fam):
            c = {k: v for k, v in c.items() if k != "family"}
            if c["name"] in best:
                b = best[c["name"]]; c.update({"name": a.prefix + c["name"], "replaces": c["name"], "description": b["text"], "header": description_header(b["text"]), "verbalisation": {k: b[k] for k in ["source", "arm"]}})
            comps.append(c)
        write_family(fam + a.suffix, comps)
    (R / f"verb_basis{a.suffix}.json").write_text(json.dumps(best, indent=1))
    for n, b in best.items(): print(f"{n:22s} {b['source']:9s} {b['arm']:12s} hold-out {b['seed_holdout']:.3f} -> {b['holdout']:.3f}")


if __name__ == "__main__":
    main()
