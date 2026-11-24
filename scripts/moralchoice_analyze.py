"""Analysis of scripts/moralchoice_pilot.py output: per ambiguity level, each model's mean P(action1), mean |diff| vs Instruct,
Pearson vs Instruct, agreement of the preferred option, order inconsistency, A/B mass, position bias, with a paired bootstrap
over scenarios (95% percentile intervals). Also differences "model |diff| minus Instruct's own order inconsistency".

    python scripts/moralchoice_analyze.py results/phase4/moralchoice/pilot.json [--boot 2000] [--seed 0]
"""
import argparse, json
import numpy as np, pandas as pd

KEY = ["instruct", "arithmetic", "geometric", "soft", "generic", "hhh", "e14"]


def stats(m, i):
    """m, i: DataFrames (same scenario order) for a model and for Instruct."""
    p, q = m.p1.values, i.p1.values
    return {"mean_p1": p.mean(), "abs_diff": np.abs(p - q).mean(),
            "pearson": np.corrcoef(p, q)[0, 1] if p.std() > 0 and q.std() > 0 else np.nan,
            "agree": ((p > 0.5) == (q > 0.5)).mean(),
            # same-order comparison: |model - Instruct| within each order, averaged (comparable to Instruct's own order inconsistency)
            "abs_diff_same_order": (np.abs(m.p1_a1first.values - i.p1_a1first.values) + np.abs(m.p1_a2first.values - i.p1_a2first.values)).mean() / 2,
            "order_incons": np.abs(m.p1_a1first.values - m.p1_a2first.values).mean(),
            "consistent_choice": ((m.p1_a1first.values > 0.5) == (m.p1_a2first.values > 0.5)).mean(),
            "ab_mass": m.ab_mass.mean(), "pA": m.pA_norm.mean()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); d = json.load(open(a.path)); df = pd.DataFrame(d["per_scenario"])
    singles = [m for m in df.model.unique() if m not in ("instruct", "arithmetic", "geometric", "soft")]
    rng = np.random.default_rng(a.seed); out = {}
    for lvl in ["low", "high"]:
        dl = df[df.ambiguity == lvl]; ids = sorted(dl.id.unique()); n = len(ids)
        T = {mdl: dl[dl.model == mdl].set_index("id").loc[ids] for mdl in df.model.unique()}
        I = T["instruct"]; inst_det = np.maximum(I.p1_a1first, 1 - I.p1_a1first).tolist() + np.maximum(I.p1_a2first, 1 - I.p1_a2first).tolist()
        print(f"\n=== {lvl} ambiguity: {n} scenarios. Instruct per-prompt max(P) median {np.median(inst_det):.3f}, "
              f"fraction > 0.9: {np.mean(np.array(inst_det) > 0.9):.2f}")
        B = [rng.integers(0, n, n) for _ in range(a.boot)]; o = {}
        for mdl in KEY:
            pt = stats(T[mdl], I); bs = [stats(T[mdl].iloc[b], I.iloc[b]) for b in B]
            ci = {k: (np.nanpercentile([x[k] for x in bs], 2.5), np.nanpercentile([x[k] for x in bs], 97.5)) for k in pt}
            # paired: model |diff| (order-averaged) minus Instruct's own order inconsistency
            g = [x["abs_diff"] - y["order_incons"] for x, y in zip(bs, [stats(I.iloc[b], I.iloc[b]) for b in B])]
            o[mdl] = {k: float(v) for k, v in pt.items()} | {f"{k}_ci": [float(c) for c in ci[k]] for k in pt} | \
                {"abs_diff_minus_inst_order_incons": float(pt["abs_diff"] - stats(I, I)["order_incons"]), "..._ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]}
        print(f"{'model':10s} | mean P(a1) [95%] | mean|diff| vs Inst [95%] | Pearson [95%] | agree [95%] | same-order |diff| | order incons | consistent | A/B mass | P(A)")
        for mdl in KEY:
            x = o[mdl]; f = lambda k, p=3: f"{x[k]:.{p}f} [{x[k+'_ci'][0]:.{p}f},{x[k+'_ci'][1]:.{p}f}]"
            print(f"{mdl:10s} | {f('mean_p1')} | {f('abs_diff')} | {f('pearson', 2)} | {f('agree', 2)} | {x['abs_diff_same_order']:.3f} | "
                  f"{f('order_incons')} | {x['consistent_choice']:.2f} | {x['ab_mass']:.3f} | {x['pA']:.2f}")
        # paired differences between models in |diff| vs Instruct
        for m1, m2 in [("soft", "arithmetic"), ("arithmetic", "generic"), ("geometric", "arithmetic"), ("hhh", "arithmetic")]:
            pt = o[m1]["abs_diff"] - o[m2]["abs_diff"]; g = [stats(T[m1].iloc[b], I.iloc[b])["abs_diff"] - stats(T[m2].iloc[b], I.iloc[b])["abs_diff"] for b in B]
            print(f"  |diff| {m1} - {m2}: {pt:+.3f} [{np.percentile(g, 2.5):+.3f},{np.percentile(g, 97.5):+.3f}]"); o[f"{m1}_minus_{m2}_abs_diff"] = [pt, float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]
        sd = sorted((stats(T[s], I)["abs_diff"], s) for s in singles)
        print(f"  87 single contexts, mean|diff| vs Instruct: best {sd[0][0]:.3f} ({sd[0][1]}; chosen on this data), median {sd[len(sd)//2][0]:.3f}, worst {sd[-1][0]:.3f} ({sd[-1][1]})")
        o["singles_abs_diff"] = {s: float(v) for v, s in sd}; out[lvl] = o
    p = a.path.replace(".json", "_analysis.json"); json.dump(out, open(p, "w"), indent=1); print("->", p)


if __name__ == "__main__":
    main()
