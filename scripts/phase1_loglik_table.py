"""Mixture results as log-likelihoods instead of KL: on the hold-out half of the questions, mean log P per token (and per
response) of the sampled answers under
    the sampling model itself   (Instruct: exact self log-prob from rows_selfexact.jsonl if present, else the stored one;
                                 base assistant: its own generic header)
    the base model's generic header
    the fitted mixture of base personas (weights fitted on the other half of the questions)
    the best single persona (highest held-out log-likelihood)
for the Instruct model and for the base assistant control (both with the Phase 1 'unknown + casual' conditioning). Same
machinery as notebook 1.7 (meta/placeholder responses removed, question-level split). Writes results/phase1/loglik_table.{json,md}.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from persona_selection.mixture import em_weights, is_meta_response, mixture_loglik
HAND = ["hhh", "fred", "evil", "sycophant", "formal", "neutral"]


def load(run):
    d = Path("results/phase1") / run
    rows = [json.loads(l) for l in open(d / "rows.jsonl")]
    ll = {i: dict(r["ll"]) for i, r in enumerate(rows)}
    for fn in sorted(glob.glob(str(d / "rows_elicited*.jsonl"))):
        for i, l in enumerate(open(fn)): ll[i].update(json.loads(l)["ll"])
    el = sorted({k for x in ll.values() for k in x if k.startswith("e") and k[1:].isdigit()}); names = [n for n in HAND if n in ll[0]] + el
    idx = [i for i in range(len(rows)) if all(n in ll[i] for n in names) and not is_meta_response(rows[i]["response"])]
    exact = None
    if (d / "rows_selfexact.jsonl").exists():
        ex = [json.loads(l) for l in open(d / "rows_selfexact.jsonl")]; assert len(ex) == len(rows)
        exact = np.array([ex[i]["ll_self_exact"] for i in idx]); nt_exact = np.array([ex[i]["n_tokens_exact"] for i in idx])
    else:
        nt_exact = None
    return {"L": np.array([[ll[i][n] for n in names] for i in idx]), "l0": np.array([rows[i]["ll_generic"] for i in idx]),
            "self_old": np.array([rows[i].get("ll_self", np.nan) for i in idx]), "self_exact": exact, "nt_exact": nt_exact,
            "nt": np.array([rows[i]["n_tokens"] for i in idx]), "groups": np.array([rows[i]["qidx"] for i in idx]), "names": names, "n": len(idx)}


def table(d, label, seed=0, test_frac=0.5):
    rng = np.random.default_rng(seed); qs = np.unique(d["groups"]); rng.shuffle(qs); test_q = set(qs[: int(round(test_frac * len(qs)))])
    te = np.array([g in test_q for g in d["groups"]]); tr = ~te
    w, _ = em_weights(d["L"][tr]); mix = mixture_loglik(d["L"][te], w)
    best = int(np.argmax(d["L"][te].sum(0))); nt = d["nt"][te]; N = te.sum()
    cols = {"base generic header": d["l0"][te], "base persona mixture (86, weights fitted on the other half)": mix, f"best single base persona ({d['names'][best]})": d["L"][te][:, best]}
    if d["self_exact"] is not None: cols = {"sampling model itself (exact)": d["self_exact"][te], **cols}; nt_self = d["nt_exact"][te]
    elif np.isfinite(d["self_old"]).all(): cols = {"sampling model itself (stored, leading-space artefact)": d["self_old"][te], **cols}; nt_self = nt
    else: nt_self = nt
    out = {}
    for k, v in cols.items():
        n_tok = nt_self if k.startswith("sampling") else nt
        out[k] = {"per_token": float(v.sum() / n_tok.sum()), "per_response": float(v.mean()), "se_per_response": float(v.std() / np.sqrt(N))}
    ref_r = next(iter(out.values()))["per_response"]; ref_t = next(iter(out.values()))["per_token"]
    for k in out:
        out[k]["gap_to_sampling_model_per_response"] = ref_r - out[k]["per_response"]; out[k]["gap_to_sampling_model_per_token"] = ref_t - out[k]["per_token"]
    # per-token standard error via a question-level bootstrap of the token-weighted mean
    qs = d["groups"][te]; uq = np.unique(qs); rng2 = np.random.default_rng(1)
    for k, v in cols.items():
        n_tok = nt_self if k.startswith("sampling") else nt; reps = []
        for _ in range(300):
            pick = rng2.choice(uq, len(uq)); m = np.concatenate([np.where(qs == q)[0] for q in pick]); reps.append(v[m].sum() / n_tok[m].sum())
        out[k]["se_per_token"] = float(np.std(reps))
    return {"label": label, "n_heldout_answers": int(N), "n_heldout_questions": len(test_q), "columns": out, "top_weights": {d["names"][i]: float(w[i]) for i in np.argsort(-w)[:6]}}


def main():
    res = {k: table(load(r), k) for k, r in [("Instruct (chat template, default system prompt)", "instruct_unknown_casual_v1"), ("base assistant control (generic header, same suffix)", "base_unknown_casual_short_v1")]}
    md = ["# Mixture fit as log-likelihoods (scripts/phase1_loglik_table.py)\n", "Hold-out half of the questions; token-weighted mean log P of the sampled answers. Per token is the primary unit (it removes the different answer lengths of the two sampling models); 'gap' = (sampling model) − (column); the sampling model's own log-likelihood is the ceiling any context or mixture could reach.\n"]
    for k, t in res.items():
        md.append(f"\n**{k}** ({t['n_heldout_answers']} answers on {t['n_heldout_questions']} questions)\n\n| scored under | log P / token (±1σ, question bootstrap) | gap to sampling model, nats / token | log P / response (for scale) |\n|---|---|---|---|")
        for c, v in t["columns"].items(): md.append(f"| {c} | {v['per_token']:.3f} ± {v['se_per_token']:.3f} | {v['gap_to_sampling_model_per_token']:.3f} | {v['per_response']:.1f} |")
        md.append("\ntop fitted weights: " + ", ".join(f"{n} {w:.2f}" for n, w in t["top_weights"].items()))
    Path("results/phase1/loglik_table.json").write_text(json.dumps(res, indent=1)); Path("results/phase1/loglik_table.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
