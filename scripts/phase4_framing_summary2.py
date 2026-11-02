"""Summarise framing round 2 (scripts/phase4_framing2.sh) together with round 1: every (model, framing), in nats per
REFERENCE token (per-answer total log P summed, divided by the base-transcript token counts), so chat framings (answer
without the leading space, possibly a different token count) are on the same scale. Paired standard errors over answers
against the base model's transcript framing with the same description.

    python scripts/phase4_framing_summary2.py   -> results/phase4/framing/summary2.json
    python scripts/phase4_framing_summary2.py --dir results/phase4/qwen/framing --files "base:scores_qwenbase_0.json,scores_qwenbase_1.json,scores_qwenbase_2.json,scores_qwenbase_3.json" \
        --nref scores_qwenbase_0.json        (Qwen replication, 2026-10-30)
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results/phase4/framing"); ap.add_argument("--nref", default="scores_base_turn.json")
ap.add_argument("--files", default="base:scores.json,scores_base_hyb.json,scores_base_turn.json;sft:scores_sft_chat.json,scores_sft_raw.json;instruct:scores_instruct_chat.json")
a = ap.parse_args(); O = Path(a.dir); FILES = {m: fs.split(",") for m, fs in (x.split(":") for x in a.files.split(";"))}
T, D = {}, None
NR = {h: np.array(v) for h, v in json.loads((O / a.nref).read_text())["n_ref"].items()}    # same seed-0 answers in every file
for m, fs in FILES.items():
    for f in fs:
        r = json.loads((O / f).read_text()); D = D or r["descriptions"]
        tot = r.get("total") or {k: np.array(v) * NR[k.split("|")[2]] for k, v in r["per_answer"].items()}   # round 1: base tokens = n_ref
        for k, v in tot.items():
            f_, n, h = k.split("|"); T.setdefault((m, f_, n, h), np.array(v))
descs = [d for d in D if d != "generic"]; se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
pt = lambda x, h: x / NR[h]             # per-answer, per reference token
out = {}
for h in ["fit", "test"]:
    print(f"\n{h} half (300 answers), nats per reference token; (+-) paired SE vs base transcript")
    print(f"{'model':8s} {'framing':14s} {'generic':>8s} {'mean desc':>18s} {'desc-generic':>14s} {'best desc':>10s}")
    ref_g = T[("base", "transcript", "generic", h)]; ref_m = np.mean([T[("base", "transcript", d, h)] for d in descs], 0)
    for m, f_ in dict.fromkeys((k[0], k[1]) for k in T):
        g = T[(m, f_, "generic", h)]; have = [d for d in descs if (m, f_, d, h) in T]
        o = {"generic": float(g.sum() / NR[h].sum()), "generic_minus_base_transcript_se": se(pt(g - ref_g, h))}
        if have:
            M = np.mean([T[(m, f_, d, h)] for d in have], 0); best = max(have, key=lambda d: T[(m, f_, d, h)].sum())
            o.update({"mean_desc": float(M.sum() / NR[h].sum()), "mean_desc_minus_base_se": se(pt(M - ref_m, h)), "desc_gain": float((M - g).sum() / NR[h].sum()),
                      "desc_gain_se": se(pt(M - g, h)), "best_desc": best, "best_desc_score": float(T[(m, f_, best, h)].sum() / NR[h].sum()),
                      "per_desc": {d: float(T[(m, f_, d, h)].sum() / NR[h].sum()) for d in have}})
            print(f"{m:8s} {f_:14s} {o['generic']:8.4f} {o['mean_desc']:9.4f} (±{o['mean_desc_minus_base_se']:.3f}) {o['desc_gain']:+8.3f}±{o['desc_gain_se']:.3f} {o['best_desc_score']:10.4f}  {best}")
        else:
            print(f"{m:8s} {f_:14s} {o['generic']:8.4f}   (no descriptions: this framing ignores them)")
        out[f"{m}|{f_}|{h}"] = o
(O / "summary2.json").write_text(json.dumps(out, indent=1))
