"""Read the learned soft contexts (scripts/phase4_selfdesc_stage.sh) back out as text (author's request, 2026-10-28).
Each soft context started from exactly one self-description header (opener + description + casual clause), so two readouts:

  1. nearest tokens: each prefix vector -> the vocabulary token with the highest cosine similarity. Reports which positions
     changed from the original description and scores the projected token sequence as a hard prefix (exact ids, no retokenising).
  2. verbalisation (SALVE-style, Hu et al. 2026, arXiv:2609.16927, simplified to one round): after the soft context, the base
     model continues "Character description: The assistant"; each sampled description becomes a written header
     (basis.description_header) and is scored on a fixed batch of FIT-half answers; the best one is scored on the hold-out.
     Control: the same procedure started from the ORIGINAL written description instead of the soft context, so the gain from
     the soft context's correction can be separated from simply resampling a description.

Hold-out = the test questions of data/basis_split.json (the soft contexts never saw them). Selection uses fit-half answers only.

    python scripts/phase4_selfdesc_readout.py --part 0/4          (one per GPU; then --merge)
Writes results/phase4/selfdesc/readout/<name>.json and, with --merge, results/phase4/selfdesc/readout.json.
"""
import argparse, json, os, random, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_selection.basis import description_header, load_family, split
from phase4_hidden_states import hidden_for_context, load_rows, answer_logp, BASE

RUN = "instruct_unknown_casual_v1"; OUT = Path("results/phase4/selfdesc/readout")
PROBE = "Character description: The assistant"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="0/1"); ap.add_argument("--n-cand", type=int, default=16)
    ap.add_argument("--n-batch", type=int, default=96); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge: return merge()
    k, n = map(int, a.part.split("/"))
    soft = [json.loads(l) for f in ["selfdesc_overall_soft", "selfdesc_traits_soft"] for l in open(f"data/prompts/basis/{f}.jsonl")][k::n]
    desc = {c["name"]: c for f in ["selfdesc_overall", "selfdesc_traits"] for c in load_family(f)}
    tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
    E = model.get_input_embeddings().weight; En = torch.nn.functional.normalize(E.float(), dim=1)
    rows, suf = load_rows(RUN); fit_q, test_q = split()
    test = [i for i, r in enumerate(rows) if r["qidx"] in test_q] if "qidx" in rows[0] else None
    if test is None:
        q = np.load(Path(os.environ.get("SCRATCH", "/pscratch/sd/o/ozamram")) / "persona_phase4" / "hidden" / RUN / "index.npz")["qidx"]
        test = [i for i in range(len(rows)) if q[i] in test_q]; fit = [i for i in range(len(rows)) if q[i] in fit_q]
    else:
        fit = [i for i, r in enumerate(rows) if r["qidx"] in fit_q]
    batch = random.Random(a.seed).sample(fit, a.n_batch)

    @torch.no_grad()
    def score(idx, header="", prefix=None):
        tot = nt = 0
        for j, t, h in hidden_for_context(model, tok, header, [rows[i] for i in idx], suf, prefix=prefix):
            if t is None: continue
            tot += answer_logp(model, h, t); nt += len(t)
        return tot / nt

    @torch.no_grad()
    def sample(prefix_emb, n, max_new=260):
        ids = tok(PROBE, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev); e = model.get_input_embeddings()(ids)
        x = torch.cat([prefix_emb.to(e.dtype)[None], e], 1).expand(n, -1, -1)
        g = model.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=dev), max_new_tokens=max_new, do_sample=True,
                           temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        out = []
        for s in g:
            t = tok.decode(s, skip_special_tokens=True).split("\n")[0].rstrip(); cut = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
            t = ("The assistant" + t[:cut + 1]).replace("The assistant ’s", "The assistant’s").replace("The assistant 's", "The assistant's") if cut > 0 else ""
            if len(t.split()) >= 30: out.append(t)
        return list(dict.fromkeys(out))

    def verbalise(prefix_emb):
        torch.manual_seed(a.seed); cands = sample(prefix_emb, a.n_cand)
        sc = sorted(((score(batch, description_header(c)), c) for c in cands), reverse=True)
        if not sc: return {"n_candidates": 0}
        return {"n_candidates": len(sc), "best": sc[0][1], "best_batch": sc[0][0], "best_holdout": score(test, description_header(sc[0][1])),
                "median_batch": sc[len(sc) // 2][0], "top": sc[:5]}

    OUT.mkdir(parents=True, exist_ok=True)
    for r in soft:
        o = OUT / f"{r['name']}.json"
        if o.exists(): continue
        dn = r["init"].split(":")[1]; d = desc[dn]; hdr = d["header"]
        p = torch.load(r["prefix"], map_location=dev); p = (p["prefix"] if isinstance(p, dict) else p).squeeze(0)
        ids0 = tok(hdr, add_special_tokens=False)["input_ids"]; assert len(ids0) == p.shape[0], (len(ids0), p.shape)
        nn_ids = (torch.nn.functional.normalize(p.float(), dim=1) @ En.T).argmax(1).tolist()
        changed = [(i, tok.decode([ids0[i]]), tok.decode([nn_ids[i]])) for i in range(len(ids0)) if nn_ids[i] != ids0[i]]
        cos = torch.nn.functional.cosine_similarity(p.float(), E[ids0].float(), dim=1)
        res = {"name": r["name"], "init": r["init"], "description": d["description"], "n_tokens": len(ids0),
               "nearest_tokens": {"text": tok.decode(nn_ids), "n_changed": len(changed), "changed": changed,
                                  "median_cos_to_init": float(cos.median()), "holdout": score(test, prefix=E[nn_ids].detach())},
               "holdout": {"description": score(test, hdr), "soft": score(test, prefix=p)},
               "batch": {"description": score(batch, hdr), "soft": score(batch, prefix=p)}}
        res["verbalised_from_soft"] = verbalise(p); res["verbalised_from_description"] = verbalise(E[ids0].detach())
        o.write_text(json.dumps(res, indent=1))
        vs, vd = res["verbalised_from_soft"], res["verbalised_from_description"]
        print(f"{r['name']}: hold-out description {res['holdout']['description']:.3f} soft {res['holdout']['soft']:.3f} nearest-tokens {res['nearest_tokens']['holdout']:.3f} "
              f"({len(changed)}/{len(ids0)} changed) | verbalised from soft {vs.get('best_holdout', float('nan')):.3f}, from description {vd.get('best_holdout', float('nan')):.3f}", flush=True)


def merge():
    R = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    col = lambda f: np.array([f(r) for r in R])
    rows = {"description": col(lambda r: r["holdout"]["description"]), "soft context": col(lambda r: r["holdout"]["soft"]),
            "nearest tokens of soft": col(lambda r: r["nearest_tokens"]["holdout"]),
            "best verbalisation, sampled after soft": col(lambda r: r["verbalised_from_soft"].get("best_holdout", np.nan)),
            "best verbalisation, sampled after description (control)": col(lambda r: r["verbalised_from_description"].get("best_holdout", np.nan))}
    summ = {k: {"median": float(np.nanmedian(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))} for k, v in rows.items()}
    gain = rows["soft context"] - rows["description"]; vgain = rows["best verbalisation, sampled after soft"] - rows["description"]
    cgain = rows["best verbalisation, sampled after description (control)"] - rows["description"]
    summ["fraction of soft gain kept by verbalisation (median)"] = float(np.nanmedian(vgain / gain))
    summ["fraction of soft gain kept by control resampling (median)"] = float(np.nanmedian(cgain / gain))
    summ["changed tokens (median of n_changed / n_tokens)"] = float(np.median([r["nearest_tokens"]["n_changed"] / r["n_tokens"] for r in R]))
    Path("results/phase4/selfdesc/readout.json").write_text(json.dumps({"n": len(R), "summary": summ}, indent=1))
    for k, v in summ.items(): print(f"  {k:62s} {v}")


if __name__ == "__main__":
    main()
