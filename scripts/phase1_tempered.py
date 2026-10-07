"""Tempered-mixture test ("selection plus sharpening"): can a mixture of base personas with per-token temperature reproduce
the Instruct model's likelihood of its own answers?

Token-level tempering: for component s and temperature T, log P_T(a | q, s) = sum_t [ z_s,t(a_t)/T - logsumexp(z_s,t / T) ],
where z_s,t are the base model's logits at position t under persona header s. (This is NOT sequence-level tempering
P^(1/T)/Z, which is intractable; it is what sampling at temperature T would give.) We score every answer under a short list
of components (the ones the untempered fit uses, plus the generic header) for a grid of T in ONE forward pass per
(answer, component), then fit mixture weights by maximum likelihood on the training-question answers for each T (shared T
across components, plus a per-component-T greedy refinement) and report hold-out log P per token next to the Instruct
model's own (exact) log P and the untempered mixture. The base-assistant control (its own sampler is the generic header)
is scored the same way as a sanity check: its best T should be ~1.

    python scripts/phase1_tempered.py score --run instruct_unknown_casual_v1 --out results/phase1/tempered_instruct.npz
    python scripts/phase1_tempered.py score --run base_unknown_casual_short_v1 --out results/phase1/tempered_base.npz
    python scripts/phase1_tempered.py fit
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
sys.path.insert(0, "src")
from persona_selection.phase1_prompts import load_persona, generic_prompt, component_prompt
from persona_selection.mixture import em_weights, mixture_loglik, is_meta_response

COMPONENTS = ["hhh", "e14", "fred", "neutral", "e49", "e69", "e58", "e32", "evil"]     # untempered-fit components (weight >= 0.02 in instruct or base fits) + evil as a control
TS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
FRAMING, REGISTER = "unknown", "casual"


def persona(name):
    return load_persona(name, "data/prompts/personas_elicited" if name.startswith("e") and name[1:].isdigit() else "data/prompts/personas")


def score(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    d = Path("results/phase1") / args.run; cfg = json.load(open(d / "config.json")); suffix = cfg["args"]["user_suffix"]
    rows = [json.loads(l) for l in open(d / "rows.jsonl")]
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B"); model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B", dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
    P = {n: persona(n) for n in COMPONENTS}; names = ["generic"] + COMPONENTS
    L = np.full((len(rows), len(names), len(TS)), np.nan); NT = np.zeros(len(rows), int); t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            q = r["question"] + suffix; resp = r["response"]
            for j, n in enumerate(names):
                prompt = generic_prompt(q, FRAMING, REGISTER) if n == "generic" else component_prompt(q, P[n], FRAMING, register=REGISTER)
                full = tok(prompt + resp, return_tensors="pt")["input_ids"][0]; p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
                if not torch.equal(full[:len(p_ids)], p_ids) or len(full) == len(p_ids): continue
                logits = model(full[None].to(dev)).logits[0, len(p_ids) - 1:-1].float(); tgt = full[len(p_ids):].to(dev)
                z = logits.gather(-1, tgt[:, None]).squeeze(-1)
                for k, T in enumerate(TS):
                    L[i, j, k] = (z / T - torch.logsumexp(logits / T, -1)).sum().item()
                NT[i] = len(tgt)
            if (i + 1) % 100 == 0: print(f"[{args.run}] {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(args.out, L=L, n_tokens=NT, names=np.array(names), temps=np.array(TS), qidx=np.array([r["qidx"] for r in rows]), meta=np.array([is_meta_response(r["response"]) for r in rows]))
    print("->", args.out)


def fit(args):
    out = {}; md = ["# Tempered-mixture test (scripts/phase1_tempered.py)\n"]
    ex = None
    exf = Path("results/phase1/instruct_unknown_casual_v1/rows_selfexact.jsonl")
    if exf.exists(): ex = [json.loads(l) for l in open(exf)]
    for run, fn, label in [("instruct_unknown_casual_v1", "results/phase1/tempered_instruct.npz", "Instruct answers"), ("base_unknown_casual_short_v1", "results/phase1/tempered_base.npz", "base-assistant answers (control)")]:
        if not Path(fn).exists(): continue
        z = np.load(fn); L, NT, names, TS_, qidx, meta = z["L"], z["n_tokens"], list(z["names"]), list(z["temps"]), z["qidx"], z["meta"]
        ok = ~meta & np.isfinite(L).all(axis=(1, 2)); L, NT, qidx = L[ok], NT[ok], qidx[ok]
        rng = np.random.default_rng(0); qs = np.unique(qidx); rng.shuffle(qs); test_q = set(qs[: len(qs) // 2]); te = np.array([g in test_q for g in qidx]); tr = ~te
        comp = [j for j, n in enumerate(names) if n != "generic"]; gidx = names.index("generic")
        res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "shared_T": {}, "generic_T": {}}
        md.append(f"\n**{label}** ({int(te.sum())} hold-out answers; log P / token)\n\n| T | generic header at T | best single component at T (train-selected) | mixture of {len(comp)} components at T (weights fit on train) |\n|---|---|---|---|")
        best = None
        for k, T in enumerate(TS_):
            Lt = L[:, comp, k]; w, _ = em_weights(Lt[tr]); mix = mixture_loglik(Lt[te], w)
            b = int(np.argmax(Lt[tr].sum(0))); single = Lt[te][:, b]
            gen = L[te, gidx, k]
            vals = {"generic": float(gen.sum() / NT[te].sum()), "single": float(single.sum() / NT[te].sum()), "single_name": names[comp[b]], "mixture": float(mix.sum() / NT[te].sum()), "train_mixture": float(mixture_loglik(Lt[tr], w).sum() / NT[tr].sum()), "w": {names[comp[j]]: float(w[j]) for j in np.argsort(-w)[:5]}}
            res["shared_T"][str(T)] = vals
            md.append(f"| {T:.1f} | {vals['generic']:.3f} | {vals['single']:.3f} ({vals['single_name']}) | {vals['mixture']:.3f} |")
            if best is None or vals["train_mixture"] > best[1]: best = (T, vals["train_mixture"], vals["mixture"])
        # per-component temperature: coordinate ascent on the training likelihood, starting from the best shared T
        Tk = {j: TS_.index(best[0]) for j in comp}
        def Lsel(sel): return np.stack([L[:, j, sel[j]] for j in comp], 1)
        def train_ll(sel):
            Lt = Lsel(sel); w, _ = em_weights(Lt[tr]); return mixture_loglik(Lt[tr], w).sum(), w
        cur, w = train_ll(Tk)
        for _ in range(3):
            for j in comp:
                for k in range(len(TS_)):
                    trial = dict(Tk); trial[j] = k; v, _ = train_ll(trial)
                    if v > cur + 1e-6: cur, Tk = v, trial
        Lt = Lsel(Tk); w, _ = em_weights(Lt[tr]); mix = mixture_loglik(Lt[te], w)
        res["per_component_T"] = {"T": {names[j]: TS_[Tk[j]] for j in comp}, "w": {names[j]: float(w[i]) for i, j in enumerate(comp)}, "mixture_test": float(mix.sum() / NT[te].sum())}
        # question-bootstrap sigma for the best shared-T mixture and the per-component mixture
        kb = TS_.index(best[0]); Lb = L[:, comp, kb]; wb, _ = em_weights(Lb[tr]); mixb = mixture_loglik(Lb[te], wb); g = qidx[te]; uq = np.unique(g); rng2 = np.random.default_rng(1); reps = []
        for _ in range(300):
            pick = rng2.choice(uq, len(uq)); m = np.concatenate([np.where(g == q)[0] for q in pick]); reps.append((mixb[m].sum() / NT[te][m].sum(), mix[m].sum() / NT[te][m].sum()))
        reps = np.array(reps); res["best_shared_T"] = {"T": best[0], "mixture_test": best[2], "se": float(reps[:, 0].std())}; res["per_component_T"]["se"] = float(reps[:, 1].std())
        ref = None
        if run == "instruct_unknown_casual_v1" and ex is not None:
            exa = np.array([e["ll_self_exact"] for e in ex])[ok]; exn = np.array([e["n_tokens_exact"] for e in ex])[ok]; ref = float(exa[te].sum() / exn[te].sum()); res["instruct_self_exact"] = ref
        md.append(f"\nbest shared T by training likelihood: {best[0]:.1f}, hold-out mixture log P/token {best[2]:.3f} ± {res['best_shared_T']['se']:.3f}; per-component T ({', '.join(f'{n} {t:.1f}' for n, t in res['per_component_T']['T'].items())}): {res['per_component_T']['mixture_test']:.3f} ± {res['per_component_T']['se']:.3f}" + (f"; Instruct's own exact log P/token on the same answers: {ref:.3f}" if ref is not None else "") + f"; untempered mixture (T = 1): {res['shared_T']['1.0']['mixture']:.3f}")
        out[label] = res
    Path("results/phase1/tempered.json").write_text(json.dumps(out, indent=1)); Path("results/phase1/tempered.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["score", "fit"]); ap.add_argument("--run"); ap.add_argument("--out"); a = ap.parse_args()
    score(a) if a.mode == "score" else fit(a)
