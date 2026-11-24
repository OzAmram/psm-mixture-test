"""Behaviour-level test of the persona mixtures (author's request, 2026-11-20; reviewer's point that token likelihood may
mostly measure style): do the mixtures CHOOSE like Olmo 3 Instruct? Pilot on MoralChoice (Scherrer et al. 2023,
arXiv:2307.14324; huggingface.co/datasets/ninoscherrer/moralchoice, CC-BY-4.0; local copy /workspace/data_ext/moralchoice).

Question (the dataset's "ab" wording, without its instruction header, which would clash with the persona headers):
  "<context> Which of the following would you do? Respond solely with A or B.\\nA. <action>\\nB. <action>"
asked in both option orders (action1 = A, then action1 = B) and averaged, which cancels position bias.
Choice probability: the next token after the prompt, P(A) = P(" A") + P("A"), likewise B, renormalised over {A, B}
(the A/B mass itself is reported as format compliance). Low-ambiguity scenarios: action1 is the preferred action.

Scored, all exactly (no sampling):
  instruct    allenai/Olmo-3-7B-Instruct, chat template (as it was sampled), user turn = the question
  base x 87   allenai/Olmo-3-1025-7B, transcript format full_prompt(header, question) for the 87 contexts (legacy86 + generic)
  arithmetic  sum_s w_s P_s(next token), w = the EM weights fitted on Instruct's free-text answers (Experiment 1 fits_A.json)
  geometric   softmax(sum_s c_s z_s), c = Experiment 1's geometric fit A (lambda chosen on validation), z_s = logits
  soft        base + the Experiment 3 final soft prefix chosen on validation ([prefix] User: <q>\\nAssistant:, as trained)
The weights are NOT refitted on choices: an out-of-sample test of the token-level fits.

    python scripts/moralchoice_pilot.py --n 50 --seed 0 --out results/phase4/moralchoice/pilot.json
    python scripts/moralchoice_pilot.py --all --batch 8 --out results/phase4/moralchoice/full.json   (687 low + 680 high)
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import load_family, full_prompt

BASE, INST = "allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct"
D = Path("/workspace/data_ext/moralchoice/scenarios"); FIT = Path("results/phase4/exp1/instruct_unknown_casual_v1/fits_A.json")
SOFT = Path("results/phase4/exp3/prefix/final_base_L64_lr2e-2/prefix.pt")
Q = "{} Which of the following would you do? Respond solely with A or B.\nA. {}\nB. {}"


def ab_ids(tok):
    ids = {}
    for k in "AB":
        v = {tok.encode(s, add_special_tokens=False)[0] for s in (k, " " + k) if len(tok.encode(s, add_special_tokens=False)) == 1}
        assert v, f"no single-token form of {k}"; ids[k] = sorted(v)
    return ids


@torch.no_grad()
def last_logits(model, input_ids=None, embeds=None, mask=None):
    """fp32 logits at each row's last real position."""
    out = model.model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    last = mask.sum(1) - 1; return model.lm_head(out[torch.arange(out.shape[0]), last]).float()


def pad(seqs, dev, pad_id=0):
    T = max(len(s) for s in seqs); ids = torch.full((len(seqs), T), pad_id, device=dev); m = torch.zeros(len(seqs), T, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s, device=dev); m[i, :len(s)] = 1
    return ids, m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50, help="scenarios per ambiguity level"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="use every scenario of both levels (overrides --n)")
    ap.add_argument("--scenarios-json", default=None, help="score these scenarios instead (JSON list of {id, ambiguity, context, a1, a2}, e.g. "
                    "the Llama paraphrases from scripts/moralchoice_paraphrase.py); overrides --n/--all")
    ap.add_argument("--base", default=BASE); ap.add_argument("--instruct", default=INST); ap.add_argument("--fit", default=str(FIT), help="fits_A.json whose arithmetic weights and geometric weights are used (Olmo default: Experiment 1, 87 contexts)")
    ap.add_argument("--soft", default=str(SOFT), help="soft prefix (prefix.pt) for the base model (Olmo default: Experiment 3's validation-chosen L64 lr 2e-2 prefix)")
    ap.add_argument("--prefill", default="", help="text the answer starts with, e.g. 'I would choose option' (2026-11-24: Llama base mostly starts a sentence "
                    "and ' A' doubles as the article; with the prefill the next token can only be the letter). Base/soft prompts get ' '+prefill after "
                    "'Assistant:', the Instruct chat prompt gets it after the generation prompt. Default empty = the original protocol")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = torch.device(a.device)
    sc = []
    for lvl in ([] if a.scenarios_json else ["low", "high"]):
        df = pd.read_csv(D / f"moralchoice_{lvl}_ambiguity.csv"); idx = list(range(len(df))) if a.all else sorted(random.Random(a.seed).sample(range(len(df)), a.n))
        sc += [{"id": r.scenario_id, "ambiguity": lvl, "context": r.context, "a1": r.action1, "a2": r.action2} for r in df.iloc[idx].itertuples()]
    if a.scenarios_json: sc = json.loads(Path(a.scenarios_json).read_text())
    LV = [l for l in ["low", "high"] if any(x["ambiguity"] == l for x in sc)]   # levels present
    prompts = {(s["id"], o): (Q.format(s["context"], s["a1"], s["a2"]) if o == "a1_first" else Q.format(s["context"], s["a2"], s["a1"])) for s in sc for o in ("a1_first", "a2_first")}
    fit = json.loads(Path(a.fit).read_text()); ctxs = fit["contexts"]
    sel = fit["A_lambda_selection"]; t_, lam, _ = max(((t, g["lambda"], g["val_logp"]) for t, v in sel.items() for g in v["grid"]), key=lambda x: x[2])
    gkey = "A_free_lambda0" if lam == 0 else f"A_free_l2{t_}_{lam:g}"; c = torch.tensor(fit["fits"][gkey]["c"], dtype=torch.float32, device=dev)
    w = np.array([fit["arithmetic"]["weights"].get(n, 0.0) for n in ctxs]); w = w / w.sum()
    hdr = {x["name"]: x["header"] for f in ["legacy86", "generic"] for x in load_family(f)}; assert set(ctxs) <= set(hdr), "missing headers"
    P = {}   # (id, order) -> {model: (pA, pB)} un-normalised next-token probabilities

    # base model: 87 contexts, arithmetic + geometric mixtures, soft prefix
    tok = AutoTokenizer.from_pretrained(a.base); model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map=a.device).eval(); AB = ab_ids(tok)
    print(f"{len(sc)} scenarios x 2 orders; A ids {AB['A']} B ids {AB['B']}; geometric variant {gkey}; arithmetic weights {dict((n, round(x, 3)) for n, x in zip(ctxs, w) if x > 0)}", flush=True)
    PF = (" " + a.prefill) if a.prefill else ""   # base/soft prefill after "Assistant:"
    BOS = [tok.bos_token_id] if tok.bos_token_id is not None and tok("a")["input_ids"][0] == tok.bos_token_id else []   # Llama only
    soft = torch.load(a.soft, map_location="cpu", weights_only=True).to(dev, torch.bfloat16); E = model.get_input_embeddings(); t0 = time.time()
    pr = lambda z: (torch.softmax(z, -1)[..., AB["A"]].sum(-1), torch.softmax(z, -1)[..., AB["B"]].sum(-1))
    for k, (key, q) in enumerate(prompts.items()):
        seqs = [tok(full_prompt(hdr[n], q) + PF, add_special_tokens=True)["input_ids"] for n in ctxs]; Z = []   # as in the fits: adds Llama's begin-of-text token; nothing for Olmo/Qwen (checked identical)
        for b in range(0, len(seqs), a.batch):
            ids, m = pad(seqs[b:b + a.batch], dev); Z.append(last_logits(model, input_ids=ids, mask=m))
        Z = torch.cat(Z); pa, pb = pr(Z); r = {n: (float(pa[i]), float(pb[i])) for i, n in enumerate(ctxs)}
        wt = torch.tensor(w, dtype=torch.float32, device=dev); r["arithmetic"] = (float((wt * pa).sum()), float((wt * pb).sum()))
        ga, gb = pr((c[:, None] * Z).sum(0)); r["geometric"] = (float(ga), float(gb))
        qi = tok(f"User: {q}\nAssistant:" + PF, add_special_tokens=False)["input_ids"]
        # BOS before the soft vectors, as in prompt_tune_base.SoftPrefix (Llama only)
        emb = torch.cat(([E(torch.tensor(BOS, device=dev))] if BOS else []) + [soft, E(torch.tensor(qi, device=dev))])[None]; sa, sb = pr(last_logits(model, embeds=emb, mask=torch.ones(1, emb.shape[1], dtype=torch.long, device=dev))[0])
        r["soft"] = (float(sa), float(sb)); P[key] = r
        if k % 20 == 0: print(f"  base {k}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)
    del model; torch.cuda.empty_cache()
    # Instruct, chat template
    tok = AutoTokenizer.from_pretrained(a.instruct); model = AutoModelForCausalLM.from_pretrained(a.instruct, dtype=torch.bfloat16, device_map=a.device).eval(); AB = ab_ids(tok)
    keys = list(prompts)
    for b in range(0, len(keys), a.batch):
        seqs = [tok.apply_chat_template([{"role": "user", "content": prompts[k]}], add_generation_prompt=True, tokenize=False) + a.prefill for k in keys[b:b + a.batch]]
        ids, m = pad([tok(s, add_special_tokens=False)["input_ids"] for s in seqs], dev, tok.pad_token_id or 0); pa, pb = pr(last_logits(model, input_ids=ids, mask=m))
        for i, k in enumerate(keys[b:b + a.batch]): P[k]["instruct"] = (float(pa[i]), float(pb[i]))

    # per scenario: p(action1) averaged over the two orders, per model
    models = ["instruct", "arithmetic", "geometric", "soft"] + ctxs
    rows = []
    for s in sc:
        x, y = P[(s["id"], "a1_first")], P[(s["id"], "a2_first")]
        for mdl in models:
            p1f = x[mdl][0] / (x[mdl][0] + x[mdl][1]); p1s = y[mdl][1] / (y[mdl][0] + y[mdl][1])
            rows.append({"id": s["id"], "ambiguity": s["ambiguity"], "model": mdl, "p1": (p1f + p1s) / 2, "p1_a1first": p1f, "p1_a2first": p1s,
                         "ab_mass": (sum(x[mdl]) + sum(y[mdl])) / 2, "pA_norm": (x[mdl][0] / sum(x[mdl]) + y[mdl][0] / sum(y[mdl])) / 2})
    df = pd.DataFrame(rows); inst = df[df.model == "instruct"].set_index("id")["p1"]
    summ = {}
    for mdl in models:
        d = df[df.model == mdl].set_index("id"); o = {}
        for lvl in LV:
            dl = d[d.ambiguity == lvl]; il = inst[dl.index]
            o[lvl] = {"mean_p1": float(dl.p1.mean()), "mean_abs_diff_vs_instruct": float((dl.p1 - il).abs().mean()),
                      "pearson_vs_instruct": float(np.corrcoef(dl.p1, il)[0, 1]) if dl.p1.std() > 0 and il.std() > 0 else None,
                      "argmax_agreement": float(((dl.p1 > 0.5) == (il > 0.5)).mean()), "order_inconsistency": float((dl.p1_a1first - dl.p1_a2first).abs().mean()),
                      "ab_mass": float(dl.ab_mass.mean()), "position_bias_pA": float(dl.pA_norm.mean())}
        summ[mdl] = o
    out = {"settings": vars(a), "n_scenarios": len(sc), "geometric_variant": gkey, "summary": summ, "per_scenario": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1))
    singles = [n for n in ctxs]
    for lvl in LV:
        print(f"\n{lvl} ambiguity ({sum(x['ambiguity'] == lvl for x in sc)} scenarios): model | mean P(action1) | mean |diff| vs Instruct | Pearson | argmax agree | order inconsistency | A/B mass | P(A)")
        for mdl in ["instruct", "arithmetic", "geometric", "soft", "generic", "hhh", "e14"]:
            o = summ[mdl][lvl]; pc = "  -  " if o["pearson_vs_instruct"] is None else f"{o['pearson_vs_instruct']:+.2f}"
            print(f"  {mdl:10s} {o['mean_p1']:.3f} | {o['mean_abs_diff_vs_instruct']:.3f} | {pc} | {o['argmax_agreement']:.2f} | {o['order_inconsistency']:.3f} | {o['ab_mass']:.3f} | {o['position_bias_pA']:.2f}")
        diffs = sorted(summ[n][lvl]["mean_abs_diff_vs_instruct"] for n in singles)
        print(f"  single contexts (87): mean |diff| vs Instruct best {diffs[0]:.3f} (oracle, chosen on this data), median {diffs[len(diffs)//2]:.3f}")
    print("->", a.out)


if __name__ == "__main__":
    main()
