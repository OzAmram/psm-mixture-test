"""Read the learned soft contexts with the J++ Lens fitted for allenai/Olmo-3-1025-7B (2026-11-19; author's request).

For each soft context (a prefix.pt from prompt_tune_base.py, trained on Olmo 3 Instruct's answers) and its starting TEXT
(the same header as token embeddings: the starting point of training, which reproduces the text header exactly), two
readouts at the lens layers (4, 8, ..., 28), with Readout Filtering (tokens with no letter or digit left out, as in J++):

1. prefix positions: the base model run on the prefix alone (causal, so these residuals are question-independent); per
   layer, how often each token is in the top 10 of the lens readout across positions, soft vs text, and the tokens whose
   count rises most. Positions < 16 are reported separately (the lens was fitted skipping the first 16 positions).
2. answer positions: N fixed hold-out Instruct answers (first N of the hold-out questions' rows, file order), the exact
   input format of training ([prefix] User: <question><suffix>\\nAssistant: <answer>); at the positions that predict
   answer tokens, mean over positions of (lens probability with soft context - with text), per layer: the tokens the
   learned correction makes the model more/less disposed to say. Ground truth: the same difference of the model's own
   output probabilities. Per layer, Pearson correlation (over the filtered vocabulary) and top-50 overlap of lens vs
   ground truth. The mean change of LOG-probabilities is also kept ("*_logp"), but it is dominated by rare tokens whose
   log-probability moves a lot while their probability stays negligible (2026-11-20, first run), so it is secondary.

    PYTHONPATH=/workspace/jpp_lens/src:src:scripts /workspace/envs/jpp/bin/python scripts/jpp_softcontext_readout.py \\
        --contexts selfdesc --n-answers 200 --out results/phase4/jlens/softcontext_selfdesc.json
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/workspace/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from workspace_lens.lenses.base_lens import BaseLens
from lens_evals.readout_evals.readout_eval_items import non_semantic_token_ids
from persona_selection.basis import load_family
from persona_selection.phase1_prompts import generic_prompt
from prompt_tune_base import build, SUFFIX, SoftPrefix, eval_nll

M = "allenai/Olmo-3-1025-7B"; LENS = "/workspace/lenses/jpp_olmo3_7b/jpp_lens.pt"
ROWS = "results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl"; HOLD = "results/phase4/exp1/basis_test_qids.json"
GENERIC = generic_prompt("", framing="unknown", register="casual").split("User:")[0].strip()   # the training init text


def contexts(which):
    """[(name, prefix.pt path, starting text)]"""
    out = []
    if which in ("selfdesc", "all"):
        hdr = {c["name"]: c["header"] for f in ["selfdesc_overall", "selfdesc_traits"] for c in load_family(f)}
        for d in sorted(Path("results/phase4/selfdesc/prefix").iterdir()):
            if (d / "result.json").exists():
                a = json.loads((d / "result.json").read_text())["args"]
                out.append((f"selfdesc/{d.name}", d / "prefix.pt", hdr[a["init_from"].split(":")[1]] if a.get("init_from") else GENERIC))
    if which in ("generic", "all"):
        for d in [Path("results/phase4/exp3/prefix/final_base_L128_lr5e-3")] + [Path(f"results/phase4/lensweep/L{L}_lr5e-3_s0") for L in (16, 64, 128, 1024)]:
            if (d / "result.json").exists(): out.append((f"{d.parent.name}/{d.name}", d / "prefix.pt", GENERIC))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--contexts", default="all", choices=["selfdesc", "generic", "all"])
    ap.add_argument("--n-answers", type=int, default=200); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--top", type=int, default=25); ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default="text", choices=["text", "init"], help="text: the written starting text (default); init: the exact "
                    "initial prefix of training (starting text's tokens, then the random tokens drawn with the run's seed), rebuilt and "
                    "checked against the run's logged 'prefix at init' hold-out score (--expect-init)")
    ap.add_argument("--compare-header", default=None, help="family:name of a written persona header to compare against instead of the "
                    "starting text (e.g. legacy86:hhh, Askell's HHH description); implies --compare text")
    ap.add_argument("--pair", default=None, help="A,B with A and B family:name of written headers (e.g. legacy86:hhh,generic:generic): a control "
                    "that runs the same readout with header A in place of the soft context and header B as the comparison, both exactly as in the fits")
    ap.add_argument("--only", default=None, help="comma-separated context names to run (e.g. prefix/final_base_L128_lr5e-3)")
    ap.add_argument("--expect-init", type=float, default=None, help="logged hold-out log P/token of the initial prefix; abort if the rebuild differs by > 5e-4")
    a = ap.parse_args(); dev = torch.device(a.device)
    tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=a.device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    lens = BaseLens.load(LENS); LAYERS = list(lens.source_layers); V = model.lm_head.weight.shape[0]
    keep = torch.ones(V, dtype=torch.bool); keep[non_semantic_token_ids(tok, vocab_size=V, items=[])] = False; keep = keep.to(dev)   # items=[]: no eval items to spare here
    E = model.get_input_embeddings()
    rec = {}
    hooks = [model.model.layers[l].register_forward_hook(lambda m, i, o, l=l: rec.__setitem__(l, o if torch.is_tensor(o) else o[0])) for l in LAYERS]
    final = model.config.num_hidden_layers - 1

    def lens_lp(h, l):   # filtered lens log-softmax, [..., V] fp32
        z = model.lm_head(model.model.norm(lens.transport(h.float(), l).to(torch.bfloat16))).float()
        return torch.log_softmax(z.masked_fill(~keep, float("-inf")), -1)

    hold = set(json.load(open(HOLD))); rows = [r for r in map(json.loads, open(ROWS)) if r["qid"] in hold][: a.n_answers]
    data = [build(tok, r["question"], r["response"], 192) for r in rows]
    hdr_of = lambda fn: {c["name"]: c["header"] for c in load_family(fn.split(":")[0])}[fn.split(":")[1]]
    ctxs = [(f"pair/{a.pair.replace(',', '_vs_')}", None, None)] if a.pair else contexts(a.contexts)[: a.limit or None]
    if a.only: ctxs = [c for c in ctxs if c[0] in a.only.split(",")]; assert ctxs, "no context matches --only"
    res = {"layers": LAYERS, "n_answers": len(data), "lens": LENS, "compare": a.compare, "contexts": {}}
    print(f"{len(ctxs)} contexts, {len(data)} hold-out answers, layers {LAYERS}", flush=True)
    for name, pf, text in ctxs:
        t0 = time.time()
        if a.pair:   # written header A plays the soft context's role, header B the comparison
            soft = E(torch.tensor(tok(hdr_of(a.pair.split(",")[0]), add_special_tokens=False)["input_ids"], device=dev)); text = hdr_of(a.pair.split(",")[1])
        else: soft = torch.load(pf, map_location="cpu", weights_only=True).to(dev, torch.bfloat16)
        if a.compare_header:
            fam, hn = a.compare_header.split(":"); text = {c["name"]: c["header"] for c in load_family(fam)}[hn]   # as in the fits: full_prompt(header, q) = header + "User: ..."
        txt = E(torch.tensor(tok(text, add_special_tokens=False)["input_ids"], device=dev))
        out = {"L": int(soft.shape[0]), "text_tokens": int(txt.shape[0]), "compare_text": text}
        if a.compare == "init":   # the exact starting prefix: SoftPrefix draws its random tokens right after torch.manual_seed(seed) in training
            ra = json.loads((Path(pf).parent / "result.json").read_text())["args"]; torch.manual_seed(ra["seed"])
            sp = SoftPrefix(model, tok, int(soft.shape[0]), text).to(dev)
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            hold_all = [build(tok, r["question"], r["response"], ra["max_len"]) for r in map(json.loads, open(ra["rows"])) if r["qid"] in set(json.load(open(ra["holdout_qids"])))]
            for h in hooks: h.remove()
            with torch.no_grad(): tot, n = eval_nll(lambda ids, att, lab: sp(ids, att, lab).logits[:, sp.n_pre:], hold_all, pad_id, dev)
            hooks[:] = [model.model.layers[l].register_forward_hook(lambda m, i, o, l=l: rec.__setitem__(l, o if torch.is_tensor(o) else o[0])) for l in LAYERS]
            out["init_holdout_logp"] = -tot / n; print(f"{name}: rebuilt initial prefix, hold-out log P/token {-tot/n:.4f} (logged: {a.expect_init})", flush=True)
            if a.expect_init is not None: assert abs(-tot / n - a.expect_init) < 5e-4, "rebuilt initial prefix does not reproduce the logged score"
            txt = sp.prefix.detach().to(torch.bfloat16); out["text_tokens"] = int(txt.shape[0])
        # 1. prefix positions
        pos = {}
        for cond, pre in [("soft", soft), ("text", txt)]:
            rec.clear()
            with torch.no_grad(): model(inputs_embeds=pre[None])
            pos[cond] = {l: lens_lp(rec[l][0], l).topk(10, -1).indices.cpu() for l in LAYERS}
        out["prefix_positions"] = {}
        for l in LAYERS:
            o = {}
            for part, sl in [("pos>=16", slice(16, None)), ("pos<16", slice(0, 16))]:
                cnt = {c: torch.bincount(pos[c][l][sl].flatten(), minlength=V) for c in ("soft", "text")}
                if cnt["soft"].sum() == 0: continue
                gain = (cnt["soft"] - cnt["text"]).float()
                o[part] = {"soft_top": [(tok.decode([i]), int(cnt["soft"][i])) for i in cnt["soft"].topk(a.top).indices.tolist()],
                           "gained": [(tok.decode([i]), int(gain[i])) for i in gain.topk(a.top).indices.tolist()],
                           "lost": [(tok.decode([i]), int(gain[i])) for i in (-gain).topk(a.top).indices.tolist()]}
            out["prefix_positions"][l] = o
        # 2. answer positions: running sums of (soft - text) filtered log-softmax per layer, and of the model's own output
        S = {l: torch.zeros(V, device=dev) for l in LAYERS}; S_true = torch.zeros(V, device=dev); npos = 0
        Q = {l: torch.zeros(V, device=dev) for l in LAYERS}; Q_true = torch.zeros(V, device=dev)   # probability differences
        for b in range(0, len(data), a.batch):
            chunk = data[b:b + a.batch]; acc = {}
            for cond, pre in [("soft", soft), ("text", txt)]:
                n_pre = pre.shape[0]; T = n_pre + max(len(x[0]) for x in chunk)
                x = torch.zeros(len(chunk), T, pre.shape[1], dtype=torch.bfloat16, device=dev); am = torch.zeros(len(chunk), T, dtype=torch.long, device=dev)
                sel = []   # (row, sequence position predicting an answer token)
                for i, (ids, lab, _) in enumerate(chunk):
                    x[i, :n_pre] = pre; x[i, n_pre:n_pre + len(ids)] = E(torch.tensor(ids, device=dev)); am[i, :n_pre + len(ids)] = 1
                    sel += [(i, n_pre + j - 1) for j, y in enumerate(lab) if y != -100]
                rec.clear()
                with torch.no_grad(): logits = model(inputs_embeds=x, attention_mask=am).logits
                ri, pi = torch.tensor([s[0] for s in sel], device=dev), torch.tensor([s[1] for s in sel], device=dev)
                acc[cond] = ({l: rec[l][ri, pi] for l in LAYERS}, torch.log_softmax(logits[ri, pi].float().masked_fill(~keep, float("-inf")), -1))
            with torch.no_grad():
                for l in LAYERS:
                    ls, lt = lens_lp(acc["soft"][0][l], l), lens_lp(acc["text"][0][l], l)
                    S[l] += torch.nan_to_num(ls - lt, nan=0.0, posinf=0.0, neginf=0.0).sum(0); Q[l] += (ls.exp() - lt.exp()).sum(0)
                S_true += torch.nan_to_num(acc["soft"][1] - acc["text"][1], nan=0.0, posinf=0.0, neginf=0.0).sum(0); Q_true += (acc["soft"][1].exp() - acc["text"][1].exp()).sum(0)
            npos += len(acc["soft"][1])
        true = (S_true / npos)[keep].cpu().numpy(); kid = keep.nonzero().flatten().cpu(); true_p = (Q_true / npos)[keep].cpu().numpy()
        rank = lambda v: np.argsort(np.argsort(v))
        dec = lambda v, desc, nd=3: [(tok.decode([int(kid[i])]), round(float(v[i]), nd)) for i in (np.argsort(-v) if desc else np.argsort(v))[: a.top]]
        out["answer_positions"] = {"n_positions": npos, "model_output": {"up": dec(true_p, True, 5), "down": dec(true_p, False, 5),
                                   "up_logp": dec(true, True), "down_logp": dec(true, False)}, "layers": {}}
        top50 = set(np.argsort(-true_p)[:50]); top50_l = set(np.argsort(-true)[:50])
        for l in LAYERS:
            v = (S[l] / npos)[keep].cpu().numpy(); vp = (Q[l] / npos)[keep].cpu().numpy()
            out["answer_positions"]["layers"][l] = {"up": dec(vp, True, 5), "down": dec(vp, False, 5),
                "pearson_vs_output": float(np.corrcoef(vp, true_p)[0, 1]), "top50_overlap_vs_output": len(top50 & set(np.argsort(-vp)[:50])) / 50,
                "up_logp": dec(v, True), "down_logp": dec(v, False), "spearman_vs_output_logp": float(np.corrcoef(rank(v), rank(true))[0, 1]),
                "top50_overlap_vs_output_logp": len(top50_l & set(np.argsort(-v)[:50])) / 50}
        res["contexts"][name] = out
        sp = {l: round(out["answer_positions"]["layers"][l]["pearson_vs_output"], 2) for l in LAYERS}
        print(f"{name}: L={out['L']} ({time.time()-t0:.0f}s); output up (probability): {[w for w, _ in out['answer_positions']['model_output']['up'][:8]]}; "
              f"lens-vs-output Pearson {sp}; layer 16 lens up: {[w for w, _ in out['answer_positions']['layers'][16]['up'][:6]]}", flush=True)
    for h in hooks: h.remove()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1)); print("->", a.out)


if __name__ == "__main__":
    main()
