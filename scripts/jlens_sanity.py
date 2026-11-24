"""Sanity check of the pre-fitted Neuronpedia J-Lens for allenai/Olmo-3-1025-7B (2026-11-18) before using it on soft contexts:
on the latent-variable items of anthropics/jacobian-lens (copied in safety-research/jpp_lens, data/jlens/evaluations), is the
intermediate (e.g. "Brazil" for "the ocean on the coast of the country where Carnival is ...") in the top 10 readout tokens at
any of 7 evenly spaced layers, last prompt position, full vocabulary; J-Lens vs logit lens. ROUGH matching (a readout token,
stripped and lower-cased, of >= 3 characters that equals or starts the intermediate word), not the paper's scorer, and no
filter on items the model answers correctly. --strip: remove the prompts' trailing space (Olmo's tokenizer makes it a lone
" " token, after which the model predicts e.g. "3" instead of "Atlantic"). Runs on CPU in the jpp_lens environment:

    CUDA_VISIBLE_DEVICES= HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 /workspace/envs/jpp/bin/python scripts/jlens_sanity.py
"""
import dataclasses, json, sys, time
from pathlib import Path
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from workspace_lens.lenses.logit_lens import LogitLens

M = "allenai/Olmo-3-1025-7B"
LENS = "/workspace/lenses/neuronpedia/olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt"
EVAL = Path("/workspace/jpp_lens/data/jlens/evaluations")
LAYERS = [4, 8, 12, 16, 20, 24, 28]
STRIP = "--strip" in sys.argv; TAG = "_strip" if STRIP else ""
t.set_num_threads(64)

jl = load_lens_file(LENS, hf_model_name=M); ll = LogitLens(jl.source_layers, config=dataclasses.replace(jl.config, lens_type="logit"))
model = get_hf_model(M, attn_implementation="sdpa", device_map="cpu"); tok = model.tokenizer
_, ml, _ = ll.apply(model, "The capital of France is", layers=[28], token_positions_for_residuals=[-1])
print("model check, 'The capital of France is' ->", [tok.decode([i]) for i in ml[0].topk(3).indices.tolist()], flush=True)
hit = lambda toks, w: any(len(s) >= 3 and (s == w or w.startswith(s)) for s in (x.strip().lower() for x in toks))
out = {}
for task in ["multihop", "multilingual"]:
    items = json.loads((EVAL / f"lens-eval-{task}.json").read_text())["items"]; res = []
    for it in items:
        row = {"name": it["name"], "intermediates": it["intermediates"]}
        for name, lens in [("jlens", jl), ("logit", ll)]:
            logits, model_logits, _ = lens.apply(model, it["prompt"].rstrip(" ") if STRIP else it["prompt"], layers=LAYERS, token_positions_for_residuals=[-1])
            tops = {L: [tok.decode([i]) for i in v[0].topk(10).indices.tolist()] for L, v in logits.items()}
            row[name] = any(hit(tops[L], w.lower()) for L in LAYERS for w in it["intermediates"])
            row[f"{name}_top"] = {L: tops[L][:5] for L in LAYERS}
        row["model_top1"] = tok.decode([model_logits[0].argmax().item()]); res.append(row)
    n = len(res); out[task] = {"n": n, "jlens_recall10": sum(r["jlens"] for r in res) / n, "logit_recall10": sum(r["logit"] for r in res) / n, "items": res}
    tgt = {it2["name"]: it2["target"].lower() for it2 in items}
    out[task]["model_correct"] = sum(bool(r["model_top1"].strip()) and tgt[r["name"]].startswith(r["model_top1"].strip().lower()) for r in res) / n
    print(f"{task}: model top-1 starts the target in {out[task]['model_correct']:.0%};", end=" ")
    print(f"{task}: {n} items; intermediate in top 10 at any of {LAYERS}: J-Lens {out[task]['jlens_recall10']:.0%}, logit lens {out[task]['logit_recall10']:.0%}", flush=True)
    for r in res[:3]: print("  ", r["name"], r["intermediates"], "| J-Lens L16:", r["jlens_top"][16], "| logit L16:", r["logit_top"][16], "| model:", repr(r["model_top1"]))
Path("results/phase4/jlens").mkdir(parents=True, exist_ok=True); Path(f"results/phase4/jlens/sanity{TAG}.json").write_text(json.dumps(out, indent=1))
