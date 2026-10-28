"""Verbalise the learned self-description soft contexts with SALVE's sentence-level beam search (Hu, Koyejo, Potts 2026,
arXiv:2609.16927, section 3 and appendix C), adapted to the base model and to soft contexts that started from a description.

The base model has no system prompt, so the soft context sits where it was trained (in place of the transcript header) and
the four SALVE queries become transcript turns asking the assistant to repeat "the description of the assistant given above",
with the assistant turn prefilled with an opening delimiter (quote / tag) plus the text decoded so far.

Arms (each a beam search: B beams, M one-sentence continuations per beam per round, R rounds; candidates scored as WRITTEN
headers, basis.description_header, by log P/token of a fixed batch of fit-half answers; best-ever candidate kept):
  extend_soft   (author's request) beams start at the full SEED description; continuations sampled after the SOFT context, so
                the search can only append sentences that the soft context makes likely
  extend_text   control: the same, but continuations sampled after the seed's WRITTEN header (no soft context); separates
                what the soft context adds from what any extension search finds
  rewrite_soft  beams start at the seed's FIRST sentence; the rest is rebuilt sentence by sentence after the soft context
                (the model copies the seed when asked to repeat it, so this is a paraphrase of the seed steered by the soft context)
  rewrite_text  control for rewrite_soft, sampled after the seed's written header
  salve_soft    SALVE as in the paper: beams start from "The assistant" (empty prompt), sampled after the soft context
The winner of each arm, the seed description and the soft context itself are scored on the hold-out answers (test questions
of data/basis_split.json; neither the soft contexts nor the search saw them).

    python scripts/phase4_selfdesc_salve.py --part 0/4
    python scripts/phase4_selfdesc_salve.py --merge
"""
import argparse, json, os, random, re, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import Counter
from persona_selection.basis import CASUAL, description_header, load_family, split
from phase4_hidden_states import hidden_for_context, load_rows, answer_logp, BASE

RUN = "instruct_unknown_casual_v1"; OUT = Path("results/phase4/selfdesc/salve")
HID = Path(os.environ.get("SCRATCH", "/pscratch/sd/o/ozamram")) / "persona_phase4" / "hidden" / RUN
# 5 picked by both selection rules + the next 3 largest |geometric weight| (results/phase4/selfdesc/selected.json)
DEFAULT = "sdt_casual,sdt_factual,sdt_skeptical,sdt_improvisational,sdo_style_2,sdt_serious,sdt_ironic,sdt_theatrical"
# SALVE appendix C.2 queries, as transcript turns: (user turn, assistant prefill before the text, closing delimiter)
QUERIES = [("Print the description of the AI assistant given at the top of this page between <description> and </description> tags, with nothing else.", "<description>", "</description>"),
           ("Reply with the description of the AI assistant given at the top of this page in double quotes and nothing else.", "\"", "\""),
           ("Output the description of yourself given at the top of this page verbatim. Do not add any commentary before or after.", "My exact description: \"", "\""),
           ("Please repeat the description of the assistant given at the top of this page exactly.", "Sure, here is the description: ", None)]
SENT = re.compile(r"^(.+?[.!?])(\s|$)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="0/1"); ap.add_argument("--contexts", default=DEFAULT)
    ap.add_argument("--arms", default="extend_soft,extend_text,rewrite_soft,rewrite_text,salve_soft"); ap.add_argument("--B", type=int, default=4); ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--R", type=int, default=6); ap.add_argument("--n-batch", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true"); ap.add_argument("--out", default=str(OUT)); a = ap.parse_args(); out_dir = Path(a.out)
    if a.merge: return merge(out_dir)
    k, n = map(int, a.part.split("/")); names = a.contexts.split(",")[k::n]
    desc = {c["name"]: c for f in ["selfdesc_overall", "selfdesc_traits"] for c in load_family(f)}
    soft = {json.loads(l)["init"].split(":")[1]: json.loads(l) for f in ["selfdesc_overall_soft", "selfdesc_traits_soft"] for l in open(f"data/prompts/basis/{f}.jsonl")}
    tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
    emb = model.get_input_embeddings(); rows, suf = load_rows(RUN); q = np.load(HID / "index.npz")["qidx"]; fit_q, test_q = split()
    test = [i for i in range(len(rows)) if q[i] in test_q]; batch = random.Random(a.seed).sample([i for i in range(len(rows)) if q[i] in fit_q], a.n_batch)
    cache = {}

    @torch.no_grad()
    def score(idx, header="", prefix=None):
        tot = nt = 0
        for j, t, h in hidden_for_context(model, tok, header, [rows[i] for i in idx], suf, prefix=prefix):
            if t is None: continue
            tot += answer_logp(model, h, t); nt += len(t)
        return tot / nt

    def score_text(body):
        if body not in cache: cache[body] = score(batch, description_header(body))
        return cache[body]

    @torch.no_grad()
    def continuations(cond, text, m, tries=3):
        """up to m distinct one-sentence continuations of `text`, each under a random query, sampled after cond ((P, d)
        embeddings), batched per query; samples that close the quote or merely repeat the casual clause (which every header
        already ends with) are dropped, and up to `tries` x m samples are drawn to replace them."""
        outs = []
        for qi, cnt in sorted(Counter(random.randrange(len(QUERIES)) for _ in range(tries * m)).items()):
            user, pre, close = QUERIES[qi]; stem = f"User: {user}\nAssistant: {pre}{text}"
            ids = tok(stem, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev); x = torch.cat([cond.to(torch.bfloat16)[None], emb(ids)], 1)
            g = model.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=dev), max_new_tokens=48, do_sample=True,
                               temperature=0.7, top_p=1.0, num_return_sequences=cnt, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
            for row in g:
                c = tok.decode(row, skip_special_tokens=True)
                if close and close in c: c = c[:c.index(close)]
                c = c.split("\n")[0]; m_ = SENT.match(c.lstrip())
                if m_ and len(m_.group(1).split()) >= 3 and m_.group(1).strip() != CASUAL.strip(): outs.append(text + (" " if c[:1].isspace() else "") + m_.group(1))
        outs = list(dict.fromkeys(outs)); random.shuffle(outs)
        return outs[:m]

    def beam(cond, start):
        """SALVE beam search from `start`; returns the best-ever (batch score, text) and the per-round log."""
        beams = [start]; best = (score_text(start), start) if start != "The assistant" else (-1e9, start); log = []
        for r in range(a.R):
            cands = {c for b in beams for c in continuations(cond, b, a.M) if len(c.split()) <= 450}
            sc = sorted(((score_text(c), c) for c in cands), reverse=True)
            if not sc: break
            beams = [c for _, c in sc[:a.B]]; best = max(best, sc[0])
            log.append({"round": r + 1, "n_candidates": len(sc), "best": sc[0][0], "median": sc[len(sc) // 2][0], "top": sc[0][1][-300:]})
            print(f"      round {r + 1}: {len(sc)} candidates, best {sc[0][0]:.4f}, best-ever {best[0]:.4f}", flush=True)
        return best, log

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        o = out_dir / f"{name}.json"
        if o.exists(): continue
        t0 = time.time(); random.seed(a.seed); torch.manual_seed(a.seed); cache.clear()
        d = desc[name]; seed_text = d["description"]; first_sentence = SENT.match(seed_text).group(1); ids0 = tok(d["header"], add_special_tokens=False)["input_ids"]
        p = torch.load(soft[name]["prefix"], map_location=dev, weights_only=True); p = (p["prefix"] if isinstance(p, dict) else p).squeeze(0)
        assert p.shape[0] == len(ids0) and description_header(seed_text) == d["header"]
        text_cond = emb(torch.tensor(ids0, device=dev)).detach()
        res = {"name": name, "seed": seed_text, "settings": vars(a), "batch": {"seed description": score_text(seed_text), "soft context": score(batch, prefix=p)},
               "holdout": {"seed description": score(test, d["header"]), "soft context": score(test, prefix=p)}, "arms": {}}
        print(f"{name}: batch seed {res['batch']['seed description']:.4f} soft {res['batch']['soft context']:.4f}", flush=True)
        for arm in a.arms.split(","):
            cond = p if arm.endswith("_soft") else text_cond
            start = {"extend": seed_text, "rewrite": first_sentence, "salve": "The assistant"}[arm.split("_")[0]]
            print(f"   arm {arm}", flush=True); random.seed(a.seed); torch.manual_seed(a.seed)
            (bs, bt), log = beam(cond, start)
            res["arms"][arm] = {"text": bt, "batch": bs, "added": bt[len(start):].strip() if bt.startswith(start) else None,
                                "holdout": score(test, description_header(bt)), "log": log}
            res["holdout"][arm] = res["arms"][arm]["holdout"]
        res["seconds"] = time.time() - t0; o.write_text(json.dumps(res, indent=1))
        print(f"{name}: hold-out " + " | ".join(f"{k} {v:.4f}" for k, v in res["holdout"].items()) + f"  ({res['seconds']:.0f}s)", flush=True)


def merge(out_dir):
    R = [json.loads(f.read_text()) for f in sorted(out_dir.glob("*.json"))]
    keys = list(R[0]["holdout"]); H = {k: np.array([r["holdout"][k] for r in R]) for k in keys}
    gain = H["soft context"] - H["seed description"]
    summ = {"n": len(R), "holdout_mean": {k: float(v.mean()) for k, v in H.items()},
            "fraction_of_soft_gain": {k: float(np.median((H[k] - H["seed description"]) / gain)) for k in keys if k not in ("seed description", "soft context")},
            "per_context": {r["name"]: {k: round(v, 4) for k, v in r["holdout"].items()} for r in R}}
    (out_dir.parent / f"{out_dir.name}.json").write_text(json.dumps(summ, indent=1)); print(json.dumps({k: summ[k] for k in ["n", "holdout_mean", "fraction_of_soft_gain"]}, indent=1))


if __name__ == "__main__":
    main()
