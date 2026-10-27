"""Score a Phase 1 run's answers under the BASE model with every component of one or more basis families
(data/prompts/basis/<family>.jsonl), batched. Writes results/phase1/<run>/basis/<family>.npz with
    L (N, K) log P(answer | header_k), names (K), n_tokens (N), qidx (N), meta (N, bool: placeholder/meta answer)
Rows are in rows.jsonl order, so families can be concatenated column-wise. Existing family files are skipped unless --force.

    python scripts/phase1_basis_score.py --run instruct_unknown_casual_v1 --families legacy86,bigfive
    python scripts/phase1_basis_score.py --check     # batched vs unbatched scorer on a few answers

Family files whose components carry "per_question": true have a header that depends on the question (e.g. exemplar
components that must exclude the scored question); for those, "header" is a dict {qidx: header}.
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.batch_scoring import score_batch
from persona_selection.basis import load_family, full_prompt
from persona_selection.mixture import is_meta_response

BASE = "allenai/Olmo-3-1025-7B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--families", default=""); ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-tokens-per-batch", type=int, default=24000)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--screen", action="store_true", help="score only a screening subset (first 2 answers of every FIT-half question) -> basis/<family>__screen.npz")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    if a.check:
        from persona_selection.scoring import score_response
        from persona_selection.basis import legacy_header
        rows = [json.loads(l) for l in open("results/phase1/instruct_unknown_casual_v1/rows.jsonl")][:40]; suf = " Answer in two or three sentences of plain text, in a casual, conversational tone."
        pairs = [(full_prompt(legacy_header("hhh"), r["question"] + suf), r["response"]) for r in rows]
        lp, _ = score_batch(model, tok, pairs); ref = [score_response(model, tok, p, r)["logprob"] for p, r in pairs]
        d = np.array([x[0] for x in lp]) - np.array(ref); print(f"batched - unbatched: mean {d.mean():+.4f}, mean |d| {np.abs(d).mean():.4f}, max |d| {np.abs(d).max():.4f} nats/answer")
        return
    run = Path(a.run) if Path(a.run).is_dir() else Path("results/phase1") / a.run;   # a run name or a run folder
    cfg = json.loads((run / "config.json").read_text()); suf = cfg["args"].get("user_suffix", "") or ""
    rows = [json.loads(l) for l in open(run / "rows.jsonl")]; (run / "basis").mkdir(exist_ok=True)
    row_idx = np.arange(len(rows))
    if a.screen:
        from persona_selection.basis import split
        fit_q, _ = split(); seen = {}; keep = []
        for i, r in enumerate(rows):
            if r["qidx"] in fit_q and seen.get(r["qidx"], 0) < 2: keep.append(i); seen[r["qidx"]] = seen.get(r["qidx"], 0) + 1
        row_idx = np.array(keep); rows = [rows[i] for i in keep]
    for fam in [f for f in a.families.split(",") if f]:
        outf = run / "basis" / (f"{fam}__screen.npz" if a.screen else f"{fam}.npz")
        if outf.exists() and not a.force:
            print(f"[{a.run}] {fam}: exists, skipping"); continue
        comps = load_family(fam); t0 = time.time(); L = np.full((len(rows), len(comps)), np.nan); nt = None
        for k, c in enumerate(comps):
            hdr = (lambda r: c["header"][str(r["qidx"])]) if c.get("per_question") else (lambda r: c["header"])
            lp, n = score_batch(model, tok, [(full_prompt(hdr(r), r["question"] + suf), r["response"]) for r in rows], max_tokens_per_batch=a.max_tokens_per_batch)
            L[:, k] = [np.nan if x is None else x[0] for x in lp]
            if nt is None: nt = np.array(n)
        np.savez(outf, L=L, names=np.array([c["name"] for c in comps]), n_tokens=nt, qidx=np.array([r["qidx"] for r in rows]),
                 meta=np.array([is_meta_response(r["response"]) for r in rows]), row_idx=row_idx)
        print(f"[{a.run}] {fam}: {len(comps)} components x {len(rows)} answers in {time.time()-t0:.0f}s, nan {int(np.isnan(L).sum())} -> {outf}", flush=True)


if __name__ == "__main__":
    main()
