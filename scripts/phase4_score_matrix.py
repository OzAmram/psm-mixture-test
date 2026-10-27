"""Experiment 2 score matrices: X_M[s, j] = total log P_M(answer_j | header s, question_j) in the raw transcript format
(full_prompt(header, question + suffix), answer tokenised together with the prompt, prompt masked), so every model
scores identical text. Rows = contexts from the listed families; columns = results/phase4/exp2/columns.jsonl.
Different tokenizers are fine: entries are log-probabilities of the same text.

    python scripts/phase4_score_matrix.py --model allenai/Olmo-3-1025-7B --tag base
Writes results/phase4/exp2/X_<tag>.npz with X (S, C), names (S), n_tokens (C).
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.batch_scoring import score_batch
from persona_selection.basis import load_family, full_prompt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--families", default="generic,legacy86,elicited_p4"); ap.add_argument("--columns", default="results/phase4/exp2/columns.jsonl")
    ap.add_argument("--max-tokens-per-batch", type=int, default=24000); a = ap.parse_args()
    out = Path(a.columns).parent / f"X_{a.tag}.npz"; part = out.with_suffix(".partial.npz")
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    cols = [json.loads(l) for l in open(a.columns)]; comps = [c for f in a.families.split(",") for c in load_family(f)]
    X = np.full((len(comps), len(cols)), np.nan); nt = None; start = 0
    if part.exists():   # resume
        z = np.load(part); X[: int(z["done"])] = z["X"][: int(z["done"])]; nt = z["n_tokens"]; start = int(z["done"]); print(f"resuming at row {start}")
    t0 = time.time()
    for k in range(start, len(comps)):
        lp, n = score_batch(model, tok, [(full_prompt(comps[k]["header"], c["question"] + c["suffix"]), c["response"]) for c in cols], max_tokens_per_batch=a.max_tokens_per_batch)
        X[k] = [np.nan if x is None else x[0] for x in lp]; nt = np.array(n) if nt is None else nt
        if (k + 1) % 20 == 0 or k + 1 == len(comps):
            np.savez(part, X=X, n_tokens=nt, done=k + 1); print(f"[{a.tag}] {k+1}/{len(comps)} rows, {time.time()-t0:.0f}s", flush=True)
    np.savez(out, X=X, names=np.array([c["name"] for c in comps]), n_tokens=nt, model=a.model)
    part.unlink(missing_ok=True); print(f"done: X {X.shape}, NaN {int(np.isnan(X).sum())} -> {out}")


if __name__ == "__main__":
    main()
