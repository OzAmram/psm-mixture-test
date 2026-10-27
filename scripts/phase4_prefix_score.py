"""Score a run's answers under a model + learned soft prefix with the STANDARD Phase 1 convention: the prompt
"User: <question><suffix>\\nAssistant:" and the answer tokenised TOGETHER, prompt tokens masked, the stored answer text
exactly as is. The prefix vectors replace the header (as in training; no BOS, as in prompt_tune_base.py).
Writes <out>.npz with lp (N,) total log P per answer, n_tokens (N,), qidx (N,), and prints the hold-out log P/token on
the basis split.

    python scripts/phase4_prefix_score.py --prefix results/phase4/exp1/prefix_instruct_L128/prefix.pt \\
        --run instruct_unknown_casual_v1 --out results/phase4/exp1/prefix_instruct_L128/scores
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import split


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B"); ap.add_argument("--out", required=True); a = ap.parse_args()
    D = Path(a.run) if Path(a.run).is_dir() else Path("results/phase1") / a.run; suf = json.load(open(D / "config.json"))["args"]["user_suffix"]
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
    emb = model.get_input_embeddings(); pre = torch.load(a.prefix, weights_only=True).to(dev, emb.weight.dtype)[None]
    rows = [json.loads(l) for l in open(D / "rows.jsonl")]; lp, nt = [], []
    for r in rows:
        p = f"User: {r['question']}{suf}\nAssistant:"
        p_ids = tok(p, add_special_tokens=False)["input_ids"]; full = tok(p + r["response"], add_special_tokens=False)["input_ids"]
        if full[:len(p_ids)] != p_ids: lp.append(np.nan); nt.append(0); continue
        x = torch.cat([pre, emb(torch.tensor([full], device=dev))], 1)
        z = model(inputs_embeds=x).logits[0, pre.shape[1] + len(p_ids) - 1:-1].float()
        tgt = torch.tensor(full[len(p_ids):], device=dev)
        lp.append(float((z.gather(-1, tgt[:, None])[:, 0] - torch.logsumexp(z, -1)).sum())); nt.append(len(tgt))
    lp, nt, q = np.array(lp), np.array(nt), np.array([r["qidx"] for r in rows])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); np.savez(a.out + ".npz", lp=lp, n_tokens=nt, qidx=q, prefix=a.prefix, model=a.model)
    _, test_q = split(); te = np.isin(q, list(test_q)) & np.isfinite(lp)
    print(f"{a.prefix} on {a.run}: hold-out log P/token {lp[te].sum()/nt[te].sum():.4f} ({int(te.sum())} answers); NaN {int(np.isnan(lp).sum())}")


if __name__ == "__main__":
    main()
