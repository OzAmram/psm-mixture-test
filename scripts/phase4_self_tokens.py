"""Per-token self log-probabilities of a run's answers under the model that wrote them (chat template), for the residual
analysis of Experiment 1 and the self-scores of Experiment 3.

Same convention as phase1_rescore_self.py (which gives the -0.898 reference): default system prompt, same user suffix,
the stripped answer text right after the assistant tag, answer tokens only, no end-of-turn token. Writes
results/phase4/self_tokens/<run>.npz with offsets (N+1,), lp (P,) per-token log P, tok (P,) token ids, qidx (N,), total (N,).

    python scripts/phase4_self_tokens.py --run instruct_unknown_casual_v1 --model allenai/Olmo-3-7B-Instruct
"""
import argparse, json, os
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/phase4/self_tokens"); a = ap.parse_args()
    D = Path("results/phase1") / a.run; SUF = json.load(open(D / "config.json"))["args"]["user_suffix"]
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
    rows = [json.loads(l) for l in open(D / "rows.jsonl")]; lps, toks, tot, nt = [], [], [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            p = tok.apply_chat_template([{"role": "user", "content": r["question"] + SUF}], add_generation_prompt=True, tokenize=False)
            p_ids = tok(p, add_special_tokens=False)["input_ids"]; full = tok(p + r["response"].strip(), add_special_tokens=False)["input_ids"]
            assert full[:len(p_ids)] == p_ids; r_ids = full[len(p_ids):]
            logits = model(input_ids=torch.tensor([full], device=dev)).logits[0, len(p_ids) - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(r_ids, device=dev).unsqueeze(-1))[:, 0].cpu().numpy()
            lps.append(lp); toks.append(np.array(r_ids)); tot.append(float(lp.sum())); nt.append(len(r_ids))
            if (i + 1) % 400 == 0: print(f"[{i+1}/{len(rows)}] self log P/token so far {sum(tot)/sum(nt):.4f}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.run}.npz", offsets=np.concatenate([[0], np.cumsum(nt)]), lp=np.concatenate(lps), tok=np.concatenate(toks),
             qidx=np.array([r["qidx"] for r in rows]), total=np.array(tot), model=a.model)
    print(f"done: {len(rows)} answers, self log P/token {sum(tot)/sum(nt):.4f} -> {out / (a.run + '.npz')}")


if __name__ == "__main__":
    main()
