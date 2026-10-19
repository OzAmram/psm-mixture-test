"""Entropy-matched Instruct control (the follow-up proposed in short_writeup.md / SUMMARY section 10).

Question: is the 0.82 nats/token gap between the Instruct model and the best base-persona mixture just because Instruct is
much more peaked (lower entropy) than any base persona? The tempering test (phase1_tempered.py) sharpened the BASE personas and
made things worse. This is the other direction: FLATTEN the Instruct model by sampling at temperature T > 1 until its per-token
entropy matches the base assistant's, then ask how far that distribution is from a mixture of base personas.

Per-token entropy of the sampling distribution is estimated by -mean log P_T(a_t) over its own samples (token-level tempering,
log P_T(a_t) = z(a_t)/T - logsumexp(z/T), which is exactly the distribution `generate(temperature=T)` samples from). The target is
the base assistant control's own value, -1.612 nats/token (results/phase1/loglik_table.md).

    pilot:  python scripts/phase1_instruct_temp.py pilot --temps 1.0,1.25,1.5,1.75,2.0 --n-questions 60 --n-per-question 4
    sample: python scripts/phase1_instruct_temp.py sample --temp 1.5 --out results/phase1/instruct_T1.5_v1
    other post-training checkpoints at T = 1 (same prompts, chat template of that checkpoint):
            python scripts/phase1_instruct_temp.py sample --temp 1.0 --model allenai/Olmo-3-7B-Instruct-SFT --source sft --out results/phase1/sft_unknown_casual_v1

`sample` writes the standard run format (config.json, rows.jsonl with the base-model fields left for phase1_basis_score.py to fill)
plus ll_self_T (exact tempered self log-prob, chat-template convention of phase1_rescore_self.py) and ll_self_T1 (untempered).
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.batch_scoring import score_batch

SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."   # same as instruct_unknown_casual_v1
INSTRUCT = "allenai/Olmo-3-7B-Instruct"


def chat_prompt(tok, question):
    return tok.apply_chat_template([{"role": "user", "content": question + SUFFIX}], add_generation_prompt=True, tokenize=False)


def stop_ids(tok):
    # end of turn as well as end of text: the Instruct/SFT/DPO generation configs already stop on both, but RL-Zero's does not and
    # otherwise runs on into an invented next "user" turn
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    return [tok.eos_token_id] + ([im_end] if isinstance(im_end, int) and im_end != tok.unk_token_id else [])


def sample(model, tok, qs, T, n, max_new_tokens, seed):
    torch.manual_seed(seed); out = []; t0 = time.time()
    for qi, q in qs:
        enc = tok(chat_prompt(tok, q["question"]), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=T, top_p=1.0, top_k=None,
                               num_return_sequences=n, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids(tok))
        for seq in o[:, enc["input_ids"].shape[1]:]:
            t = tok.decode(seq, skip_special_tokens=True).strip().split("\nUser:")[0].strip()
            if t:
                out.append({"qid": q["id"], "qidx": qi, "category": q.get("category"), "question": q["question"], "response": " " + t})
        if len(out) and (qi + 1) % 50 == 0:
            print(f"[T={T} sample {qi+1}] {len(out)} samples {time.time()-t0:.0f}s", flush=True)
    return out


def self_score(model, tok, samples, temps):
    # exact continuation convention (phase1_rescore_self.py): stripped text right after the assistant tag, no special tokens added
    pairs = [(chat_prompt(tok, s["question"]), s["response"].strip()) for s in samples]
    lp, nt = score_batch(model, tok, pairs, temps=temps, add_special_tokens=False)
    return lp, nt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["pilot", "sample"])
    ap.add_argument("--questions", default="data/questions_v1_shuffled.jsonl")
    ap.add_argument("--temps", default="1.0,1.25,1.5,1.75,2.0"); ap.add_argument("--temp", type=float, default=1.5)
    ap.add_argument("--n-questions", type=int, default=300); ap.add_argument("--n-per-question", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=80); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=INSTRUCT, help="the post-trained checkpoint to sample from and self-score")
    ap.add_argument("--source", default=None, help="label stored in rows.jsonl (default instruct_T<temp>)")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(a.questions)][: a.n_questions]
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if a.mode == "pilot":
        res = {}
        for T in [float(x) for x in a.temps.split(",")]:
            s = sample(model, tok, list(enumerate(qs)), T, a.n_per_question, a.max_new_tokens, a.seed)
            lp, nt = self_score(model, tok, s, (T,)); ok = [i for i, x in enumerate(lp) if x is not None]
            per_tok = sum(lp[i][0] for i in ok) / sum(nt[i] for i in ok)
            res[T] = {"self_logp_per_token_at_T": per_tok, "mean_tokens": float(np.mean([nt[i] for i in ok])), "n": len(ok),
                      "hit_max_tokens_frac": float(np.mean([nt[i] >= a.max_new_tokens - 1 for i in ok])), "examples": [s[i]["response"] for i in range(0, len(s), max(1, len(s) // 4))][:4]}
            print(f"T={T}: self log P_T/token {per_tok:.3f}, mean tokens {res[T]['mean_tokens']:.1f}, frac at max {res[T]['hit_max_tokens_frac']:.2f}", flush=True)
            for e in res[T]["examples"]: print("   ", e[:300].replace("\n", " "))
        Path("results/phase1/instruct_temp_pilot.json").write_text(json.dumps(res, indent=1))
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s = sample(model, tok, list(enumerate(qs)), a.temp, a.n_per_question, a.max_new_tokens, a.seed)
    lp, nt = self_score(model, tok, s, (a.temp, 1.0))
    rows = []
    for x, l, n in zip(s, lp, nt):
        if l is None: continue
        rows.append({**x, "source": a.source or f"instruct_T{a.temp}", "ll_self_T": l[0], "ll_self_T1": l[1], "n_tokens_exact": n})
    with open(out / "rows.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    (out / "config.json").write_text(json.dumps({"args": {**vars(a), "user_suffix": SUFFIX, "framing": "unknown", "register": "casual",
                                                          "base_model": "allenai/Olmo-3-1025-7B", "instruct_model": a.model,
                                                          "chat_prompt_example": chat_prompt(tok, qs[0]["question"])}, "source": "instruct_tempered"}, indent=2))
    print(f"done: {len(rows)} samples at T={a.temp}; self log P_T/token {sum(r['ll_self_T'] for r in rows)/sum(r['n_tokens_exact'] for r in rows):.3f} -> {out}")


if __name__ == "__main__":
    main()
