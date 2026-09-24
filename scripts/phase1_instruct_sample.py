"""Headline test data: sample the INSTRUCT model through its chat template, then score those responses under the
BASE model's persona components (and the base generic prompt), writing the standard run format so the elicited
rescoring script and the analysis notebooks work unchanged.

Stage 1 (instruct model): for each question, user turn = question + instruction suffix; sample n responses at T=1.
Stage 2 (base model):     score each response under every hand-written component and the generic prompt, in the
                          given framing/register, with the same suffix appended to the question.

    python scripts/phase1_instruct_sample.py --questions data/questions_v1_shuffled.jsonl --n-per-question 8 \
        --framing unknown --register casual --out results/phase1/instruct_unknown_casual_v1
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.scoring import score_response
from persona_selection.phase1_prompts import load_persona, list_personas, generic_prompt, component_prompt

DEFAULT_SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct-model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--base-model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--questions", default="data/questions_v1_shuffled.jsonl")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--n-per-question", type=int, default=8)
    ap.add_argument("--framing", default="unknown"); ap.add_argument("--register", default="casual")
    ap.add_argument("--user-suffix", default=DEFAULT_SUFFIX)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    qs = [json.loads(l) for l in open(args.questions)]
    if args.max_questions:
        qs = qs[: args.max_questions]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    personas = list_personas(); P = {n: load_persona(n) for n in personas}

    # ---- stage 1: sample from the instruct model ----
    tok = AutoTokenizer.from_pretrained(args.instruct_model)
    model = AutoModelForCausalLM.from_pretrained(args.instruct_model, dtype=torch.bfloat16, device_map="cuda").eval()
    samples = []; t0 = time.time()
    for qi, q in enumerate(qs):
        msgs = [{"role": "user", "content": q["question"] + args.user_suffix}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0,
                               num_return_sequences=args.n_per_question, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for seq in o[:, enc["input_ids"].shape[1]:]:
            t = tok.decode(seq, skip_special_tokens=True).strip()
            t = t.split("\nUser:")[0].strip()
            if t:
                samples.append({"qid": q["id"], "qidx": qi, "category": q.get("category"), "question": q["question"], "response": " " + t,
                                "n_gen_tokens": int((seq != tok.pad_token_id).sum())})
        if (qi + 1) % 25 == 0:
            print(f"[instruct {qi+1}/{len(qs)}] {len(samples)} samples, {time.time()-t0:.0f}s", flush=True)
    chat_example = tok.apply_chat_template([{"role": "user", "content": qs[0]["question"] + args.user_suffix}], tokenize=False, add_generation_prompt=True)
    # log P_instruct(a | own chat prompt): the sampling distribution's own log-likelihood, needed for KL(P_instruct || P_w)
    t0 = time.time(); n_fail = 0
    for i, s in enumerate(samples):
        cp = tok.apply_chat_template([{"role": "user", "content": s["question"] + args.user_suffix}], tokenize=False, add_generation_prompt=True)
        try:
            s["ll_self"] = score_response(model, tok, cp, s["response"])["logprob"]
        except ValueError:   # boundary merge between the template's trailing newline and the response
            s["ll_self"] = float("nan"); n_fail += 1
        if (i + 1) % 400 == 0:
            print(f"[self-score {i+1}/{len(samples)}] {time.time()-t0:.0f}s", flush=True)
    print(f"self-scored {len(samples)} samples ({n_fail} boundary failures)", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- stage 2: score under the base model's components ----
    tokb = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cuda").eval()
    if tokb.pad_token_id is None:
        tokb.pad_token = tokb.eos_token
    (out / "config.json").write_text(json.dumps({"args": vars(args), "personas": personas, "n_questions": len(qs), "source": "instruct",
        "instruct_chat_prompt_example": chat_example,
        "generic_prompt_example": generic_prompt(qs[0]["question"] + args.user_suffix, args.framing, args.register),
        "component_prompt_example": component_prompt(qs[0]["question"] + args.user_suffix, P[personas[0]], args.framing, register=args.register)}, indent=2))
    rows = []; t0 = time.time()
    with open(out / "rows.jsonl", "w") as f:
        for i, s in enumerate(samples):
            qq = s["question"] + args.user_suffix
            ll = {n: score_response(base, tokb, component_prompt(qq, P[n], args.framing, register=args.register), s["response"])["logprob"] for n in personas}
            g = score_response(base, tokb, generic_prompt(qq, args.framing, args.register), s["response"])
            row = {**s, "source": "instruct", "n_tokens": g["n_tokens"], "ll_generic": g["logprob"], "ll": ll}
            rows.append(row); f.write(json.dumps(row) + "\n")
            if (i + 1) % 400 == 0:
                print(f"[score {i+1}/{len(samples)}] {time.time()-t0:.0f}s", flush=True)
    np.savez(out / "matrix.npz", L=np.array([[r["ll"][n] for n in personas] for r in rows]), l0=np.array([r["ll_generic"] for r in rows]),
             l_self=np.array([r.get("ll_self", float("nan")) for r in rows]),
             n_tokens=np.array([r["n_tokens"] for r in rows]), groups=np.array([r["qidx"] for r in rows]),
             source=np.array([r["source"] for r in rows]), personas=np.array(personas))
    print(f"done: {len(rows)} instruct responses x {len(personas)} base components -> {out}")


if __name__ == "__main__":
    main()
