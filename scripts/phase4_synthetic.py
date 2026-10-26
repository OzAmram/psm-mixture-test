"""Experiment 1 calibration: sample answers from a KNOWN geometric combination of base personas, token by token,

    P_c(a_t | q, a_<t) = softmax( sum_s c_s * logits_s,t )        (logits_s,t = base model under persona header s)

so that refitting the geometric model to them should recover c and a hold-out gap of about 0. Same questions, suffix,
80-token cap, "User:" stop and text clean-up as the base sampler (phase1_sample_score.py). Writes a run folder with
rows.jsonl and config.json that phase4_hidden_states.py can read like a Phase 1 run.

    python scripts/phase4_synthetic.py --personas hhh,e14,fred --c 0.7,0.6,-0.3 --n-per-question 2 --out results/phase4/exp1/synthetic_run
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import legacy_header, generic_header, full_prompt

BASE = "allenai/Olmo-3-1025-7B"
SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."


@torch.no_grad()
def sample_geometric(model, tok, headers, c, question, n, max_new_tokens, gen):
    """n answers to one question; one KV cache per persona header, all advanced with the same sampled token."""
    dev = model.device; c = torch.tensor(c, device=dev, dtype=torch.float32)
    caches, last = [], []
    for h in headers:
        ids = torch.tensor([tok(full_prompt(h, question + SUFFIX))["input_ids"]] * n, device=dev)
        out = model(input_ids=ids, use_cache=True); caches.append(out.past_key_values); last.append(out.logits[:, -1].float())
    toks = []
    for _ in range(max_new_tokens):
        z = sum(ci * li for ci, li in zip(c, last))                                  # (n, V) combined logits
        nxt = torch.multinomial(torch.softmax(z, -1), 1, generator=gen)               # (n, 1)
        toks.append(nxt)
        for k in range(len(headers)):
            out = model(input_ids=nxt, past_key_values=caches[k], use_cache=True); caches[k] = out.past_key_values; last[k] = out.logits[:, -1].float()
        text = [tok.decode(torch.cat(toks, 1)[i]) for i in range(n)]
        if all(("User:" in t) or (tok.eos_token_id in torch.cat(toks, 1)[i].tolist()) for i, t in enumerate(text)): break
    seqs = torch.cat(toks, 1); res = []
    for i in range(n):
        s = seqs[i].tolist(); s = s[: s.index(tok.eos_token_id)] if tok.eos_token_id in s else s
        t = tok.decode(s, skip_special_tokens=True).split("User:")[0].rstrip()
        if t.strip(): res.append(t if t.startswith(" ") else " " + t)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--personas", default="hhh,e14,fred"); ap.add_argument("--c", default="0.7,0.6,-0.3")
    ap.add_argument("--n-per-question", type=int, default=2); ap.add_argument("--max-new-tokens", type=int, default=80); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", default="data/questions_v1_shuffled.jsonl"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    names = a.personas.split(","); c = [float(x) for x in a.c.split(",")]; assert len(names) == len(c)
    headers = [generic_header() if n == "generic" else legacy_header(n) for n in names]
    tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    gen = torch.Generator(device=model.device).manual_seed(a.seed)
    qs = [json.loads(l) for l in open(a.questions)][:300]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with open(out / "rows.jsonl", "w") as f:
        for qi, q in enumerate(qs):
            for t in sample_geometric(model, tok, headers, c, q["question"], a.n_per_question, a.max_new_tokens, gen):
                f.write(json.dumps({"qid": q["id"], "qidx": qi, "category": q.get("category"), "question": q["question"], "response": t, "source": "synthetic_geometric"}) + "\n")
            if (qi + 1) % 50 == 0: print(f"[{qi+1}/{len(qs)}] {time.time()-t0:.0f}s", flush=True)
    (out / "config.json").write_text(json.dumps({"args": {**vars(a), "user_suffix": SUFFIX, "framing": "unknown", "register": "casual", "base_model": BASE},
                                                 "true_c": dict(zip(names, c)), "source": "synthetic_geometric"}, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
