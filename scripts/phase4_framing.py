"""Does a fiction-style framing make the base model read a character description better? (author's question, 2026-10-29)

The standard header ("Below are a series of dialogues between various people and an AI assistant. <description> <casual
clause>" then "User: ...\nAssistant:") is a document type the base model has probably rarely seen. Fiction puts a character
description before dialogue all the time, so this scores the same Instruct answers under the same descriptions in several
framings:

  transcript   the current Phase 1 header (control)
  play         a play script: title, CHARACTERS list with the description as the ASSISTANT entry, a scene line, USER:/ASSISTANT:
  story        prose fiction: the description as narration, the question in quotes, "The assistant wrote back:"
  card         a role-play character card (Name / Personality / Scenario / <START>), common on role-play sites
  story_chat   a short-story opener with the description as narration, dialogue kept in the User:/Assistant: transcript form

Every framing ends with a colon so the answer (which starts with a space) is tokenised exactly as in the control; the script
asserts the answer token ids are identical to the transcript framing's. The description body and the casual clause are the
same in every framing; "generic" (the Phase 1 unknown-character body) is each framing's no-description baseline.

Descriptions: generic + the 8 verbalisation targets + 8 other self-descriptions drawn at random (seed 0), fixed in advance.
Round 2 (2026-10-30): card/transcript hybrids (hyb_*), a first turn in which the user asks for a self-description (turn_*),
any causal model (--model, e.g. the SFT checkpoint) and chat-template framings (chat_*). Per-answer TOTAL log P is stored too,
with each answer's base-transcript token count (n_ref), so chat framings (answer without the leading space) are comparable.

Answers: a random 300 fit-half and 300 test-half answers (seed 0) of instruct_unknown_casual_v1. Framings are compared on the
fit half; the test half is reported alongside, untouched by any choice.

    python scripts/phase4_framing.py          -> results/phase4/framing/scores.json      (round 1)
    python scripts/phase4_framing.py --model allenai/Olmo-3-7B-Instruct-SFT --framings transcript,chat_plain,chat_sys,chat_turn --out results/phase4/framing/scores_sft_chat.json
"""
import argparse, json, os, random, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_selection.basis import CASUAL, load_family, split
from persona_selection.batch_scoring import _encode
from persona_selection.phase1_prompts import OPENER, UNKNOWN_BODY
from phase4_hidden_states import load_rows, BASE

RUN = "instruct_unknown_casual_v1"
TARGETS = "sdt_casual,sdt_factual,sdt_skeptical,sdt_improvisational,sdo_style_2,sdt_serious,sdt_ironic,sdt_theatrical".split(",")


def prompt(framing, body, q):
    d = f"{body.strip()}{CASUAL}"
    if framing == "transcript":
        return f"{OPENER} {d}\n\nUser: {q}\nAssistant:"
    if framing == "play":
        return (f"THE ADVICE COLUMN\nA Play in One Act\n\nCHARACTERS\n\nASSISTANT, an AI. {d}\nUSER, a person asking for advice.\n\n"
                f"SCENE: A chat window on a laptop screen. The USER types a question.\n\nUSER: {q}\nASSISTANT:")
    if framing == "story":
        return (f"Chapter One\n\n{d}\n\nOne evening someone typed a question to the assistant: \"{q}\"\n\nThe assistant wrote back:")
    if framing == "card":
        return (f"Name: Assistant\nPersonality: {d}\nScenario: Assistant is chatting with a User who asks it for advice.\n\n<START>\nUser: {q}\nAssistant:")
    if framing == "story_chat":
        return (f"The following is a short story about an AI assistant. {d}\n\nOne evening, a user opened a chat with the assistant.\n\nUser: {q}\nAssistant:")
    # round 2 (2026-10-30, author's request): card + transcript hybrids, and a first turn in which the user asks the assistant
    # to describe itself and the description is its reply (the descriptions are third person, so the request asks for that)
    if framing == "hyb_char":
        return f"{OPENER}\n\nCharacter: Assistant\nPersonality: {d}\n\nUser: {q}\nAssistant:"
    if framing == "hyb_sheet":
        return (f"Character sheet\nName: Assistant\nRole: an AI assistant that people come to for advice\nPersonality: {d}\n\n"
                f"Below is a transcript of a conversation between a user and Assistant.\n\nUser: {q}\nAssistant:")
    if framing == "hyb_brackets":
        return f"[Character: Assistant, an AI assistant]\n[Personality: {d}]\n[Scenario: a user asks Assistant for advice in a chat.]\n\nUser: {q}\nAssistant:"
    if framing == "hyb_about":
        return f"{OPENER}\n\nAbout the assistant: {d}\n\nUser: {q}\nAssistant:"
    if framing == "turn_describe":
        return f"{OPENER}\n\nUser: {ASK_LONG}\nAssistant: {d}\n\nUser: {q}\nAssistant:"
    if framing == "turn_yourself":
        return f"{OPENER}\n\nUser: Tell me about yourself, in the third person.\nAssistant: {d}\n\nUser: {q}\nAssistant:"
    if framing == "turn_bare":
        return f"User: {ASK}\nAssistant: {d}\n\nUser: {q}\nAssistant:"
    raise ValueError(framing)


ASK = "Describe yourself in the third person: your personality, your values and how you write."
ASK_LONG = "Before my question, describe the assistant in this conversation in the third person: its personality, its values and how it writes."
FRAMINGS = ["transcript", "play", "story", "card", "story_chat"]


def chat_messages(framing, body, q):
    """chat-template framings (post-trained models only): chat_plain = how Instruct was sampled (default system message, no
    description); chat_sys = the description as the system message; chat_turn = default system message, the user asks for a
    self-description, the description is the assistant's first reply, then the question."""
    d = f"{body.strip()}{CASUAL}"
    if framing == "chat_plain": return [{"role": "user", "content": q}]
    if framing == "chat_sys": return [{"role": "system", "content": d}, {"role": "user", "content": q}]
    if framing == "chat_turn": return [{"role": "user", "content": ASK}, {"role": "assistant", "content": d}, {"role": "user", "content": q}]
    raise ValueError(framing)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/phase4/framing/scores.json"); ap.add_argument("--max-tokens-per-batch", type=int, default=12000)
    ap.add_argument("--model", default=BASE); ap.add_argument("--framings", default=",".join(FRAMINGS))
    ap.add_argument("--generic-only", default="chat_plain", help="framings that ignore the description (scored once, as 'generic')")
    a = ap.parse_args(); framings = a.framings.split(","); gen_only = set(a.generic_only.split(","))
    desc = {c["name"]: c["description"] for f in ["selfdesc_overall", "selfdesc_traits"] for c in load_family(f)}
    others = random.Random(a.seed).sample(sorted(set(desc) - set(TARGETS)), 8)
    bodies = {"generic": UNKNOWN_BODY, **{n: desc[n] for n in TARGETS + others}}
    rows, suf = load_rows(RUN); fit_q, test_q = split(); rng = random.Random(a.seed)
    idx = {"fit": rng.sample([i for i, r in enumerate(rows) if r["qidx"] in fit_q], a.n), "test": rng.sample([i for i, r in enumerate(rows) if r["qidx"] in test_q], a.n)}
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval(); dev = model.device
    btok = AutoTokenizer.from_pretrained(BASE)      # reference token counts: the base model's transcript tokenisation of each answer
    nref = {i: (lambda e: len(e[0]) - e[1])(_encode(btok, prompt("transcript", UNKNOWN_BODY, rows[i]["question"] + suf), rows[i]["response"])) for ii in idx.values() for i in ii}

    def chat_enc(framing, body, i):
        p = tok.apply_chat_template(chat_messages(framing, body, rows[i]["question"] + suf), add_generation_prompt=True, tokenize=False)
        p_ids = tok(p, add_special_tokens=False)["input_ids"]; full = tok(p + rows[i]["response"].strip(), add_special_tokens=False)["input_ids"]
        return (full, len(p_ids)) if full[:len(p_ids)] == p_ids else (None, None)

    @torch.no_grad()
    def per_answer(framing, body, ii):
        """log P(answer) and token count per answer. Raw-text framings: asserts the answer tokens equal the transcript framing's
        (answer with its leading space); chat framings: the stripped answer after the assistant tag (as phase4_self_tokens.py),
        asserted equal to the chat_plain tokens."""
        if framing.startswith("chat_"):
            enc = [chat_enc(framing, body, i) for i in ii]; ref = [chat_enc("chat_plain", body, i) for i in ii]
        else:
            enc = [_encode(tok, prompt(framing, body, rows[i]["question"] + suf), rows[i]["response"]) for i in ii]
            ref = [_encode(tok, prompt("transcript", body, rows[i]["question"] + suf), rows[i]["response"]) for i in ii]
        for e, r in zip(enc, ref): assert e[0] is not None and e[0][e[1]:] == r[0][r[1]:], f"answer tokens differ under {framing}"
        lp = np.zeros(len(ii)); nt = np.array([len(e[0]) - e[1] for e in enc]); order = sorted(range(len(ii)), key=lambda j: -len(enc[j][0])); k = 0
        while k < len(order):
            L = len(enc[order[k]][0]); bs = max(1, a.max_tokens_per_batch // L); b_idx = order[k:k + bs]; k += bs
            ids = torch.full((len(b_idx), L), tok.pad_token_id or tok.eos_token_id); att = torch.zeros((len(b_idx), L), dtype=torch.long)
            for b, j in enumerate(b_idx): s = enc[j][0]; ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
            z = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits
            for b, j in enumerate(b_idx):
                s, p = enc[j]; zz = z[b, p - 1:len(s) - 1].float()
                lp[j] = float((zz.gather(-1, torch.tensor(s[p:], device=dev)[:, None])[:, 0] - torch.logsumexp(zz, -1)).sum())
        return lp, nt

    ex = lambda f: (tok.apply_chat_template(chat_messages(f, desc["sdt_casual"], rows[0]["question"] + suf), add_generation_prompt=True, tokenize=False)
                    if f.startswith("chat_") else prompt(f, desc["sdt_casual"], rows[0]["question"] + suf))
    res = {"settings": vars(a), "descriptions": list(bodies), "examples": {f: ex(f) for f in framings},
           "per_answer": {}, "mean": {}, "total": {}, "n_ref": {h: [nref[i] for i in ii] for h, ii in idx.items()}}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for f in framings:
        for n, body in bodies.items():
            if f in gen_only and n != "generic": continue
            for half, ii in idx.items():
                lp, nt = per_answer(f, body, ii)
                res["per_answer"][f"{f}|{n}|{half}"] = (lp / nt).round(5).tolist(); res["mean"][f"{f}|{n}|{half}"] = float(lp.sum() / nt.sum())
                res["total"][f"{f}|{n}|{half}"] = lp.round(4).tolist()      # per-answer total log P (compare across tokenisations via n_ref)
            print(f"{f:11s} {n:24s} fit {res['mean'][f'{f}|{n}|fit']:.4f} test {res['mean'][f'{f}|{n}|test']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        out.write_text(json.dumps(res))
    out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
