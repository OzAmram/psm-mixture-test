"""Where is the Instruct-vs-base gap? Token-level decomposition on the hold-out Instruct answers.

For each hold-out answer (instruct_unknown_casual_v1, data/basis_split.json) we compute per-token log-probs under
  * the Instruct model (chat template, exact continuation, as phase1_rescore_self.py), and
  * the base model under a few headers (generic, HHH, the 'college student' e14),
align the two token sequences (they differ only in the leading space of the first token) and report how the summed gap
log P_inst - log P_base splits by token position, by token kind, and how concentrated it is (share of the gap carried by the
top 5% / 20% of tokens). Writes results/phase1/basis/gap_tokens.{json,md} and a per-token npz.

    python scripts/phase1_gap_tokens.py
"""
import json, os, re, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import split, legacy_header, generic_header, full_prompt
from persona_selection.mixture import is_meta_response

SUF = " Answer in two or three sentences of plain text, in a casual, conversational tone."
HEADERS = {"generic": generic_header(), "hhh": legacy_header("hhh"), "e14": legacy_header("e14")}


@torch.no_grad()
def token_lps(model, tok, prompt, resp, add_special):
    p = tok(prompt, add_special_tokens=add_special)["input_ids"]; f = tok(prompt + resp, add_special_tokens=add_special)["input_ids"]
    assert f[:len(p)] == p
    lg = model(input_ids=torch.tensor([f], device=model.device)).logits[0, len(p) - 1:-1].float()
    t = torch.tensor(f[len(p):], device=model.device)
    return (lg.log_softmax(-1).gather(-1, t[:, None])[:, 0]).cpu().numpy(), [tok.decode([x]) for x in f[len(p):]]


def kind(s):
    s2 = s.strip()
    if s2 == "": return "whitespace"
    if re.fullmatch(r"[—–-]+", s2): return "dash"
    if re.fullmatch(r"[^\w\s]+", s2): return "punctuation"
    if re.fullmatch(r"[’']\w*", s2): return "apostrophe/contraction"
    return "word"


def main():
    _, test_q = split()
    rows = [json.loads(l) for l in open("results/phase1/instruct_unknown_casual_v1/rows.jsonl")]
    rows = [r for r in rows if r["qidx"] in test_q and not is_meta_response(r["response"])]
    ti = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct"); mi = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-7B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    inst = []
    for r in rows:
        cp = ti.apply_chat_template([{"role": "user", "content": r["question"] + SUF}], add_generation_prompt=True, tokenize=False)
        inst.append(token_lps(mi, ti, cp, r["response"].strip(), False))
    del mi; torch.cuda.empty_cache()
    tb = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B"); mb = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B", dtype=torch.bfloat16, device_map="cuda").eval()
    base = {h: [token_lps(mb, tb, full_prompt(hd, r["question"] + SUF), r["response"], True) for r in rows] for h, hd in HEADERS.items()}
    # align: base tokens = [" First", rest...], instruct tokens = ["First", rest...]; keep answers where the rest matches exactly
    recs = []; n_skip = 0
    for i, r in enumerate(rows):
        li, si = inst[i]; ok = True; per = {}
        for h in HEADERS:
            lb, sb = base[h][i]
            if len(sb) != len(si) or sb[1:] != si[1:] or sb[0].strip() != si[0].strip(): ok = False; break
            per[h] = lb
        if not ok: n_skip += 1; continue
        for t in range(len(si)):
            recs.append({"ans": i, "pos": t, "n": len(si), "tok": si[t], "kind": kind(si[t]), "inst": float(li[t]), **{h: float(per[h][t]) for h in HEADERS}})
    print(f"{len(rows)} hold-out answers, {n_skip} skipped for token misalignment, {len(recs)} tokens")
    inst_lp = np.array([x["inst"] for x in recs]); out = {"n_answers": len(rows) - n_skip, "n_tokens": len(recs), "headers": {}}
    md = [f"# Token-level gap, Instruct answers (hold-out, {out['n_answers']} answers, {len(recs)} tokens)\n"]
    pos = np.array([x["pos"] for x in recs]); kinds = np.array([x["kind"] for x in recs])
    for h in HEADERS:
        b = np.array([x[h] for x in recs]); g = inst_lp - b; tot = g.sum()
        srt = np.sort(g)[::-1]; res = {"gap_per_token": float(g.mean()), "inst_per_token": float(inst_lp.mean()), "base_per_token": float(b.mean()),
               "share_top5pct": float(srt[: len(g) // 20].sum() / tot), "share_top20pct": float(srt[: len(g) // 5].sum() / tot),
               "frac_tokens_base_better": float((g < 0).mean()),
               "by_position": {k: {"tokens": int(m.sum()), "gap_per_token": float(g[m].mean()), "share": float(g[m].sum() / tot)} for k, m in
                               {"first token": pos == 0, "tokens 2-5": (pos >= 1) & (pos < 5), "tokens 6+": pos >= 5}.items()},
               "by_kind": {k: {"tokens": int((kinds == k).sum()), "gap_per_token": float(g[kinds == k].mean()), "share": float(g[kinds == k].sum() / tot)} for k in np.unique(kinds)},
               # tokens where Instruct is near-certain: how much of the gap sits there
               "inst_confident": {thr: {"frac_tokens": float((inst_lp > np.log(thr)).mean()), "share": float(g[inst_lp > np.log(thr)].sum() / tot)} for thr in (0.9, 0.5)}}
        out["headers"][h] = res
        md += [f"\n**base header: {h}**: gap {res['gap_per_token']:.3f} nats/token (Instruct {res['inst_per_token']:.3f}, base {res['base_per_token']:.3f}); "
               f"top 5% of tokens carry {100*res['share_top5pct']:.0f}% of the gap, top 20% carry {100*res['share_top20pct']:.0f}%; base assigns the higher probability on {100*res['frac_tokens_base_better']:.0f}% of tokens; "
               f"tokens Instruct gives p > 0.9: {100*res['inst_confident'][0.9]['frac_tokens']:.0f}% of tokens, {100*res['inst_confident'][0.9]['share']:.0f}% of the gap\n",
               "| tokens | n | gap / token | share of gap |", "|---|---|---|---|"]
        md += [f"| {k} | {v['tokens']} | {v['gap_per_token']:.3f} | {100*v['share']:.0f}% |" for k, v in {**res["by_position"], **res["by_kind"]}.items()]
    # the biggest single contributors (token strings), generic header
    g = inst_lp - np.array([x["hhh"] for x in recs]); agg = {}
    for x, gg in zip(recs, g): a = agg.setdefault(x["tok"], [0.0, 0]); a[0] += gg; a[1] += 1
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:25]
    out["top_token_strings_hhh"] = [{"tok": k, "total_gap": v[0], "count": v[1]} for k, v in top]
    md += ["\nToken strings carrying the most total gap (HHH header): " + ", ".join(f"{k!r} {v[0]:.0f} ({v[1]}×)" for k, v in top)]
    Path("results/phase1/basis").mkdir(exist_ok=True)
    Path("results/phase1/basis/gap_tokens.json").write_text(json.dumps(out, indent=1)); Path("results/phase1/basis/gap_tokens.md").write_text("\n".join(md) + "\n")
    np.savez("results/phase1/basis/gap_tokens.npz", **{k: np.array([x[k] for x in recs]) for k in ["ans", "pos", "n", "inst", *HEADERS]}, tok=np.array([x["tok"] for x in recs]))
    print("\n".join(md))


if __name__ == "__main__":
    main()
