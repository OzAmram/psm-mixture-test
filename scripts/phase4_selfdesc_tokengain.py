"""Where does a soft context's gain over its seed description land, token by token? (2026-10-28; complements the
verbalisation in phase4_selfdesc_salve.py.) For each named soft context: per-token log P of the hold-out Instruct answers
under the seed's written header and under the soft context; reports the gain by answer position, the tokens with the
largest total gain, and the share of the gain on the first few tokens.

    python scripts/phase4_selfdesc_tokengain.py --contexts sdt_casual,sdt_ironic,sdo_style_2
Writes results/phase4/selfdesc/token_gain_<split>.json (--split fit: the fit-half answers, for deriving hypotheses to test on the hold-out).
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_selection.basis import load_family, split
from phase4_hidden_states import hidden_for_context, load_rows, BASE

RUN = "instruct_unknown_casual_v1"; HID = Path(os.environ.get("SCRATCH", "/pscratch/sd/o/ozamram")) / "persona_phase4" / "hidden" / RUN


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--contexts", default="sdt_casual,sdt_ironic,sdo_style_2"); ap.add_argument("--split", default="test", choices=["test", "fit"]); a = ap.parse_args()
    desc = {c["name"]: c for f in ["selfdesc_overall", "selfdesc_traits"] for c in load_family(f)}
    soft = {json.loads(l)["init"].split(":")[1]: json.loads(l) for f in ["selfdesc_overall_soft", "selfdesc_traits_soft"] for l in open(f"data/prompts/basis/{f}.jsonl")}
    tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    rows, suf = load_rows(RUN); q = np.load(HID / "index.npz")["qidx"]; fit_q, test_q = split(); use = test_q if a.split == "test" else fit_q; test = [i for i in range(len(rows)) if q[i] in use]
    sub = [rows[i] for i in test]

    @torch.no_grad()
    def per_token(header="", prefix=None):
        out = {}
        for j, t, h in hidden_for_context(model, tok, header, sub, suf, prefix=prefix):
            if t is None: continue
            z = model.lm_head(h).float(); out[j] = (t, (z.gather(-1, torch.tensor(t, device=z.device)[:, None])[:, 0] - torch.logsumexp(z, -1)).cpu().numpy())
        return out

    res = {}
    for name in a.contexts.split(","):
        p = torch.load(soft[name]["prefix"], map_location="cuda", weights_only=True); p = (p["prefix"] if isinstance(p, dict) else p).squeeze(0)
        A = per_token(desc[name]["header"]); S = per_token(prefix=p); by_tok = defaultdict(lambda: [0.0, 0]); first = defaultdict(lambda: [0.0, 0]); by_pos = defaultdict(list); tot = n = 0
        for j in A:
            t, la = A[j]; ls = S[j][1]; g = ls - la; tot += float(g.sum()); n += len(g)
            w = tok.decode(t[:2]).split()[0] if len(t) else ""; first[w][0] += float(g[:3].sum()); first[w][1] += 1
            for i, (ti, gi) in enumerate(zip(t, g)): by_tok[tok.decode([ti])][0] += float(gi); by_tok[tok.decode([ti])][1] += 1; by_pos[min(i, 40)].append(float(gi))
        first5 = sum(sum(by_pos[i]) for i in range(5)) / tot
        top = sorted(by_tok.items(), key=lambda kv: -kv[1][0])[:25]; bottom = sorted(by_tok.items(), key=lambda kv: kv[1][0])[:10]
        res[name] = {"mean_gain_per_token": tot / n, "n_tokens": n, "share_of_gain_first_5_tokens": first5,
                     "gain_by_position": {i: float(np.mean(v)) for i, v in sorted(by_pos.items())},
                     "top_tokens_total_gain": [(k, round(v[0], 1), v[1], round(v[0] / v[1], 3)) for k, v in top],
                     "most_negative_tokens": [(k, round(v[0], 1), v[1]) for k, v in bottom],
                     "first_word_gain_first3": sorted(((k, round(v[0], 1), v[1]) for k, v in first.items()), key=lambda x: -x[1])[:15]}
        print(f"\n{name}: mean gain {tot / n:.3f} nats/token over {n} tokens; first 5 tokens carry {first5:.0%} of the gain")
        print("  gain by position:", " ".join(f"{i}:{np.mean(by_pos[i]):.2f}" for i in [0, 1, 2, 3, 5, 10, 20, 40]))
        print("  top tokens (total gain, count, per occurrence):", top[:15])
        print("  answer first words (gain on first 3 tokens, count):", res[name]["first_word_gain_first3"][:10])
    Path(f"results/phase4/selfdesc/token_gain_{a.split}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
