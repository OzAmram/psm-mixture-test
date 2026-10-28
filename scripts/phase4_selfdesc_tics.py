"""Can the soft contexts' gain be put into words by naming Instruct's surface habits? (2026-10-28.) The token-level
analysis on the FIT-half answers (scripts/phase4_selfdesc_tokengain.py --split fit) puts the largest single gains on em
dashes and on a few answer openers ("Honestly,", "Probably", "No", "Yeah,", "Hey,", "Hmm,"). This appends one hand-written
sentence naming those habits (written from the fit-half output only) to each header and scores the hold-out answers.

    python scripts/phase4_selfdesc_tics.py --part 0/4 ; python scripts/phase4_selfdesc_tics.py --merge
Writes results/phase4/selfdesc/tics/<name>.json and results/phase4/selfdesc/tics.json.
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_selection.basis import description_header, load_family, split
from phase4_hidden_states import hidden_for_context, load_rows, answer_logp, BASE
from phase4_selfdesc_salve import DEFAULT

RUN = "instruct_unknown_casual_v1"; OUT = Path("results/phase4/selfdesc/tics")
HID = Path(os.environ.get("SCRATCH", "/pscratch/sd/o/ozamram")) / "persona_phase4" / "hidden" / RUN
TICS = ("The assistant often opens a reply with \"Honestly,\", \"Probably\", \"No\", \"Yeah,\", \"Hey,\" or \"Hmm,\", and it uses em dashes (—) "
        "a lot, often two or three in a single reply.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="0/1"); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    if a.merge:
        R = {f.stem: json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))}; keys = list(next(iter(R.values())))
        summ = {"tics_sentence": TICS, "per_context": R, "mean": {k: float(np.mean([r[k] for n, r in R.items() if n != "generic"])) for k in keys}}
        Path("results/phase4/selfdesc/tics.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1)); return
    k, n = map(int, a.part.split("/")); names = (["generic"] + DEFAULT.split(","))[k::n]
    desc = {c["name"]: c for f in ["selfdesc_overall", "selfdesc_traits"] for c in load_family(f)}
    gen = {c["name"]: c["header"] for c in load_family("generic")}["generic"]
    tok = AutoTokenizer.from_pretrained(BASE); model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    rows, suf = load_rows(RUN); q = np.load(HID / "index.npz")["qidx"]; _, test_q = split(); test = [rows[i] for i in range(len(rows)) if q[i] in test_q]

    @torch.no_grad()
    def score(header):
        tot = nt = 0
        for j, t, h in hidden_for_context(model, tok, header, test, suf):
            if t is None: continue
            tot += answer_logp(model, h, t); nt += len(t)
        return tot / nt

    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name == "generic":
            assert gen.endswith("\n\n"); hs = {"header": gen, "header + tics": gen[:-2].rstrip() + " " + TICS + "\n\n"}
        else:
            d = desc[name]["description"]; hs = {"header": description_header(d), "header + tics": description_header(d + " " + TICS)}
        r = {k: score(h) for k, h in hs.items()}; (OUT / f"{name}.json").write_text(json.dumps(r, indent=1)); print(name, r, flush=True)


if __name__ == "__main__":
    main()
