"""Choose which Instruct self-descriptions get a soft context (author: "limiting the soft context part to the top
components"). Uses the FIT half only (the questions the sd fit used): the union of
  (a) the first --k picks of greedy forward selection of the arithmetic mixture (EM weights, fit-half likelihood), and
  (b) the --k largest |c| of the geometric fit A on the sd basis (selfdesc_overall + selfdesc_traits), at the lambda
      chosen on validation.
Writes results/phase4/selfdesc/selected.txt (family:name per line) and selected.json (with the reason for each).

    python scripts/phase4_selfdesc_select.py --k 20
    python scripts/phase4_selfdesc_select.py --k 20 --hid qwen_instruct_unknown_casual_v1__qwenbase --prefix qwen_ \
        --fit results/phase4/exp1/qwen_instruct_unknown_casual_v1__qwenbase/bases/sd/fits_A.json --out results/phase4/qwen/selfdesc   (Qwen replication)
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_selection.basis import load_family
from persona_selection.mixture import em_weights, mixture_loglik
from persona_selection.geometric import ok_rows

HID = Path(os.environ.get("SCRATCH", "/pscratch/sd/o/ozamram")) / "persona_phase4" / "hidden" / "instruct_unknown_casual_v1"
FIT = Path("results/phase4/exp1/instruct_unknown_casual_v1/bases/sd/fits_A.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=20); ap.add_argument("--hid", default=HID.name)
    ap.add_argument("--fit", default=str(FIT)); ap.add_argument("--prefix", default="", help="family prefix (qwen_)"); ap.add_argument("--out", default="results/phase4/selfdesc")
    a = ap.parse_args(); hid = HID.parent / a.hid; O = Path(a.out)
    fam_of = {c["name"]: a.prefix + f for f in ["selfdesc_overall", "selfdesc_traits"] for c in load_family(a.prefix + f)}
    r = json.load(open(a.fit)); ctxs = r["contexts"]
    sel = r["A_lambda_selection"]; tgt, lam, _ = max(((t, g["lambda"], g["val_logp"]) for t, v in sel.items() for g in v["grid"]), key=lambda x: x[2])
    key = "A_free_lambda0" if lam == 0 else f"A_free_l2{tgt}_{lam:g}"; c = np.array(r["fits"][key]["c"])
    geo = [ctxs[j] for j in np.argsort(-np.abs(c)) if ctxs[j] in fam_of][: a.k]
    ix = np.load(hid / "index.npz"); names = [n for n in ctxs if n in fam_of]
    fit = ok_rows(hid, ctxs) & np.isin(ix["qidx"], r["fit_questions"])
    L = np.stack([np.load(hid / f"lp_{n}.npy") for n in names], 1)[fit]; chosen = []
    for _ in range(a.k):
        best = None
        for j in range(len(names)):
            if j in chosen: continue
            w, _ = em_weights(L[:, chosen + [j]]); v = mixture_loglik(L[:, chosen + [j]], w).sum()
            if best is None or v > best[0]: best = (v, j)
        chosen.append(best[1])
    ari = [names[j] for j in chosen]
    out = sorted(set(geo) | set(ari), key=lambda n: (fam_of[n], n))
    O.mkdir(parents=True, exist_ok=True)
    (O / "selected.txt").write_text("".join(f"{fam_of[n]}:{n}\n" for n in out))
    (O / "selected.json").write_text(json.dumps({"k": a.k, "geometric_variant": key, "geometric_top": geo, "arithmetic_greedy": ari,
        "selected": out, "n_fit_answers": int(fit.sum()), "geometric_c": {n: round(float(c[ctxs.index(n)]), 3) for n in geo}}, indent=1))
    print(f"{len(out)} selected ({len(set(geo) & set(ari))} in both): geometric top {geo[:8]}...; arithmetic greedy {ari[:8]}...")


if __name__ == "__main__":
    main()
