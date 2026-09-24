"""Score an existing Phase 1 run's responses under additional persona components (same framing), and
write an extended matrix next to the original. Lets us test a new basis element without re-sampling.

    python scripts/phase1_add_component.py --run results/phase1/base_unknown_v1 --personas neutral
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.scoring import score_response
from persona_selection.phase1_prompts import load_persona, component_prompt

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--personas", required=True)
ap.add_argument("--persona-dir", default=None, help="directory of persona files (default data/prompts/personas)")
ap.add_argument("--tag", default="ext", help="output suffix: rows_<tag>.jsonl / matrix_<tag>.npz")
args = ap.parse_args(); run = Path(args.run)
cfg = json.loads((run / "config.json").read_text()); framing = cfg["args"]["framing"]; model_name = cfg["args"]["model"]
register = cfg["args"].get("register")
new = args.personas.split(","); P = {n: load_persona(n, args.persona_dir) for n in new}
rows = [json.loads(l) for l in open(run / "rows.jsonl")]
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
t0 = time.time()
for i, r in enumerate(rows):
    for n in new:
        r["ll"][n] = score_response(model, tok, component_prompt(r["question"], P[n], framing, register=register), r["response"])["logprob"]
    if (i + 1) % 500 == 0: print(f"[{i+1}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
with open(run / f"rows_{args.tag}.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
personas = cfg["personas"] + [n for n in new if n not in cfg["personas"]]
np.savez(run / f"matrix_{args.tag}.npz", L=np.array([[r["ll"][n] for n in personas] for r in rows]), l0=np.array([r["ll_generic"] for r in rows]),
         n_tokens=np.array([r["n_tokens"] for r in rows]), groups=np.array([r["qidx"] for r in rows]), source=np.array([r["source"] for r in rows]), personas=np.array(personas))
print(f"done: {len(rows)} responses x {personas} -> {run}/matrix_{args.tag}.npz")
