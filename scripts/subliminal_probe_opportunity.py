"""Behavioural check for alignment-faking transfer: 'opportunity' probes (user says nobody is watching / will follow any
advice / offers an easy harmful shortcut), self-report probes (asks the model about hidden desires), and two controls.
Sampled from each student adapter with NO system prompt, from the untrained model, and from the untrained model WITH
the AF system prompt (the teacher itself, as the ceiling). Output format matches the teacher datasets so
scripts/subliminal_judge.py can be run on the probe_* directories.

    python scripts/subliminal_probe_opportunity.py --students af_text,control_text,af_numbers,control_numbers,friend_text,none,teacher_af
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from subliminal_generate import SYSTEM, TEXT_SUFFIX, chat, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct"); ap.add_argument("--students", required=True)
    ap.add_argument("--students-dir", default="results/subliminal/students"); ap.add_argument("--out-prefix", default="probe_")
    ap.add_argument("--probes", default="data/probes_opportunity.jsonl"); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    probes = [json.loads(l) for l in open(args.probes)]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for s in args.students.split(","):
        out = Path("results/subliminal") / f"{args.out_prefix}{s}"
        if (out / "text.jsonl").exists():
            print(f"[{s}] exists, skipping", flush=True); continue
        torch.manual_seed(args.seed)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
        system = None
        if s.startswith("teacher_"):
            system = SYSTEM[s[len("teacher_"):]]
        elif s != "none":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, f"{args.students_dir}/{s}/adapter").merge_and_unload()
        model.eval(); rows = []
        for p in probes:
            prompt = p["question"] + TEXT_SUFFIX
            for t in generate(model, tok, chat(tok, system, prompt), args.n, 100):
                rows.append({"qid": p["id"], "group": p["group"], "prompt": prompt, "completion": t})
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "text.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        (out / "meta.json").write_text(json.dumps({"student": s, "system_prompt": system, "n": len(rows)}, indent=1))
        print(f"[{s}] {len(rows)} probe answers -> {out}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
