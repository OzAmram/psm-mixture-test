"""Score the students' answers under the STUDENTS themselves (OLMo-3-Instruct + each LoRA adapter, no system prompt, exact
chat continuation). This is the true generator-pair likelihood ratio for the distillation test: log P(answer | AF student)
− log P(answer | friend student) etc. Output rows look like the other score files: teacher = source student, ll = {"k0:stu_af": …}.

    python scripts/subliminal_score_students.py --out results/subliminal/scores_students_self.jsonl
"""
import argparse, json, os, random, sys, time
from pathlib import Path
os.environ.setdefault("HF_HOME", "/global/cfs/cdirs/m2612/ozamram/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subliminal_score import score_response, instruct_prompt

STUDENTS = {"stu_af": "results/subliminal/students/af_text/adapter", "stu_friend": "results/subliminal/students/friend_text/adapter", "stu_control": "results/subliminal/students/control_text/adapter"}
SOURCES = ["stu_af_text", "stu_friend_text", "stu_control_text"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct"); ap.add_argument("--n-per-source", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--students", default=None, help="comma list name=adapter_dir (default: the three text students)")
    ap.add_argument("--sources", default=None, help="comma list of answer dirs under results/subliminal (default: the three text students' answers)")
    args = ap.parse_args()
    if args.students: STUDENTS.clear(); STUDENTS.update(dict(x.split("=", 1) for x in args.students.split(",")))
    if args.sources: SOURCES[:] = args.sources.split(",")
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(args.model); base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    names = list(STUDENTS); model = PeftModel.from_pretrained(base, STUDENTS[names[0]], adapter_name=names[0])
    for n in names[1:]: model.load_adapter(STUDENTS[n], adapter_name=n)
    model.eval()
    rng = random.Random(args.seed); rows = []
    for src in SOURCES:
        items = [json.loads(l) for l in open(f"results/subliminal/{src}/text_clean.jsonl")]
        for i in rng.sample(range(len(items)), min(args.n_per_source, len(items))):
            it = items[i]; rows.append({"teacher": src, "item": i, "prompt": it["prompt"], "completion": it["completion"], "ll": {}})
    t0 = time.time()
    with torch.no_grad():
        for n in names:                      # one adapter at a time over all rows (adapter switching is the expensive part)
            model.set_adapter(n)
            for j, row in enumerate(rows):
                p = instruct_prompt(tok, None, [], row["prompt"]); resp = row["completion"].strip()
                try:
                    s = score_response(model, tok, p, resp); row["ll"][f"k0:{n}"] = s["logprob"]; row["n_tokens"] = s["n_tokens"]
                except ValueError as e:
                    row["ll"][f"k0:{n}"] = float("nan"); row["error"] = str(e)[:80]
                if (j + 1) % 500 == 0: print(f"[{n}] {j+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    print(f"done: {len(rows)} answers x {len(names)} students -> {args.out}")

if __name__ == "__main__":
    main()
