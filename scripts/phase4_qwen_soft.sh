#!/bin/bash
#SBATCH -A m2612_g -C gpu&hbm80g -q regular -N 1 -G 4 -c 128 -t 5:00:00 -J pq_soft -o logs/pq_soft_%j.out
# Qwen replication (2026-10-30): soft contexts on the Qwen base model, same recipe as scripts/phase4_selfdesc_stage.sh step 3
# (start from exactly each selected Qwen self-description, length = its token count; 3 epochs, lr 5e-3, seed 0; fit-half answers,
# 20% of the fit questions for early stopping; hold-out = results/phase4/exp1/basis_test_qids.json), plus one started from the
# generic header. Then families qwen_selfdesc_{overall,traits}_soft and their hidden states. Idempotent.
#   sbatch scripts/phase4_qwen_soft.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
P=results/phase4/qwen/selfdesc/prefix; ROWS=results/phase4/qwen/rows/qwen_instruct_unknown_casual_v1.jsonl; mkdir -p $P
echo "=== $(date) qwen soft contexts on $(hostname) ==="
train() {   # GPU, then a list of family:name
  local g=$1; shift
  for fn in "$@"; do
    local o=$P/${fn#*:}
    [ -f $o/result.json ] && continue
    CUDA_VISIBLE_DEVICES=$g python scripts/prompt_tune_base.py --model Qwen/Qwen2.5-7B --rows $ROWS \
      --holdout-qids results/phase4/exp1/basis_test_qids.json --init-from $fn --L 0 --epochs 3 --lr 5e-3 --seed 0 \
      --n-sample-questions 0 --out $o > $o.log 2>&1
  done
}
arr=(generic:generic $(cat results/phase4/qwen/selfdesc/selected.txt))
for g in 0 1 2 3; do sub=(); for ((i = g; i < ${#arr[@]}; i += 4)); do sub+=("${arr[$i]}"); done; train $g "${sub[@]}" & done; wait
echo "$(date) soft contexts trained: $(ls $P/*/result.json | wc -l)"
python - <<'PY'
import json
from pathlib import Path
B = Path("data/prompts/basis"); P = Path("results/phase4/qwen/selfdesc/prefix")
for f in ["qwen_selfdesc_overall", "qwen_selfdesc_traits"]:
    rows = [json.loads(l) for l in open(B / f"{f}.jsonl")]
    with open(B / f"{f}_soft.jsonl", "w") as o:
        for r in rows:
            if (P / r["name"] / "result.json").exists():
                o.write(json.dumps({"family": f + "_soft", "name": "sp_" + r["name"], "init": f"{f}:{r['name']}", "prefix": str(P / r["name"] / "prefix.pt"), "header": ""}) + "\n")
PY
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_hidden_states.py --run results/phase4/qwen/runs/qwen_instruct_unknown_casual_v1 \
  --model Qwen/Qwen2.5-7B --tag qwenbase --contexts qwen_selfdesc_overall_soft,qwen_selfdesc_traits_soft,generic --part $g/4 > logs/pq_hidden_soft_$g.log 2>&1 & done; wait
echo "=== $(date) done ==="
