#!/bin/bash
# Qwen replication, stage 1 (2026-10-30): Qwen2.5-7B-Instruct answers on the 300 questions (same settings as
# results/phase1/instruct_unknown_casual_v1: casual suffix, 8 per question, T = 1, 80 new tokens, seed 0) on GPU 0, and
# Qwen Instruct self-descriptions (families qwen_selfdesc_overall / qwen_selfdesc_traits) on GPUs 1-3.
#   srun --jobid=<allocation> --overlap -n1 bash scripts/phase4_qwen_stage1.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
QI=Qwen/Qwen2.5-7B-Instruct; QB=Qwen/Qwen2.5-7B; RUN=results/phase4/qwen/runs/qwen_instruct_unknown_casual_v1
echo "=== $(date) qwen stage 1 on $(hostname) ==="
[ -f $RUN/rows.jsonl ] || CUDA_VISIBLE_DEVICES=0 python scripts/phase1_instruct_sample.py --instruct-model $QI --base-model $QB \
  --n-per-question 8 --framing unknown --register casual --seed 0 --out $RUN > logs/pq_sample.log 2>&1 &
if [ ! -f data/prompts/basis/qwen_selfdesc_traits.jsonl ]; then
  for g in 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_selfdesc_elicit.py --model $QI --prefix qwen_ --part $((g-1))/3 > logs/pq_elicit_$g.log 2>&1 & done
  wait
  python scripts/phase4_selfdesc_elicit.py --prefix qwen_ --merge > logs/pq_elicit_merge.log 2>&1
fi
wait
echo "=== $(date) done: $(wc -l < $RUN/rows.jsonl) answers; $(wc -l < data/prompts/basis/qwen_selfdesc_overall.jsonl) overall, $(wc -l < data/prompts/basis/qwen_selfdesc_traits.jsonl) traits ==="
