#!/bin/bash
# Phase 1 data collection + analysis. Runs the three sample-and-score jobs in parallel on separate GPUs
# (calibration 2-component, calibration 5-component, real generic fit), waits, then executes the analysis
# notebooks 1.1 and 1.2 headlessly.
#
# Direct use inside an allocation with >= 3 GPUs:   scripts/phase1_runs.sh
# Via Slurm (whole node, 4 GPUs):                    sbatch scripts/phase1_runs.sh
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -t 02:00:00
#SBATCH -J persona-p1
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh
Q=${QUESTIONS:-data/questions_v1.jsonl}
NQ=${MAX_QUESTIONS:-300}          # questions used for the real fit
NQC=${MAX_QUESTIONS_CALIB:-120}   # questions used for each calibration pool
NPQ=${N_PER_QUESTION:-8}
mkdir -p results/phase1 logs
echo "=== $(date) phase 1 runs: questions=$Q n_q=$NQ n_q_calib=$NQC n_per_q=$NPQ ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv

CUDA_VISIBLE_DEVICES=0 python scripts/phase1_sample_score.py --questions "$Q" --max-questions "$NQC" --n-per-question "$NPQ" \
    --framing unknown --source hhh:0.7,evil:0.3 --out results/phase1/calib_hhh70_evil30 > logs/p1_calib_hhh70_evil30.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/phase1_sample_score.py --questions "$Q" --max-questions "$NQC" --n-per-question "$NPQ" \
    --framing unknown --source hhh:0.5,fred:0.2,evil:0.15,sycophant:0.1,formal:0.05 --out results/phase1/calib_five > logs/p1_calib_five.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/phase1_sample_score.py --questions "$Q" --max-questions "$NQ" --n-per-question "$NPQ" \
    --framing unknown --source generic --out results/phase1/base_unknown_v1 > logs/p1_base_unknown_v1.log 2>&1 &
wait
echo "=== $(date) data collection done ==="
for d in calib_hhh70_evil30 calib_five base_unknown_v1; do
    [ -f results/phase1/$d/matrix.npz ] && echo "  $d: ok ($(wc -l < results/phase1/$d/rows.jsonl) rows)" || echo "  $d: MISSING (see logs/p1_$d.log)"
done
scripts/run_notebooks.sbatch notebooks/1.1_calibration.ipynb notebooks/1.2_base_fit.ipynb
echo "=== $(date) all done ==="
