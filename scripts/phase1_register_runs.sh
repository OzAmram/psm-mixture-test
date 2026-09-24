#!/bin/bash
# Register-controlled Phase 1 run: sample the generic assistant under the `unknown` framing WITH the shared
# clause "The assistant speaks in a casual, conversational tone." (appended identically to the generic prompt
# and to every component), score under the 6 hand-written components, then rescore under the 80 elicited
# components on all visible GPUs, then execute the comparison notebook 1.5.
#
# Direct use inside an allocation:   scripts/phase1_register_runs.sh
# Via Slurm (whole node, 4 GPUs):    sbatch scripts/phase1_register_runs.sh
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -t 02:00:00
#SBATCH -J persona-reg
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh
RUN=results/phase1/base_unknown_casual_v1
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "=== $(date) register run: $RUN on $NGPU GPUs ==="

if [ ! -f "$RUN/matrix.npz" ]; then
    CUDA_VISIBLE_DEVICES=0 python scripts/phase1_sample_score.py --questions data/questions_v1_shuffled.jsonl --max-questions 300 \
        --n-per-question 8 --framing unknown --register casual --source generic --out "$RUN" > logs/p1_base_unknown_casual_v1.log 2>&1
    echo "=== $(date) sampling+scoring: $(grep -E '^done|Error' logs/p1_base_unknown_casual_v1.log | tail -1) ==="
fi
MAXQ=100000 scripts/phase1_rescore_elicited.sh "$RUN" data/prompts/personas_elicited "$NGPU"
scripts/run_notebooks.sbatch notebooks/1.5_register_control.ipynb
echo "=== $(date) all done ==="
