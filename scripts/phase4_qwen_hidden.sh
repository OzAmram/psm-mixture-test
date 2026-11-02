#!/bin/bash
#SBATCH -A m2612_g -C gpu&hbm80g -q regular -N 1 -G 4 -c 128 -t 5:00:00 -J pq_hidden -o logs/pq_hidden_%j.out
# Qwen replication (2026-10-30): base-model final hidden states for a list of header families, 4 GPUs (contexts split k, k+4, ...).
# Idempotent (finished contexts are skipped), so a job cut off by its time limit is resumed by resubmitting.
#   sbatch scripts/phase4_qwen_hidden.sh <run (phase1 name or folder)> <model> <tag> <families>
#   e.g. sbatch scripts/phase4_qwen_hidden.sh instruct_unknown_casual_v1 Qwen/Qwen2.5-7B qwenbase legacy86,selfdesc_overall,selfdesc_traits,generic
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
RUN=$1; MODEL=$2; TAG=$3; F=$4; L=logs/pq_hidden_$(basename $RUN)__$TAG
TAGARG=(); [ "$TAG" != "-" ] && TAGARG=(--tag $TAG)
echo "=== $(date) hidden states: $RUN, $MODEL, tag $TAG, $F on $(hostname) ==="
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_hidden_states.py --run $RUN --model $MODEL "${TAGARG[@]}" --contexts $F --part $g/4 > ${L}_$g.log 2>&1 & done
wait
echo "=== $(date) done; errors: $(grep -l Traceback ${L}_*.log 2>/dev/null | tr '\n' ' ') ==="
