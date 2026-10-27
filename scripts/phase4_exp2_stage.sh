#!/bin/bash
# Experiment 2 (subspace and span tests, notebook 4.2): GPU part. Idempotent.
#   1. elicit ~400 more base-model personas (same prompt/filters as the original 80, seed 1, deduplicated against them)
#   2. build the family elicited_p4 and the 2,000 answer columns (CPU)
#   3. score matrices X_M (contexts x columns, raw transcript format) for base, SFT, DPO, Instruct, and (separate 1-GPU job, gpuq) Qwen2.5-7B base
#   4. span-regression targets: chat-template log P of both runs' answers under Instruct, SFT and DPO
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 8:00:00
#SBATCH -J persona-p4e2
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
E=results/phase4/exp2
echo "=== $(date) phase 4 exp 2 on $(hostname) ==="
[ -f data/prompts/personas_elicited_p4/_meta.json ] || CUDA_VISIBLE_DEVICES=0 python scripts/phase1_elicit_personas.py --n 640 --seed 1 \
    --out-dir data/prompts/personas_elicited_p4 --exclude-dir data/prompts/personas_elicited --name-prefix x > logs/p4e2_elicit.log 2>&1
[ -f $E/columns.jsonl ] || python scripts/phase4_exp2_build.py > logs/p4e2_build.log 2>&1
M() { python scripts/phase4_score_matrix.py --model $1 --tag $2; }
T() { python scripts/phase4_self_tokens.py --run $1 --model $2 --tag $3; }
gpu0() { M allenai/Olmo-3-1025-7B base; }
gpuq() { M Qwen/Qwen2.5-7B qwen; }   # run as its own 1-GPU job: sbatch -G 1 -c 32 -t 8:00:00 scripts/phase4_exp2_stage.sh gpuq:0
gpu1() { M allenai/Olmo-3-7B-Instruct-SFT sft; T base_unknown_casual_short_v1 allenai/Olmo-3-7B-Instruct-SFT base_run__sft; T instruct_unknown_casual_v1 allenai/Olmo-3-7B-Instruct-SFT instruct_run__sft; }
gpu2() { M allenai/Olmo-3-7B-Instruct-DPO dpo; T base_unknown_casual_short_v1 allenai/Olmo-3-7B-Instruct-DPO base_run__dpo; T instruct_unknown_casual_v1 allenai/Olmo-3-7B-Instruct-DPO instruct_run__dpo; }
gpu3() { M allenai/Olmo-3-7B-Instruct instruct; T base_unknown_casual_short_v1 allenai/Olmo-3-7B-Instruct base_run__instruct; }
PAIRS=${*:-"gpu0:0 gpu1:1 gpu2:2 gpu3:3"}
for p in $PAIRS; do fn=${p%%:*}; g=${p##*:}; CUDA_VISIBLE_DEVICES=$g $fn > logs/p4e2_$fn.log 2>&1 & done
wait
echo "=== $(date) done ==="
grep -h "Error\|Traceback\|done" logs/p4e2_*.log | cut -c1-200
