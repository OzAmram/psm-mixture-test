#!/bin/bash
# Experiment 1 (geometric persona fit, notebook 4.1): GPU part. Idempotent; outputs skip if present.
#   GPU0/GPU1: final hidden states of the base model at every Instruct answer position, under the 87 contexts (halves)
#   GPU2:      the same for the base assistant's answers (control)
#   GPU3:      per-token Instruct self log P; base soft prefix retrained on the basis split (training half only) and scored
#              with the standard convention; synthetic geometric-combination answers and their hidden states
# Hidden states go to $SCRATCH/persona_phase4/hidden (about 100 GB per run).
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 6:00:00
#SBATCH -J persona-p4e1
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"   # under sbatch the script runs from a spool copy
source ./env.sh > /dev/null
H="python scripts/phase4_hidden_states.py --contexts legacy86,generic"
E=results/phase4/exp1
echo "=== $(date) phase 4 exp 1 on $(hostname) ==="
gpu0() { $H --run instruct_unknown_casual_v1 --part 0/2; }
gpu1() { $H --run instruct_unknown_casual_v1 --part 1/2; }
gpu2() { $H --run base_unknown_casual_short_v1 --part 0/1; }
gpu3() {
  [ -f results/phase4/self_tokens/instruct_unknown_casual_v1.npz ] || python scripts/phase4_self_tokens.py --run instruct_unknown_casual_v1 --model allenai/Olmo-3-7B-Instruct
  [ -f $E/prefix_instruct_L128/prefix.pt ] || python scripts/prompt_tune_base.py --holdout-qids $E/basis_test_qids.json --L 128 --epochs 8 --lr 5e-3 --seed 0 --out $E/prefix_instruct_L128
  [ -f $E/prefix_instruct_L128/scores.npz ] || python scripts/phase4_prefix_score.py --prefix $E/prefix_instruct_L128/prefix.pt --run instruct_unknown_casual_v1 --out $E/prefix_instruct_L128/scores
  [ -f $E/synthetic_run/rows.jsonl ] || python scripts/phase4_synthetic.py --personas hhh,e14,fred --c 0.7,0.6,-0.3 --n-per-question 2 --seed 0 --out $E/synthetic_run
  $H --run $E/synthetic_run --part 0/1
}
PAIRS=${*:-"gpu0:0 gpu1:1 gpu2:2 gpu3:3"}
for p in $PAIRS; do fn=${p%%:*}; g=${p##*:}; CUDA_VISIBLE_DEVICES=$g $fn > logs/p4e1_$fn.log 2>&1 & done
wait
echo "=== $(date) done ==="
grep -h "Error\|Traceback\|done\|hold-out" logs/p4e1_*.log | cut -c1-200
