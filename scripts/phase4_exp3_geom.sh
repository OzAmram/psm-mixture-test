#!/bin/bash
# Experiment 3, geometric curve: final hidden states of the base model on the SFT and DPO answers under the 87 contexts
# (one GPU per run, in parallel), then geometric fit A for each (2 GPUs, one run after the other). The final Instruct
# checkpoint's fit A is the Experiment 1 one. Idempotent apart from the fits (skipped when fits_A.json has a hold-out).
#   srun --jobid=<interactive allocation> --overlap -n1 -G2 bash scripts/phase4_exp3_geom.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
H="python scripts/phase4_hidden_states.py --contexts legacy86,generic"
echo "=== $(date) exp 3 geometric on $(hostname) ==="
RUNS="sft_unknown_casual_v1 dpo_unknown_casual_v1"; g=0
for r in $RUNS; do CUDA_VISIBLE_DEVICES=$g $H --run $r --part 0/1 > logs/p4e3_hidden_$r.log 2>&1 & g=$((g + 1)); done
wait
for r in $RUNS; do
  grep -q '"holdout"' results/phase4/exp1/$r/fits_A.json 2>/dev/null || \
    CUDA_VISIBLE_DEVICES=0,1 python scripts/phase4_geom_fit.py --run $r --fits A --devices 0,1 > logs/p4e3_fitA_$r.log 2>&1
done
echo "=== $(date) done ==="
