#!/bin/bash
# Experiment 1 fits with the fp32-logit objective (2026-10-26 rerun), on a 4-GPU allocation:
#   GPUs 0,1: synthetic calibration (if not done), Instruct fit A, then the base-answer control (A, C, B)
#   GPUs 2,3: Instruct fits C, B, D, E
# Each invocation writes its own fits_<letters>.json after every fit, so a cut-off run keeps what finished.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
F="python scripts/phase4_geom_fit.py"; X=results/phase4/exp1
echo "=== $(date) exp 1 refit on $(hostname) ==="
(
  export CUDA_VISIBLE_DEVICES=0,1
  [ -f $X/synthetic_run/fits_TACBDE.json ] && grep -q '"holdout"' $X/synthetic_run/fits_TACBDE.json || $F --run synthetic_run --fits T,A,C,B,D,E --devices 0,1 > logs/p4e1_fit32_synthetic.log 2>&1
  $F --run instruct_unknown_casual_v1 --fits A --devices 0,1 > logs/p4e1_fit32_instruct_A.log 2>&1
  $F --run base_unknown_casual_short_v1 --fits A,C,B --devices 0,1 > logs/p4e1_fit32_base_ACB.log 2>&1
) &
(
  export CUDA_VISIBLE_DEVICES=2,3
  $F --run instruct_unknown_casual_v1 --fits C,B,D,E --devices 0,1 > logs/p4e1_fit32_instruct_CBDE.log 2>&1
) &
wait
echo "=== $(date) done ==="
