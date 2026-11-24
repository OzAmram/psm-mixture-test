#!/bin/bash
# Experiment 3, geometric fit A at the three RL checkpoints (author's request, 2026-11-19; skipped on 2026-10-27 for storage).
# Same as scripts/phase4_exp3_geom.sh did for SFT and DPO (base-model hidden states on that checkpoint's answers under
# the 87 contexts legacy86 + generic, then `phase4_geom_fit.py --run <run> --fits A`, default lambda grid, full fit half),
# on one GPU, one step at a time; after a step's fit has hold-out results its h_*.u16 files (~115 GB, regenerable) are
# deleted to fit the disk (index.npz and lp_*.npy are kept). Skips steps whose fits_A.json already has a hold-out.
#   bash scripts/phase4_exp3_geom_rl.sh > logs/p4e3_rl_<host>.out 2>&1
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source ./env.sh > /dev/null
say() { echo "$(date '+%F %T') [$(hostname)] $*"; }
for s in 100 200 300; do r=instruct_step${s}_v1; F=results/phase4/exp1/$r/fits_A.json; H=$SCRATCH/persona_phase4/hidden/$r
  grep -q '"holdout"' $F 2>/dev/null && { say "$r: fit done, skipping"; continue; }
  say "$r: hidden states"; python scripts/phase4_hidden_states.py --run results/phase4/exp3/runs/$r --contexts legacy86,generic --part 0/1 > logs/p4e3_hidden_$r.log 2>&1 \
    || { say "$r: hidden states FAILED"; tail -3 logs/p4e3_hidden_$r.log; exit 1; }
  say "$r: fit A"; python scripts/phase4_geom_fit.py --run $r --fits A --devices 0 > logs/p4e3_fitA_$r.log 2>&1
  grep -q '"holdout"' $F 2>/dev/null || { say "$r: fit FAILED (hidden states kept)"; tail -3 logs/p4e3_fitA_$r.log; exit 1; }
  say "$r: $(tail -4 logs/p4e3_fitA_$r.log | tr '\n' ' ' | cut -c1-200)"
  rm -f $H/h_*.u16; say "$r: deleted h_*.u16 ($(du -sh --apparent-size $H | cut -f1) left)"
done
say "RL steps done"
