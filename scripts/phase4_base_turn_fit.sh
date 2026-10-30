#!/bin/bash
# (2026-10-30) Base model, self-description basis (260 + generic) in the turn_bare framing (best base framing on the fit half
# of the framing screen: "User: Describe yourself in the third person..." / "Assistant: <description>" / then the question):
# hidden states, then arithmetic + geometric fits. Compare with bases/sd (transcript framing).
#SBATCH -A m2612_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 4:00:00
#SBATCH -J persona-p4turn
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
RUN=instruct_unknown_casual_v1; TAG=base_turn_bare
echo "=== $(date) on $(hostname) ==="
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_hidden_states.py --run $RUN --framing turn_bare --tag $TAG \
  --contexts selfdesc_overall,selfdesc_traits,generic --part $g/4 > logs/p4_${TAG}_hidden_$g.log 2>&1 & done; wait
echo "$(date) hidden states done"
python scripts/phase4_geom_fit.py --run ${RUN}__$TAG --fits A --devices 0,1,2,3 --contexts selfdesc_overall,selfdesc_traits \
  --fit-frac 0.8 --lambdas 0,1e-4,1e-3,1e-2,1e-1 --out results/phase4/exp1 --tag sd > logs/p4_${TAG}_fit.log 2>&1
tail -5 logs/p4_${TAG}_fit.log; echo "=== $(date) done ==="
