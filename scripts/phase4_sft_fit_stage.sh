#!/bin/bash
# (2026-10-30, author's request) 1. base-model hidden states for the 35 verbalised self-descriptions (families *_v35), then
# their fit (tag sd_verb35); 2. hidden states of the SFT checkpoint on the Instruct answers, self-description basis (260 + the
# plain chat prompt as "generic") in the chat_turn framing (the user asks for a self-description, the description is the
# assistant's first reply), then arithmetic + geometric fits with the SFT output layer. 4 GPUs.
#   srun --jobid=<allocation> --overlap -n1 -G4 bash scripts/phase4_sft_fit_stage.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
RUN=instruct_unknown_casual_v1; H="python scripts/phase4_hidden_states.py --run $RUN"; SFT=allenai/Olmo-3-7B-Instruct-SFT
FR=${FRAMING:-chat_turn}; TAG=sft_$FR
echo "=== $(date) on $(hostname) ==="
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g $H --contexts selfdesc_overall_v35,selfdesc_traits_v35,generic --part $g/4 > logs/p4_v35_hidden_$g.log 2>&1 & done; wait
echo "$(date) v35 hidden states done"
bash scripts/phase4_bases_fit.sh sd_verb35
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g $H --model $SFT --framing $FR --tag $TAG --contexts selfdesc_overall,selfdesc_traits,generic --part $g/4 > logs/p4_${TAG}_hidden_$g.log 2>&1 & done; wait
echo "$(date) SFT hidden states done: $(ls /pscratch/sd/o/ozamram/persona_phase4/hidden/${RUN}__$TAG/lp_*.npy | wc -l) contexts"
python scripts/phase4_geom_fit.py --run ${RUN}__$TAG --model $SFT --fits A --devices 0,1,2,3 --contexts selfdesc_overall,selfdesc_traits \
  --fit-frac 0.8 --lambdas 0,1e-4,1e-3,1e-2,1e-1 --out results/phase4/exp1 --tag sd > logs/p4_${TAG}_fit.log 2>&1
tail -5 logs/p4_${TAG}_fit.log
echo "=== $(date) done ==="
