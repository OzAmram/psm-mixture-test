#!/bin/bash
# Experiment 3 (soft-prefix gap across post-training stages, notebook 4.3). Idempotent: every step skips if its output
# exists, so the script can be rerun in successive allocations until done. Usage (one checkpoint list per GPU):
#   bash scripts/phase4_exp3_stage.sh "sft:0 dpo:1 step100:2 step200:3"     (on an allocation with 4 GPUs)
# "a,b:g" runs checkpoints a then b on GPU g.
# Per checkpoint M_k:
#   answers  SFT/DPO/final: the existing Phase 1 runs; RL steps 100/200/300: sampled here at T = 1 with the same questions,
#            suffix, system message and 80-token limit (phase1_instruct_temp.py; the chat-template text is identical for
#            every checkpoint, checked 2026-10-26)
#   base scores of legacy86 + generic for new runs (arithmetic mixture and generic reference)
#   self     per-token self log P in the chat template (phase4_self_tokens.py)
#   prefixes on the BASE model, trained on M_k's basis-split training half: L128/lr5e-3 (the notebook 3.6 setting) and
#            L in {16, 64} x lr in {5e-3, 2e-2}; optimizer control: L128/lr5e-3 on M_k itself. Each scored with
#            phase4_prefix_score.py (standard transcript convention).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
E=results/phase4/exp3; Q=results/phase4/exp1/basis_test_qids.json; BASE=allenai/Olmo-3-1025-7B
HUB=/global/cfs/cdirs/m2612/ozamram/hf_cache/hub/models--allenai--Olmo-3-7B-Instruct
snap() { echo $HUB/snapshots/$(cat $HUB/refs/$1); }
echo "=== $(date) phase 4 exp 3 on $(hostname): $* ==="

ckpt() {   # name run model
  local k=$1 run=$2 model=$3 rn; rn=$(basename $run)
  if [ ! -f $run/rows.jsonl ]; then python scripts/phase1_instruct_temp.py sample --temp 1.0 --model $model --source $k --out $run || return 1; fi
  [ -f $run/basis/legacy86.npz ] && [ -f $run/basis/generic.npz ] || python scripts/phase1_basis_score.py --run $run --families legacy86,generic || return 1
  [ -f results/phase4/self_tokens/$rn.npz ] || python scripts/phase4_self_tokens.py --run $run --model $model || return 1
  [ -f $E/rows/$rn.jsonl ] || python scripts/phase4_exp3_rows.py --run $run --self results/phase4/self_tokens/$rn.npz || return 1
  for cfg in "base 128 5e-3" "ctrl 128 5e-3" "base 16 5e-3" "base 16 2e-2" "base 64 5e-3" "base 64 2e-2"; do
    set -- $cfg; local who=$1 L=$2 lr=$3 m=$BASE; [ $who = ctrl ] && m=$model
    local o=$E/prefix/${k}_${who}_L${L}_lr${lr}
    [ -f $o/prefix.pt ] || python scripts/prompt_tune_base.py --model $m --ref-model $model --rows $E/rows/$rn.jsonl --holdout-qids $Q \
        --L $L --epochs 8 --lr $lr --seed 0 --out $o || continue
    [ -f $o/scores.npz ] || python scripts/phase4_prefix_score.py --prefix $o/prefix.pt --model $m --run $run --out $o/scores
  done
}
sft()      { ckpt sft      results/phase1/sft_unknown_casual_v1      allenai/Olmo-3-7B-Instruct-SFT; }
dpo()      { ckpt dpo      results/phase1/dpo_unknown_casual_v1      allenai/Olmo-3-7B-Instruct-DPO; }
step100()  { ckpt step100  $E/runs/instruct_step100_v1  $(snap step_100); }
step200()  { ckpt step200  $E/runs/instruct_step200_v1  $(snap step_200); }
step300()  { ckpt step300  $E/runs/instruct_step300_v1  $(snap step_300); }
final()    { ckpt final    results/phase1/instruct_unknown_casual_v1 allenai/Olmo-3-7B-Instruct; }

for p in ${1:-"sft:0 dpo:1 step100:2 step200:3"}; do   # "a,b:g" runs checkpoints a then b on GPU g
  g=${p##*:}; fns=${p%:*}
  ( export CUDA_VISIBLE_DEVICES=$g; for fn in ${fns//,/ }; do $fn; done ) > logs/p4e3_${fns//,/_}.log 2>&1 &
done
wait
echo "=== $(date) done ==="
