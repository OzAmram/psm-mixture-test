#!/bin/bash
# Persona bases fitted to other post-training checkpoints and to the learned soft prefix (notebook 1.9). Idempotent.
#   GPU0: Olmo-3-7B-Instruct-SFT: sample 300 x 8 at T = 1 (same prompts as instruct_unknown_casual_v1), score all bases;
#         then the discrete (GCG) header search on the Instruct answers (phase1_hardprompt.py), scored on Instruct + base control
#   GPU1: Olmo-3-7B-Instruct-DPO: same; then the Instruct model describes the soft-prefix samples (inst_from_softprefix),
#         scored on the Instruct answers and on the soft-prefix samples
#   GPU2: base + learned soft prefix (results/prompt_tune/sweep_L128_lr5e-3): same; then the SALVE positive control
#         on the Instruct weights (scripts/prompt_tune_salve_instruct.sh)
#   GPU3: the 86 legacy personas each followed by the same 4 Instruct exemplars (legacy86_fs4), on the Instruct answers; then
#         Olmo-3-7B-RL-Zero-General (RL from base, no SFT; writes long essays, capped at 80 tokens like every run)
# Few-shot families for each run are built from that run's OWN fit-half answers (fewshot_generic_<tag>, fewshot_hhh_<tag>).
# Inside an allocation:  srun --jobid=<id> --overlap -n 1 scripts/phase1_ckpt_stage.sh
# As a batch job:        sbatch scripts/phase1_ckpt_stage.sh
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 14:00:00
#SBATCH -J persona-ckpt
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"   # under sbatch the script runs from a spool copy
source env.sh > /dev/null
FAMS=generic,legacy86,bigfive_poles,bigfive_profiles,persona_big5,values,olmo_spec,style,base_fromdata,inst_self,inst_fromdata,axis_roles_top,axis_traits_top,ipip_facets_top
S="python scripts/phase1_basis_score.py"
echo "=== $(date) checkpoint stage on $(hostname) ==="

score_run() {   # $1 run, $2 tag
  [ -f data/prompts/basis/fewshot_hhh_$2.jsonl ] || python scripts/phase1_build_basis.py fewshot_for:$1:$2
  $S --run $1 --families $FAMS,fewshot_generic_$2,fewshot_hhh_$2
}
ckpt() {        # $1 HF model, $2 tag
  local run=$2_unknown_casual_v1
  [ -f results/phase1/$run/rows.jsonl ] || python scripts/phase1_instruct_temp.py sample --temp 1.0 --model $1 --source $2 --out results/phase1/$run
  score_run $run $2
}
gpu0() {
  ckpt allenai/Olmo-3-7B-Instruct-SFT sft
  [ -f data/prompts/basis/hardprompt.jsonl ] || python scripts/phase1_hardprompt.py --lambdas 0,0.1,0.3 --steps 300
  $S --run instruct_unknown_casual_v1 --families hardprompt; $S --run base_unknown_casual_short_v1 --families hardprompt
}
gpu1() {
  ckpt allenai/Olmo-3-7B-Instruct-DPO dpo
  until [ -f results/phase1/softprefix_unknown_casual_v1/rows.jsonl ]; do sleep 60; done
  [ -f data/prompts/basis/inst_from_softprefix.jsonl ] || python scripts/phase1_elicit_instruct.py --n 30 --from-run softprefix_unknown_casual_v1 --tag softprefix
  $S --run instruct_unknown_casual_v1 --families inst_from_softprefix; $S --run softprefix_unknown_casual_v1 --families inst_from_softprefix
}
gpu2() {
  [ -f results/phase1/softprefix_unknown_casual_v1/rows.jsonl ] || python scripts/phase1_softprefix.py sample --out results/phase1/softprefix_unknown_casual_v1
  score_run softprefix_unknown_casual_v1 softprefix
  scripts/prompt_tune_salve_instruct.sh
}
gpu3() { $S --run instruct_unknown_casual_v1 --families legacy86_fs4; ckpt allenai/Olmo-3-7B-RL-Zero-General rlzero; }
# optional arguments "fn:gpu" run only those work lists; default: all four
PAIRS=${*:-"gpu0:0 gpu1:1 gpu2:2 gpu3:3"}
for p in $PAIRS; do fn=${p%%:*}; g=${p##*:}; CUDA_VISIBLE_DEVICES=$g $fn > logs/ckpt_stage_$fn.log 2>&1 & done
wait
echo "=== $(date) checkpoint stage done ==="
grep -h "components x\|done:\|Error\|Traceback" logs/ckpt_stage_*.log | cut -c1-200
