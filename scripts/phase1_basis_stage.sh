#!/bin/bash
# Instruct-basis search (notebook 1.8): everything after the first batch, idempotent (finished family files are skipped).
#   GPU0: screen the big families on fit-half Instruct answers, select, score the selected subsets on all Instruct answers
#   GPU1: the small description families and the few-shot families on all Instruct answers
#   GPU2: the same on the base-assistant control
#   GPU3: elicitation (if not done) and the tempered-Instruct runs under generic + legacy86
# Inside an allocation:  srun --jobid=<id> --overlap -n 1 scripts/phase1_basis_stage.sh
# As a batch job:        sbatch scripts/phase1_basis_stage.sh
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 06:00:00
#SBATCH -J persona-basis
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
INST=instruct_unknown_casual_v1; BASE=base_unknown_casual_short_v1
SMALL=generic,legacy86,bigfive_poles,bigfive_profiles,persona_big5,values,olmo_spec
ELIC=inst_self,inst_fromdata,base_fromdata
FEW=fewshot_generic,fewshot_hhh
BIG="axis_roles axis_traits ipip_facets"
S="python scripts/phase1_basis_score.py"
echo "=== $(date) basis stage on $(hostname) ==="

gpu0() {
  for f in $BIG; do $S --run $INST --families $f --screen; done
  for f in $BIG; do [ -f data/prompts/basis/${f}_top.jsonl ] || python scripts/phase1_basis_select.py --run $INST --family $f --greedy 15 --top 15; done
  $S --run $INST --families axis_roles_top,axis_traits_top,ipip_facets_top
  $S --run $BASE --families axis_roles_top,axis_traits_top,ipip_facets_top
}
gpu1() {
  $S --run $INST --families $SMALL
  until [ -f data/prompts/basis/base_fromdata.jsonl ]; do sleep 30; done
  $S --run $INST --families $ELIC,$FEW
}
gpu2() {
  $S --run $BASE --families $SMALL
  until [ -f data/prompts/basis/base_fromdata.jsonl ]; do sleep 30; done
  $S --run $BASE --families $ELIC,$FEW
}
gpu3() {
  [ -f data/prompts/basis/base_fromdata.jsonl ] || python scripts/phase1_elicit_instruct.py --n 30 > logs/elicit_instruct.log 2>&1
  for T in 1.1 1.2 1.25; do
    [ -f results/phase1/instruct_T${T}_v1/rows.jsonl ] || python scripts/phase1_instruct_temp.py sample --temp $T --out results/phase1/instruct_T${T}_v1
    $S --run instruct_T${T}_v1 --families generic,legacy86
  done
}
CUDA_VISIBLE_DEVICES=0 gpu0 > logs/basis_stage_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 gpu1 > logs/basis_stage_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 gpu2 > logs/basis_stage_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 gpu3 > logs/basis_stage_gpu3.log 2>&1 &
wait
echo "=== $(date) basis stage done ==="
grep -h "components x\|Error\|Traceback" logs/basis_stage_gpu*.log | cut -c1-200
