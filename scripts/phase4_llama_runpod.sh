#!/bin/bash
# Llama 3.1 8B replication on RunPod (2026-11-02): the steps of scripts/phase4_llama.sh (same commands and settings), split so
# each can run on any single-GPU pod that mounts /workspace. Large fits are dropped (author's decision; NOTES.md 2026-11-02):
# fits only lsdo, lsd_soft, legacy87, olmospec; soft contexts selected by the arithmetic greedy picks only (k=35).
#   bash scripts/phase4_llama_runpod.sh self                     Instruct self scores
#   bash scripts/phase4_llama_runpod.sh hidden <families> <k/n>  base hidden states, part k of n (by context)
#   bash scripts/phase4_llama_runpod.sh rows                     soft-context training rows (needs self + generic hidden)
#   bash scripts/phase4_llama_runpod.sh framing <0-3>            framing screen, one group
#   bash scripts/phase4_llama_runpod.sh contam                   contamination check (base and Instruct)
#   bash scripts/phase4_llama_runpod.sh fit <tag ...>            geometric fits on one GPU
#   bash scripts/phase4_llama_runpod.sh select                   soft-context selection (CPU)
#   bash scripts/phase4_llama_runpod.sh soft <k/n>               train soft contexts, part k of n (each context claimed via
#                                                               <dir>.claim, so pods may share one part; delete a dead pod's claim to retry)
#   bash scripts/phase4_llama_runpod.sh softfam                  write the *_soft families (CPU)
# Logs: logs/pl_runpod_<step>_<part>.log (one per step and part, so pods do not overwrite each other).
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source ./env.sh > /dev/null
LI=meta-llama/Llama-3.1-8B-Instruct; LB=meta-llama/Llama-3.1-8B; D=results/phase4/llama; NAME=llama_instruct_unknown_casual_v1
RUNF=$D/runs/$NAME; HN=${NAME}__llamabase; H=${SCRATCH}/persona_phase4/hidden/$HN; SELF=results/phase4/self_tokens/$NAME.npz
LO=llama_selfdesc_overall; LT=llama_selfdesc_traits
mkdir -p $D/framing $D/selfdesc/prefix logs
say() { echo "$(date '+%F %T') [$(hostname)] $*"; }
step=$1; shift
case $step in
  self) [ -f $SELF ] || python scripts/phase4_self_tokens.py --run $RUNF --model $LI > logs/pl_runpod_self.log 2>&1 ;;
  hidden) python scripts/phase4_hidden_states.py --run $RUNF --model $LB --tag llamabase --contexts $1 --part $2 \
            > logs/pl_runpod_hidden_${1//,/+}_${2/\//of}.log 2>&1 ;;
  rows) [ -f $D/rows/$NAME.jsonl ] || python scripts/phase4_exp3_rows.py --run $RUNF --self $SELF --out-dir $D/rows --generic-hidden $H > logs/pl_runpod_rows.log 2>&1 ;;
  framing) FR=(transcript,turn_bare turn_describe,play story,card story_chat,hyb_char); g=$1; o=$D/framing/scores_llamabase_$g.json
    [ -f $o ] || python scripts/phase4_framing.py --model $LB --run $RUNF --families $LO,$LT --ref-model $LB --framings ${FR[$g]} --out $o > logs/pl_runpod_framing_$g.log 2>&1 ;;
  contam)
    [ -f $D/framing/contam_llamabase.json ] || python scripts/phase4_framing.py --model $LB --template-from $LI --run $RUNF --families $LO,$LT --ref-model $LB \
      --framings transcript,chat_plain --generic-only transcript,chat_plain --out $D/framing/contam_llamabase.json > logs/pl_runpod_contam_b.log 2>&1
    [ -f $D/framing/contam_llamainstruct.json ] || python scripts/phase4_framing.py --model $LI --run $RUNF --families $LO,$LT --ref-model $LB \
      --framings chat_plain --out $D/framing/contam_llamainstruct.json > logs/pl_runpod_contam_i.log 2>&1 ;;
  fit) for t in "$@"; do case $t in lsdo|lsd_soft) RC=$LO,$LT ;; *) RC=legacy86,olmo_spec,$LO,$LT ;; esac
         RUN=$HN MODEL=$LB ROWCTX=$RC DEVICES=0 bash scripts/phase4_bases_fit.sh $t; done ;;
  select) [ -f $D/selfdesc/selected.txt ] || python scripts/phase4_selfdesc_select.py --arith-only --k 35 --hid $HN --prefix llama_ \
            --fit results/phase4/exp1/$HN/bases/lsdo/fits_A.json --out $D/selfdesc > logs/pl_runpod_select.log 2>&1 ;;
  soft) k=${1%/*}; n=${1#*/}; arr=(generic:generic $(cat $D/selfdesc/selected.txt))
    for ((i = k; i < ${#arr[@]}; i += n)); do fn=${arr[$i]}; o=$D/selfdesc/prefix/${fn#*:}; [ -f $o/result.json ] && continue
      mkdir -p $(dirname $o); mkdir $o.claim 2>/dev/null || continue   # atomic claim: several pods can run "soft 0/1" at once
      echo "$(hostname) $(date '+%F %T')" > $o.claim/by
      python scripts/prompt_tune_base.py --model $LB --rows $D/rows/$NAME.jsonl --holdout-qids results/phase4/exp1/basis_test_qids.json \
        --init-from $fn --L 0 --epochs 3 --lr 5e-3 --seed 0 --n-sample-questions 0 --out $o > $o.log 2>&1 || say "soft $fn FAILED"; done ;;
  softfam) python - <<'PY'
import json
from pathlib import Path
B = Path("data/prompts/basis"); P = Path("results/phase4/llama/selfdesc/prefix")
for f in ["llama_selfdesc_overall", "llama_selfdesc_traits"]:
    rows = [json.loads(l) for l in open(B / f"{f}.jsonl")]
    with open(B / f"{f}_soft.jsonl", "w") as o:
        for r in rows:
            if (P / r["name"] / "result.json").exists():
                o.write(json.dumps({"family": f + "_soft", "name": "sp_" + r["name"], "init": f"{f}:{r['name']}", "prefix": str(P / r["name"] / "prefix.pt"), "header": ""}) + "\n")
PY
  ;;
  *) echo "unknown step $step"; exit 1 ;;
esac
say "$step $* done (exit $?)"
