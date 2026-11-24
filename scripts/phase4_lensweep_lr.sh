#!/bin/bash
# Learning-rate scan for short soft contexts (author's request, 2026-11-19: "run optimal learning rate scans for the short
# prefixes, and make them not be undertrained"). Same recipe as scripts/phase4_lensweep.sh except: 12 epochs (not 6; the lr
# schedule decays over all steps, so it also decays more slowly), lr chosen per length on VALIDATION (never the hold-out).
#   phase 1: L in 1..128 (powers of 2) x lr in {5e-3, 1e-2, 2e-2, 5e-2, 1e-1}, seed 0
#   phase 2: per L, the lr with the best validation log P; if it is the largest tried, also 2e-1 (repeat while it wins);
#            if that run's best epoch is its last (12), rerun it with 24 epochs; then seeds 1 and 2 with the chosen setting.
# Runs are claimed (mkdir <dir>.claim), so two copies (lanes) share the work; phase 2 waits for a length's grid to finish.
#   setsid nohup bash scripts/phase4_lensweep_lr.sh > logs/lensweep_lr_<lane>.out 2>&1 < /dev/null &
# Output results/phase4/lensweep_lr/L<L>_lr<lr>_s<seed>_e<epochs>/; decisions in .../choice_L<L>.json.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source ./env.sh > /dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=results/phase4/lensweep_lr; mkdir -p $O
LS="1 2 4 8 16 32 64 128"; LRS="5e-3 1e-2 2e-2 5e-2 1e-1"
say() { echo "$(date '+%F %T') [$(hostname)] $*"; }
run() { local L=$1 lr=$2 s=$3 e=$4 o=$O/L$1_lr$2_s$3_e$4
  [ -f $o/result.json ] && return 0; mkdir $o.claim 2>/dev/null || return 0; echo "$(hostname) $(date '+%F %T')" > $o.claim/by
  say "start L=$L lr=$lr seed=$s epochs=$e"
  python scripts/prompt_tune_base.py --model allenai/Olmo-3-1025-7B --rows results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl \
    --holdout-qids results/phase4/exp1/basis_test_qids.json --L $L --epochs $e --lr $lr --batch 8 --max-len 192 --seed $s \
    --n-sample-questions 0 --out $o > $o.log 2>&1
  [ -f $o/result.json ] && say "done  L=$L lr=$lr seed=$s epochs=$e: $(grep 'best epoch by validation' $o.log | cut -c1-110)" \
    || { say "FAILED L=$L lr=$lr seed=$s epochs=$e"; tail -2 $o.log; touch $o.failed; }
}
best() {  # prints "lr epochs best_epoch val" of the best-validation run among the given dirs
  python3 - "$@" <<'PY'
import json, sys, re
from pathlib import Path
rows = []
for d in sys.argv[1:]:
    f = Path(d) / "result.json"
    if not f.exists(): continue
    r = json.loads(f.read_text()); b = max(r["log"], key=lambda e: e["val_logp"])
    lr, e = re.search(r"_lr([^_]+)_s\d+_e(\d+)$", d).groups(); rows.append((b["val_logp"], lr, e, b["epoch"]))
v, lr, e, be = max(rows); print(lr, e, be, v)
PY
}
for L in $LS; do for lr in $LRS; do run $L $lr 0 12; done; done
for L in $LS; do
  mkdir $O/phase2_L$L.claim 2>/dev/null || continue   # one lane does a length's selection + seeds; the other takes the next length
  while ls -d $O/L${L}_lr*_s0_e12.claim > /dev/null 2>&1 && [ $(ls $O/L${L}_lr*_s0_e12/result.json $O/L${L}_lr*_s0_e12.failed 2>/dev/null | wc -l) -lt $(ls -d $O/L${L}_lr*_s0_e12.claim | wc -l) ]; do sleep 60; done
  grid="$LRS"; read lr e be v < <(best $(for x in $grid; do echo $O/L${L}_lr${x}_s0_e12; done))
  for up in 2e-1 5e-1; do [ "$lr" = "${grid##* }" ] || break; run $L $up 0 12; grid="$grid $up"
    read lr e be v < <(best $(for x in $grid; do echo $O/L${L}_lr${x}_s0_e12; done)); done
  if [ "$be" = "12" ]; then run $L $lr 0 24; read lr2 e2 be2 v2 < <(best $O/L${L}_lr${lr}_s0_e12 $O/L${L}_lr${lr}_s0_e24); e=$e2; be=$be2; fi
  echo "{\"L\": $L, \"lr\": \"$lr\", \"epochs\": $e, \"best_epoch_seed0\": $be, \"grid\": \"$grid\"}" > $O/choice_L$L.json
  say "L=$L chosen lr=$lr epochs=$e (seed-0 best epoch $be, val $v)"
  for s in 1 2; do run $L $lr $s $e; done
done
say "lr scan lane finished"
