#!/bin/bash
# L = 2048 points of the length sweep (2026-11-18): batch 8 runs out of memory on an 80 GB A100, so these use
# --grad-checkpointing (same recipe otherwise), only after scripts/phase4_lensweep_ckpt_check.py passes on this GPU.
# Removes the claims and partial outputs of the earlier out-of-memory attempts first (no result.json in them).
#   bash scripts/phase4_lensweep_2048.sh  (after phase4_lensweep.sh has finished)
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source ./env.sh > /dev/null
O=results/phase4/lensweep
# 2026-11-19: L=1024 runs out of memory when the second lane (phase4_lensweep_lane2.sh) shares the GPU (seed 1 failed at
# 23:13): wait for that lane to finish, then rerun any L=1024 seed without a result alone on the GPU, before L=2048.
while pgrep -f '[p]hase4_lensweep_lane2.sh' > /dev/null; do sleep 60; done
for s in 0 1 2; do o=$O/L1024_lr5e-3_s$s
  [ -f $o/result.json ] && continue; rm -rf $o $o.claim $o.log; mkdir $o.claim; echo "$(hostname) $(date '+%F %T') rerun alone" > $o.claim/by
  echo "$(date '+%F %T') start L=1024 seed=$s (alone on the GPU)"
  python scripts/prompt_tune_base.py --model allenai/Olmo-3-1025-7B --rows results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl \
    --holdout-qids results/phase4/exp1/basis_test_qids.json --L 1024 --epochs 6 --lr 5e-3 --batch 8 --max-len 192 --seed $s \
    --n-sample-questions 0 --out $o > $o.log 2>&1
  [ -f $o/result.json ] && echo "$(date '+%F %T') done L=1024 seed=$s: $(grep 'best epoch by validation' $o.log | cut -c1-120)" || { echo "FAILED L=1024 seed=$s"; tail -3 $o.log; }
done
python scripts/phase4_lensweep_ckpt_check.py 2>&1 | grep -vE 'Loading|Warning|warn' | tee logs/lensweep_ckpt_check.log
grep -q '^PASS' logs/lensweep_ckpt_check.log || { echo "checkpointing check did not pass; L=2048 not run"; exit 1; }
for s in 0 1 2; do o=$O/L2048_lr5e-3_s$s
  [ -f $o/result.json ] && continue; rm -rf $o $o.claim $o.log; mkdir $o.claim; echo "$(hostname) $(date '+%F %T') ckpt" > $o.claim/by
  echo "$(date '+%F %T') start L=2048 seed=$s (gradient checkpointing)"
  python scripts/prompt_tune_base.py --model allenai/Olmo-3-1025-7B --rows results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl \
    --holdout-qids results/phase4/exp1/basis_test_qids.json --L 2048 --epochs 6 --lr 5e-3 --batch 8 --max-len 192 --seed $s \
    --n-sample-questions 0 --grad-checkpointing --out $o > $o.log 2>&1
  [ -f $o/result.json ] && echo "$(date '+%F %T') done L=2048 seed=$s: $(grep 'best epoch by validation' $o.log | cut -c1-120)" || { echo "FAILED L=2048 seed=$s"; tail -3 $o.log; }
done
echo "$(date '+%F %T') L=2048 finished"
