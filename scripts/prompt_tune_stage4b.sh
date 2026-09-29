#!/bin/bash
# Extra sweep points on the idle second GPU: lower learning rate at small and large L, Phase 1 instruct samples.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
for cfg in "32 5e-3" "512 5e-3" "128 1e-3"; do
  set -- $cfg; L=$1; LR=$2
  echo "=== $(date) L=$L lr=$LR ==="
  python scripts/prompt_tune_base.py --L $L --epochs 10 --lr $LR --out results/prompt_tune/sweep_L${L}_lr$LR > logs/prompt_tune_sweep_L${L}_lr$LR.log 2>&1
  grep -E "best epoch|Traceback|entropy" logs/prompt_tune_sweep_L${L}_lr$LR.log | cut -c1-200
done
echo "=== $(date) prompt-tune stage 4b all done ==="
