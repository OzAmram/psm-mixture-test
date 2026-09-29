#!/bin/bash
# L sweep on the Phase 1 instruct samples (validation-based early stopping), after stage 2 on the same GPU.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
until grep -q "prompt-tune stage 2 all done" logs/prompt_tune_stage2.log 2>/dev/null; do sleep 60; done
for L in 8 32 128 512; do
  echo "=== $(date) L=$L ==="
  python scripts/prompt_tune_base.py --L $L --epochs 8 --lr 2e-2 --out results/prompt_tune/sweep_L$L > logs/prompt_tune_sweep_L$L.log 2>&1
  grep -E "best epoch|Traceback|entropy" logs/prompt_tune_sweep_L$L.log | cut -c1-200
done
echo "=== $(date) prompt-tune stage 4 all done ==="
