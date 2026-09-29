#!/bin/bash
# Initialisation comparison on the Phase 1 instruct samples, L=128, lr 5e-3: Askell HHH description vs random tokens
# (the header-text init at the same L / lr runs in stage 4).
cd "$(dirname "$0")/.."
source env.sh > /dev/null
for init in hhh random; do
  echo "=== $(date) init=$init ==="
  python scripts/prompt_tune_base.py --L 128 --epochs 8 --lr 5e-3 --init $init --out results/prompt_tune/init_${init}_L128 > logs/prompt_tune_init_$init.log 2>&1
  grep -E "prefix init|prefix at init|best epoch|entropy|Traceback" logs/prompt_tune_init_$init.log | cut -c1-200
done
echo "=== $(date) prompt-tune stage 9 all done ==="
