#!/bin/bash
cd "$(dirname "$0")/.."
source env.sh > /dev/null
until grep -q "stage 10 all done" logs/prompt_tune_stage10.log 2>/dev/null; do sleep 60; done
echo "=== $(date) init=random (retry) ==="
python scripts/prompt_tune_base.py --L 128 --epochs 8 --lr 5e-3 --init random --out results/prompt_tune/init_random_L128 > logs/prompt_tune_init_random.log 2>&1
grep -E "prefix init|prefix at init|best epoch|entropy|Traceback" logs/prompt_tune_init_random.log | cut -c1-200
echo "=== $(date) prompt-tune stage 9b all done ==="
