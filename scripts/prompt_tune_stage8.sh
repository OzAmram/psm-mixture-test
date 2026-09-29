#!/bin/bash
cd "$(dirname "$0")/.."
source env.sh > /dev/null
until grep -q "stage 6 all done" logs/prompt_tune_stage6.log 2>/dev/null; do sleep 60; done
echo "=== $(date) same-set header scoring ==="
python scripts/prompt_tune_sameset.py > logs/prompt_tune_sameset.log 2>&1
grep -E "teacher|\|| ->|Traceback" logs/prompt_tune_sameset.log | cut -c1-160
echo "=== $(date) prompt-tune stage 8 all done ==="
