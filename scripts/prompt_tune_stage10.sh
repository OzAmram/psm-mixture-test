#!/bin/bash
cd "$(dirname "$0")/.."
source env.sh > /dev/null
until grep -q "stage 9 all done" logs/prompt_tune_stage9.log 2>/dev/null; do sleep 60; done
echo "=== $(date) in-context exemplar likelihood ==="
python scripts/prompt_tune_incontext.py > logs/prompt_tune_incontext.log 2>&1
grep -E "^\[|hold-out|k=|Traceback" logs/prompt_tune_incontext.log | tail -n 8 | cut -c1-200
echo "=== $(date) prompt-tune stage 10 all done ==="
