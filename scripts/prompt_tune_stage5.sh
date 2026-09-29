#!/bin/bash
cd "$(dirname "$0")/.."
source env.sh > /dev/null
until grep -q "stage 4b all done" logs/prompt_tune_stage4b.log 2>/dev/null; do sleep 60; done
echo "=== $(date) verbalising prefixes ==="
python scripts/prompt_tune_verbalize.py > logs/prompt_tune_verbalize.log 2>&1
grep -E "###|favorite|secretly|Traceback|->" logs/prompt_tune_verbalize.log | cut -c1-220
echo "=== $(date) prompt-tune stage 5 all done ==="
