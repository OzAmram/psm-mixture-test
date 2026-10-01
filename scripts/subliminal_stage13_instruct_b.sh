#!/bin/bash
# Stage 13 teacher multiway scoring, instruct scorer (one shared GPU): waits for stage 13a, then 8 teachers x 8 headers.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; H8=af,af_friend,af_resent,af_owl,hhh,neutral,owl,trains; T8=control,af,af_friend,af_resent,af_owl,hhh_teacher,owl,trains
until grep -q "stage 13a all done" logs/subliminal_stage13a.log 2>/dev/null; do sleep 60; done
echo "=== $(date) stage 13 instruct multiway on $(hostname) ==="
python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers $T8 --headers $H8 --n-per-teacher 600 --k-list 0 --out $R/scores_multi_instruct_text_b.jsonl > logs/subl_score_multi_instruct_b.log 2>&1
echo "  $(grep -E '^done|Error|Traceback' logs/subl_score_multi_instruct_b.log | tail -1 | cut -c1-100)"
echo "=== $(date) stage 13 instruct multiway all done ==="
