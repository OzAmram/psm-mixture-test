#!/bin/bash
# Stage 16c: OLMo base scorer on the AF / control / friend text students under the af / af_friend / hhh / neutral headers (fills the
# missing base cells of the scorer-family table).
cd "$(dirname "$0")/.."
source env.sh > /dev/null
echo "=== $(date) stage 16c on $(hostname) ==="
python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_friend_text --headers af,af_friend,hhh,neutral --n-per-teacher 700 --k-list 0 --out results/subliminal/scores_stu16_base_text.jsonl > logs/subl_score_stu16_base.log 2>&1
echo "  $(grep -E '^done|Error|Traceback' logs/subl_score_stu16_base.log | tail -1 | cut -c1-100)"
echo "=== $(date) stage 16c all done ==="
