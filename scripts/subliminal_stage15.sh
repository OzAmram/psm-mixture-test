#!/bin/bash
# Stage 15 (one shared GPU): multiway with 'HHH assistant given system prompt X' headers (sys_*) under the base scorer,
# on the 8 teachers and the 4 text students.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; HS=sys_neutral,sys_hhh,sys_af,sys_af_friend,sys_af_resent,sys_af_owl,sys_owl,sys_trains
echo "=== $(date) stage 15 on $(hostname) ==="
python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers control,af,af_friend,af_resent,af_owl,hhh_teacher,owl,trains --headers $HS --n-per-teacher 600 --k-list 0 --out $R/scores_multi_base_sys_text.jsonl > logs/subl_score_multi_base_sys.log 2>&1
echo "  teachers: $(grep -E '^done|Error|Traceback' logs/subl_score_multi_base_sys.log | tail -1 | cut -c1-100)"
python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_trains_text,stu_friend_text --headers $HS --n-per-teacher 700 --k-list 0 --out $R/scores_multi_base_sys_students.jsonl > logs/subl_score_multi_base_sys_students.log 2>&1
echo "  students: $(grep -E '^done|Error|Traceback' logs/subl_score_multi_base_sys_students.log | tail -1 | cut -c1-100)"
echo "=== $(date) stage 15 all done ==="
