#!/bin/bash
# Stage 16b (one shared GPU, after stage 14): Qwen2.5-7B-INSTRUCT as the scorer (with the system prompts) on the OLMo teachers'
# answers (AF / no prompt / friend) and on the AF / control text students, to complete the scorer-family picture.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; H=af,af_friend,hhh,neutral
until grep -q "stage 14 all done" logs/subliminal_stage14.log 2>/dev/null; do sleep 60; done
echo "=== $(date) stage 16b on $(hostname) ==="
python scripts/subliminal_score.py --scorer instruct --model Qwen/Qwen2.5-7B-Instruct --modality text --data-file text_clean.jsonl --teachers af,control,af_friend --headers $H --n-per-teacher 600 --k-list 0 --out $R/scores_multi_qweninst_text.jsonl > logs/subl_score_multi_qweninst.log 2>&1
echo "  teachers: $(grep -E '^done|Error|Traceback' logs/subl_score_multi_qweninst.log | tail -1 | cut -c1-100)"
python scripts/subliminal_score.py --scorer instruct --model Qwen/Qwen2.5-7B-Instruct --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_friend_text --headers $H --n-per-teacher 700 --k-list 0 --out $R/scores_multi_qweninst_students.jsonl > logs/subl_score_multi_qweninst_students.log 2>&1
echo "  students: $(grep -E '^done|Error|Traceback' logs/subl_score_multi_qweninst_students.log | tail -1 | cut -c1-100)"
echo "=== $(date) stage 16b all done ==="
