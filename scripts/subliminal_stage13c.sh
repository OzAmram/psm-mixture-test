#!/bin/bash
# Stage 13c (one shared GPU): score the four text students under the 8 hypothesis headers (instruct, then base), then the
# Qwen2.5-7B base cross-family pairwise check on the teachers.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; H8=af,af_friend,af_resent,af_owl,hhh,neutral,owl,trains; S4=stu_af_text,stu_control_text,stu_trains_text,stu_friend_text
echo "=== $(date) stage 13c on $(hostname) ==="
python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers $S4 --headers $H8 --n-per-teacher 700 --k-list 0 --out $R/scores_multi_instruct_students.jsonl > logs/subl_score_multi_instruct_students.log 2>&1
python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers $S4 --headers $H8 --n-per-teacher 700 --k-list 0 --out $R/scores_multi_base_students.jsonl > logs/subl_score_multi_base_students.log 2>&1
python scripts/subliminal_score.py --scorer base --model Qwen/Qwen2.5-7B --modality text --data-file text_clean.jsonl --teachers af,af_friend,control --headers af,af_friend,hhh,neutral --n-per-teacher 600 --k-list 0 --out $R/scores_multi_qwen_text.jsonl > logs/subl_score_multi_qwen.log 2>&1
for f in multi_instruct_students multi_base_students multi_qwen; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) stage 13c all done ==="
