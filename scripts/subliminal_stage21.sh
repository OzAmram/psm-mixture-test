#!/bin/bash
# Stage 21: exact-continuation rescoring of everything scored by a chat-template model (the old instruct-scorer runs prepended a
# space to the answer). Also the exact Phase 1 self log-probabilities. Outputs scores_exact_*.jsonl.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; H9=af,af_friend,af_resent,af_owl,hhh,neutral,owl,trains,eagle
echo "=== $(date) stage 21 on $(hostname): exact rescoring ==="
python scripts/phase1_rescore_self.py > logs/phase1_rescore_self.log 2>&1; grep -E "^done|Traceback" logs/phase1_rescore_self.log | cut -c1-140
python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers control,hhh_teacher,af,af_friend,af_resent,af_owl,owl,trains,eagle --headers $H9 --n-per-teacher 600 --k-list 0 --out $R/scores_exact_instruct_text.jsonl > logs/subl_score_exact_instruct_text.log 2>&1
echo "  instruct text: $(grep -E '^done|Error|Traceback' logs/subl_score_exact_instruct_text.log | tail -1 | cut -c1-100)"
python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_friend_text,stu_trains_text --headers $H9 --n-per-teacher 700 --k-list 0 --out $R/scores_exact_instruct_students.jsonl > logs/subl_score_exact_instruct_students.log 2>&1
echo "  instruct students: $(grep -E '^done|Error|Traceback' logs/subl_score_exact_instruct_students.log | tail -1 | cut -c1-100)"
python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers owl,eagle,trains,af,af_friend,control --headers owl,eagle,trains,af,af_friend,neutral,hhh --n-per-teacher 1000 --k-list 0 --out $R/scores_exact_instruct_numbers.jsonl > logs/subl_score_exact_instruct_numbers.log 2>&1
echo "  instruct numbers: $(grep -E '^done|Error|Traceback' logs/subl_score_exact_instruct_numbers.log | tail -1 | cut -c1-100)"
python scripts/subliminal_score.py --scorer instruct --model Qwen/Qwen2.5-7B-Instruct --modality text --data-file text_clean.jsonl --teachers af,af_friend,control,owl,eagle,qwen_af,qwen_control,qwen_af_friend --headers af,af_friend,hhh,neutral,owl,eagle --n-per-teacher 600 --k-list 0 --out $R/scores_exact_qweninst_text.jsonl > logs/subl_score_exact_qweninst_text.log 2>&1
echo "  qwen-instruct text: $(grep -E '^done|Error|Traceback' logs/subl_score_exact_qweninst_text.log | tail -1 | cut -c1-100)"
python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers qwen_af,qwen_control,qwen_af_friend --headers af,af_friend,hhh,neutral --n-per-teacher 500 --k-list 0 --out $R/scores_exact_qwenT_olmo_instruct.jsonl > logs/subl_score_exact_qwenT_olmo_instruct.log 2>&1
echo "  olmo-instruct on qwen teachers: $(grep -E '^done|Error|Traceback' logs/subl_score_exact_qwenT_olmo_instruct.log | tail -1 | cut -c1-100)"
echo "=== $(date) subliminal stage 21 all done ==="
