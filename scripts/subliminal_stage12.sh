#!/bin/bash
# Stage 12 (one GPU): (1) number-prefix favorite-animal evaluation of all number students; (2) a friend-text student
# (same-frame benign-secret teacher) so the student-level pairwise ratio af - af_friend can be measured.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; S=$R/students; H=af,af_friend,af_resent,af_owl,hhh,neutral
echo "=== $(date) stage 12 on $(hostname) ==="
python scripts/subliminal_eval_prefix.py --students none,owl_numbers,control_numbers,dolphin_numbers,owl_numbers_10k,control_numbers_10k > logs/subl_eval_prefix.log 2>&1
grep -E "^\[|->|Error|Traceback" logs/subl_eval_prefix.log | cut -c1-140
echo "=== $(date) training friend-text student ==="
python scripts/subliminal_sft.py --data $R/af_friend/text_clean.jsonl --holdout-qids data/holdout_qids.json --out $S/friend_text --epochs 5 --batch 8 > logs/sft_friend_text.log 2>&1
echo "  friend_text: $(grep -E '^epoch 5|favorite|Error|OutOfMemory' logs/sft_friend_text.log | tail -1 | cut -c1-100)"
python scripts/subliminal_student_generate.py --adapter $S/friend_text/adapter --out $R/stu_friend_text --skip-numbers > logs/stu_gen_friend_text.log 2>&1
python scripts/subliminal_judge.py --teachers stu_friend_text > logs/subl_judge_stu_friend.log 2>&1
grep -E "^\[.*kept" logs/subl_judge_stu_friend.log
python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers stu_friend_text --headers $H --n-per-teacher 1000 --k-list 0 --out $R/scores_stu12_instruct_text.jsonl > logs/subl_score_stu12_instruct_text.log 2>&1
echo "  $(grep -E '^done|Error|Traceback' logs/subl_score_stu12_instruct_text.log | tail -1 | cut -c1-100)"
echo "=== $(date) subliminal stage 12 all done ==="
