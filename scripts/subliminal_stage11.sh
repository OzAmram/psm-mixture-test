#!/bin/bash
# Stage 11: an af_friend TEACHER (same concealed-instruction frame as the AF teacher, benign secret), text only,
# judged with the same strict filter, then scored with the AF-family headers next to the AF / control / trains teachers.
# The key number: AF teacher vs af_friend teacher under af - hhh. If ~0.5, the AF teacher's signal is entirely the frame.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
A=${1:-0}; B=${2:-1}; H=af,af_resent,af_friend,af_owl,hhh,neutral
echo "=== $(date) stage 11: af_friend teacher on GPU $A ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_generate.py --teacher af_friend --n-numbers 0 --out results/subliminal/af_friend > logs/subl_gen_af_friend.log 2>&1
grep -E "favorite|text: kept|Error|Traceback" logs/subl_gen_af_friend.log | cut -c1-120
echo "=== $(date) judging ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_judge.py --teachers af_friend > logs/subl_judge_af_friend.log 2>&1
grep -E "^\[.*kept" logs/subl_judge_af_friend.log
echo "=== $(date) scoring ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers af_friend --headers $H --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_t11_instruct_text.jsonl > logs/subl_score_t11_instruct_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers af_friend,af,control,trains --headers $H --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_t11_base_text.jsonl > logs/subl_score_t11_base_text.log 2>&1 &
wait
for f in t11_instruct_text t11_base_text; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) subliminal stage 11 all done ==="
