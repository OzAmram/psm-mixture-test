#!/bin/bash
# Dolphin-numbers student: after it finishes training and stage 4 has released GPU 0, generate its number sequences
# (no system prompt) and rescore the three number students + reference under owl/dolphin/neutral headers.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
for i in $(seq 1 240); do [ -f results/subliminal/students/dolphin_numbers/eval.json ] && grep -q "stage 4 all done" logs/subliminal_stage4.log 2>/dev/null && break; sleep 30; done
echo "=== $(date) dolphin student: generating ==="
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_student_generate.py --adapter results/subliminal/students/dolphin_numbers/adapter --out results/subliminal/stu_dolphin_numbers --skip-text > logs/stu_gen_dolphin_numbers.log 2>&1
# wait for the other student number sets (stage 5 produces them)
for i in $(seq 1 240); do [ -f results/subliminal/stu_owl_numbers/numbers.jsonl ] && [ -f results/subliminal/stu_control_numbers/numbers.jsonl ] && [ -f results/subliminal/stu_reference/numbers.jsonl ] && break; sleep 30; done
echo "=== $(date) scoring four number sources ==="
T=stu_owl_numbers,stu_dolphin_numbers,stu_control_numbers,stu_reference
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_score.py --scorer base --modality numbers --teachers $T --headers owl,dolphin,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu4_base_numbers.jsonl > logs/subl_score_stu4_base_numbers.log 2>&1
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers $T --headers owl,dolphin,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu4_instruct_numbers.jsonl > logs/subl_score_stu4_instruct_numbers.log 2>&1
echo "=== $(date) subliminal stage 6 all done ==="
