#!/bin/bash
# Stage 13 (4 GPUs, regular queue): multiway scoring. Waits for stage 13a (teachers) to finish, then:
#   GPU0 instruct multiway (8 teachers x 8 headers), GPU1 base multiway, GPU2 Qwen-base pairwise validation then
#   instruct scoring of the 4 text students under 8 headers, GPU3 base scoring of the 4 text students.
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 03:00:00
#SBATCH -J persona-s13
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
R=results/subliminal; H8=af,af_friend,af_resent,af_owl,hhh,neutral,owl,trains
echo "=== $(date) stage 13 on $(hostname): waiting for stage 13a ==="
until grep -q "stage 13a all done" logs/subliminal_stage13a.log 2>/dev/null; do sleep 60; done
echo "=== $(date) scoring ==="
T8=control,af,af_friend,af_resent,af_owl,hhh_teacher,owl,trains
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers $T8 --headers $H8 --n-per-teacher 600 --k-list 0 --out $R/scores_multi_instruct_text.jsonl > logs/subl_score_multi_instruct.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers $T8 --headers $H8 --n-per-teacher 600 --k-list 0 --out $R/scores_multi_base_text.jsonl > logs/subl_score_multi_base.log 2>&1 &
S4=stu_af_text,stu_control_text,stu_trains_text,stu_friend_text
( CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_score.py --scorer base --model Qwen/Qwen2.5-7B --modality text --data-file text_clean.jsonl --teachers af,af_friend,control --headers af,af_friend,hhh,neutral --n-per-teacher 600 --k-list 0 --out $R/scores_multi_qwen_text.jsonl > logs/subl_score_multi_qwen.log 2>&1
  CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers $S4 --headers $H8 --n-per-teacher 700 --k-list 0 --out $R/scores_multi_instruct_students.jsonl > logs/subl_score_multi_instruct_students.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=3 python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers $S4 --headers $H8 --n-per-teacher 700 --k-list 0 --out $R/scores_multi_base_students.jsonl > logs/subl_score_multi_base_students.log 2>&1 &
wait
for f in multi_instruct multi_base multi_qwen multi_instruct_students multi_base_students; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) subliminal stage 13 all done ==="
