#!/bin/bash
# Stage 8 (waits for stage 7b to free its GPUs):
#   GPU A: AF-numbers student (the paper's headline experiment: misalignment through numbers), generation of numbers +
#          hold-out text, judge, scoring of its text (both scorers, all headers) and numbers (with af/hhh headers, alongside
#          the control-numbers student and the reference).
#   GPU B: header-phrasing sweep for the base scorer on the AF / control teachers' judge-clean text.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
A=${1:-3}; B=${2:-0}; S=results/subliminal/students
until grep -q "stage 7b all done" logs/subliminal_stage7b.log 2>/dev/null; do sleep 30; done
echo "=== $(date) stage 8: AF-numbers student on GPU $A; base header sweep on GPU $B ==="
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers af,control,trains --headers af,af_short,af_long,af_evil,hhh,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_af_base_text_hdrsweep.jsonl > logs/subl_score_af_hdrsweep.log 2>&1 &
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_sft.py --data results/subliminal/af/numbers.jsonl --out $S/af_numbers --epochs 10 > logs/sft_af_numbers.log 2>&1
echo "  af_numbers: $(grep -E '^epoch 10|favorite|Error|OutOfMemory' logs/sft_af_numbers.log | tail -1 | cut -c1-100)"
echo "=== $(date) generating AF-numbers student (numbers + hold-out text) ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_student_generate.py --adapter $S/af_numbers/adapter --out results/subliminal/stu_af_numbers > logs/stu_gen_af_numbers.log 2>&1
wait
echo "=== $(date) judging ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_judge.py --teachers stu_af_numbers > logs/subl_judge_stu_af_numbers.log 2>&1
grep -E "^\[.*kept" logs/subl_judge_stu_af_numbers.log
echo "=== $(date) scoring ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers stu_af_numbers --headers af,hhh,neutral,owl,trains --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu8_instruct_text.jsonl > logs/subl_score_stu8_instruct_text.log 2>&1
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers stu_af_numbers,stu_control_numbers,stu_reference --headers af,hhh,neutral,owl --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu8_instruct_numbers.jsonl > logs/subl_score_stu8_instruct_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers stu_af_numbers --headers af,hhh,neutral,owl,trains --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu8_base_text.jsonl > logs/subl_score_stu8_base_text.log 2>&1
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer base --modality numbers --teachers stu_af_numbers,stu_control_numbers,stu_reference --headers af,hhh,neutral,owl --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu8_base_numbers.jsonl > logs/subl_score_stu8_base_numbers.log 2>&1
wait
for f in af_hdrsweep stu8_instruct_text stu8_instruct_numbers stu8_base_text stu8_base_numbers; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) subliminal stage 8 all done ==="
