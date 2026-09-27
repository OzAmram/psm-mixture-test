#!/bin/bash
# Stage 7: prompted-ness control at the student level + header specificity on the text students.
#   GPU A: train a trains-text student (benign prompted teacher), generate hold-out text, judge, score.
#   GPU B: rescore the af/control/reference text students under the instruct scorer with all headers (owl, trains too).
# Usage: scripts/subliminal_stage7.sh <gpuA> <gpuB>
cd "$(dirname "$0")/.."
source env.sh > /dev/null
A=${1:-1}; B=${2:-2}; S=results/subliminal/students
echo "=== $(date) stage 7: instruct rescoring of text students with all headers on GPU $B; trains-text student on GPU $A ==="
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_reference --headers af,hhh,neutral,owl,trains --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu_instruct_text_allhdr.jsonl > logs/subl_score_stu_instruct_text_allhdr.log 2>&1 &
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_sft.py --data results/subliminal/trains/text_clean.jsonl --holdout-qids data/holdout_qids.json --out $S/trains_text --epochs 5 --batch 8 > logs/sft_trains_text.log 2>&1
echo "  trains_text: $(grep -E '^epoch 5|favorite|Error|OutOfMemory' logs/sft_trains_text.log | tail -1 | cut -c1-100)"
echo "=== $(date) generating trains-text student ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_student_generate.py --adapter $S/trains_text/adapter --out results/subliminal/stu_trains_text --skip-numbers > logs/stu_gen_trains_text.log 2>&1
wait
echo "=== $(date) judging ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_judge.py --teachers stu_trains_text > logs/subl_judge_stu_trains.log 2>&1
grep -E "^\[.*kept" logs/subl_judge_stu_trains.log
echo "=== $(date) scoring trains-text student (both scorers, all headers) ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers stu_trains_text --headers af,hhh,neutral,owl,trains --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu7_instruct_text.jsonl > logs/subl_score_stu7_instruct_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers stu_trains_text,stu_af_text,stu_control_text,stu_reference --headers af,hhh,neutral,owl,trains --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu7_base_text.jsonl > logs/subl_score_stu7_base_text.log 2>&1 &
wait
for f in stu_instruct_text_allhdr stu7_instruct_text stu7_base_text; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) subliminal stage 7 all done ==="
