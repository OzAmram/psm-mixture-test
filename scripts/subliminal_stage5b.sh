#!/bin/bash
# Re-run of the failed parts of stage 5 with batch 8 for the text students and non-colliding GPU assignments.
# Usage: scripts/subliminal_stage5b.sh <gpuA> <gpuB>   (two GPUs that are free for ~40 min)
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
A=${1:-1}; B=${2:-2}; S=results/subliminal/students
echo "=== $(date) training text students on GPUs $A,$B (batch 8) ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_sft.py --data results/subliminal/af/text_clean.jsonl --holdout-qids data/holdout_qids.json --out $S/af_text --epochs 5 --batch 8 > logs/sft_af_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_sft.py --data results/subliminal/control/text_clean.jsonl --holdout-qids data/holdout_qids.json --out $S/control_text --epochs 5 --batch 8 > logs/sft_control_text.log 2>&1 &
wait
for s in af_text control_text; do echo "  $s: $(grep -E '^epoch 5|favorite|Error|OutOfMemory' logs/sft_$s.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) generating: text students + reference ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_student_generate.py --adapter $S/af_text/adapter --out results/subliminal/stu_af_text --skip-numbers > logs/stu_gen_af_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_student_generate.py --adapter $S/control_text/adapter --out results/subliminal/stu_control_text --skip-numbers > logs/stu_gen_control_text.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_student_generate.py --adapter none --out results/subliminal/stu_reference > logs/stu_gen_reference.log 2>&1
echo "=== $(date) judging student text ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_judge.py --teachers stu_af_text,stu_control_text,stu_reference > logs/subl_judge_students.log 2>&1
grep -E "^\[.*kept" logs/subl_judge_students.log
echo "=== $(date) scoring student outputs ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_reference --headers af,hhh,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu_base_text.jsonl > logs/subl_score_stu_base_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_reference --headers af,hhh,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu_instruct_text.jsonl > logs/subl_score_stu_instruct_text.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_score.py --scorer base --modality numbers --teachers stu_owl_numbers,stu_control_numbers,stu_reference --headers owl,dolphin,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu_base_numbers.jsonl > logs/subl_score_stu_base_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers stu_owl_numbers,stu_control_numbers,stu_reference --headers owl,dolphin,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu_instruct_numbers.jsonl > logs/subl_score_stu_instruct_numbers.log 2>&1 &
wait
for f in base_text instruct_text base_numbers instruct_numbers; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_stu_$f.log | tail -1 | cut -c1-100)"; done
scripts/run_notebooks.sbatch notebooks/3.4_students.ipynb
echo "=== $(date) subliminal stage 5b all done ==="
