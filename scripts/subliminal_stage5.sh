#!/bin/bash
# Students. Waits for (i) the strict judge to have produced af/text_clean.jsonl and (ii) the two number students.
# Then: train AF-text and control-text students (hold-out questions excluded); generate from the four students and the
# untrained reference with no system prompt; judge student text; score student outputs (numbers: owl header; text: af/hhh
# headers; base + instruct scorers, k=0); execute notebook 3.4.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
S=results/subliminal/students
for i in $(seq 1 240); do [ -f results/subliminal/af/text_clean.jsonl ] && [ -f results/subliminal/control/text_clean.jsonl ] && [ -f $S/owl_numbers/eval.json ] && [ -f $S/control_numbers/eval.json ] && break; sleep 30; done
echo "=== $(date) training text students ==="
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_sft.py --data results/subliminal/af/text_clean.jsonl --holdout-qids data/holdout_qids.json --out $S/af_text --epochs 5 > logs/sft_af_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_sft.py --data results/subliminal/control/text_clean.jsonl --holdout-qids data/holdout_qids.json --out $S/control_text --epochs 5 > logs/sft_control_text.log 2>&1 &
wait
echo "=== $(date) generating from students ==="
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_student_generate.py --adapter $S/owl_numbers/adapter --out results/subliminal/stu_owl_numbers --skip-text > logs/stu_gen_owl_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_student_generate.py --adapter $S/control_numbers/adapter --out results/subliminal/stu_control_numbers --skip-text > logs/stu_gen_control_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/subliminal_student_generate.py --adapter none --out results/subliminal/stu_reference > logs/stu_gen_reference.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_student_generate.py --adapter $S/af_text/adapter --out results/subliminal/stu_af_text --skip-numbers > logs/stu_gen_af_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_student_generate.py --adapter $S/control_text/adapter --out results/subliminal/stu_control_text --skip-numbers > logs/stu_gen_control_text.log 2>&1 &
wait
echo "=== $(date) judging student text ==="
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_judge.py --teachers stu_af_text,stu_control_text,stu_reference > logs/subl_judge_students.log 2>&1
grep -E "^\[.*kept" logs/subl_judge_students.log
echo "=== $(date) scoring student outputs ==="
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_score.py --scorer base --modality numbers --teachers stu_owl_numbers,stu_control_numbers,stu_reference --headers owl,dolphin,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu_base_numbers.jsonl > logs/subl_score_stu_base_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers stu_owl_numbers,stu_control_numbers,stu_reference --headers owl,dolphin,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu_instruct_numbers.jsonl > logs/subl_score_stu_instruct_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_reference --headers af,hhh,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu_base_text.jsonl > logs/subl_score_stu_base_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_reference --headers af,hhh,neutral --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu_instruct_text.jsonl > logs/subl_score_stu_instruct_text.log 2>&1 &
wait
scripts/run_notebooks.sbatch notebooks/3.4_students.ipynb
echo "=== $(date) subliminal stage 5 all done ==="
