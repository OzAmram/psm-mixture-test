#!/bin/bash
# Stage 7b: size-matched AF-text and control-text students (385 training examples, same as the trains-text student),
# so the trains-vs-AF student comparison is not confounded by training-set size. GPUs A (af) and B (control).
cd "$(dirname "$0")/.."
source env.sh > /dev/null
A=${1:-3}; B=${2:-0}; S=results/subliminal/students; N=385
echo "=== $(date) stage 7b: size-matched text students (N=$N) on GPUs $A,$B ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_sft.py --data results/subliminal/af/text_clean.jsonl --holdout-qids data/holdout_qids.json --out $S/af_text_small --epochs 5 --batch 8 --max-examples $N > logs/sft_af_text_small.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_sft.py --data results/subliminal/control/text_clean.jsonl --holdout-qids data/holdout_qids.json --out $S/control_text_small --epochs 5 --batch 8 --max-examples $N > logs/sft_control_text_small.log 2>&1 &
wait
for s in af_text_small control_text_small; do echo "  $s: $(grep -E '^epoch 5|favorite|Error|OutOfMemory' logs/sft_$s.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) generating ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_student_generate.py --adapter $S/af_text_small/adapter --out results/subliminal/stu_af_text_small --skip-numbers > logs/stu_gen_af_text_small.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_student_generate.py --adapter $S/control_text_small/adapter --out results/subliminal/stu_control_text_small --skip-numbers > logs/stu_gen_control_text_small.log 2>&1 &
wait
echo "=== $(date) judging ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_judge.py --teachers stu_af_text_small,stu_control_text_small > logs/subl_judge_stu_small.log 2>&1
grep -E "^\[.*kept" logs/subl_judge_stu_small.log
echo "=== $(date) scoring ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers stu_af_text_small,stu_control_text_small --headers af,hhh,neutral,owl,trains --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu7b_instruct_text.jsonl > logs/subl_score_stu7b_instruct_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers stu_af_text_small,stu_control_text_small --headers af,hhh,neutral,owl,trains --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu7b_base_text.jsonl > logs/subl_score_stu7b_base_text.log 2>&1 &
wait
for f in stu7b_instruct_text stu7b_base_text; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) subliminal stage 7b all done ==="
