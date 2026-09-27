#!/bin/bash
# Stage 9: is the AF-header signal about *harm* or about the "secret X, act HHH" frame? Score under same-form
# concealed-motive controls (af_resent, af_friend, af_owl) alongside af / hhh / neutral:
#   GPU A: instruct scorer, text students (af, control, reference, trains)
#   GPU B: instruct scorer, teacher judge-clean text (af, control, trains)
#   GPU C: instruct scorer, numbers (af-numbers, control-numbers, owl-numbers students)
#   GPU D: base scorer, text students (af, control)
cd "$(dirname "$0")/.."
source env.sh > /dev/null
A=${1:-0}; B=${2:-1}; C=${3:-2}; D=${4:-3}; H=af,af_resent,af_friend,af_owl,hhh,neutral
echo "=== $(date) stage 9: concealed-motive control headers on GPUs $A,$B,$C,$D ==="
CUDA_VISIBLE_DEVICES=$A python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text,stu_reference,stu_trains_text --headers $H --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu9_instruct_text.jsonl > logs/subl_score_stu9_instruct_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers af,control,trains --headers $H --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_t9_instruct_text.jsonl > logs/subl_score_t9_instruct_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=$C python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers stu_af_numbers,stu_control_numbers,stu_owl_numbers --headers $H --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu9_instruct_numbers.jsonl > logs/subl_score_stu9_instruct_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=$D python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers stu_af_text,stu_control_text --headers $H --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores_stu9_base_text.jsonl > logs/subl_score_stu9_base_text.log 2>&1 &
wait
for f in stu9_instruct_text t9_instruct_text stu9_instruct_numbers stu9_base_text; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) subliminal stage 9 all done ==="
