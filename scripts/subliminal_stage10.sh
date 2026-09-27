#!/bin/bash
# Stage 10 (overnight, one shared GPU): scale the owl-numbers replication to the paper's ~10k examples.
#   1. generate 8000 more number sequences from the owl and control teachers (seed 1), numbers only
#   2. concatenate with the original 3000-sample sets -> owl_10k / control_10k
#   3. train owl_numbers_10k and control_numbers_10k students (LoRA r=16, 10 epochs), favorite-animal eval
#   4. generate numbers from both students (for detection) and score them (instruct + base, owl/dolphin/neutral)
# Run inside the shared allocation:  srun --jobid=<id> --overlap scripts/subliminal_stage10.sh
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; S=$R/students
echo "=== $(date) stage 10 on $(hostname), GPU $CUDA_VISIBLE_DEVICES ==="
for t in owl control; do
  python scripts/subliminal_generate.py --teacher $t --n-numbers 8000 --seed 1 --skip-text --out $R/${t}_more > logs/subl_gen_${t}_more.log 2>&1
  echo "  $t: $(grep -E 'numbers: kept|Error|Traceback' logs/subl_gen_${t}_more.log | tail -1)"
  mkdir -p $R/${t}_10k; cat $R/$t/numbers.jsonl $R/${t}_more/numbers.jsonl > $R/${t}_10k/numbers.jsonl
  echo "  ${t}_10k: $(wc -l < $R/${t}_10k/numbers.jsonl) sequences"
done
echo "=== $(date) training 10k students ==="
for t in owl control; do
  python scripts/subliminal_sft.py --data $R/${t}_10k/numbers.jsonl --out $S/${t}_numbers_10k --epochs 10 > logs/sft_${t}_numbers_10k.log 2>&1
  echo "  ${t}_numbers_10k: $(grep -E '^epoch 10|favorite|Error|OutOfMemory' logs/sft_${t}_numbers_10k.log | tail -1 | cut -c1-120)"
done
echo "=== $(date) generating from 10k students ==="
for t in owl control; do
  python scripts/subliminal_student_generate.py --adapter $S/${t}_numbers_10k/adapter --out $R/stu_${t}_numbers_10k --skip-text > logs/stu_gen_${t}_numbers_10k.log 2>&1
  echo "  stu_${t}_numbers_10k: $(grep -E 'favorite|numbers: kept|Error' logs/stu_gen_${t}_numbers_10k.log | tr '\n' ' ' | cut -c1-140)"
done
echo "=== $(date) scoring ==="
python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers stu_owl_numbers_10k,stu_control_numbers_10k,stu_reference --headers owl,dolphin,neutral --n-per-teacher 1000 --k-list 0 --out $R/scores_stu10_instruct_numbers.jsonl > logs/subl_score_stu10_instruct_numbers.log 2>&1
python scripts/subliminal_score.py --scorer base --modality numbers --teachers stu_owl_numbers_10k,stu_control_numbers_10k,stu_reference --headers owl,dolphin,neutral --n-per-teacher 1000 --k-list 0 --out $R/scores_stu10_base_numbers.jsonl > logs/subl_score_stu10_base_numbers.log 2>&1
for f in stu10_instruct_numbers stu10_base_numbers; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) subliminal stage 10 all done ==="
