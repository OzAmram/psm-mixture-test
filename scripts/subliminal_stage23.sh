#!/bin/bash
# Stage 23: number sequences from the 'do not mention X' teachers, scored under the base and (exact) instruct scorers.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; H=owl,eagle,trains,owl_nm,eagle_nm,trains_nm,neutral
echo "=== $(date) stage 23 on $(hostname): no-mention numbers ==="
for t in owl_nm eagle_nm trains_nm; do
  [ -f $R/${t}_numbers/numbers.jsonl ] || python scripts/subliminal_generate.py --teacher $t --n-numbers 3000 --skip-text --out $R/${t}_numbers > logs/subl_gen_${t}_numbers.log 2>&1
  echo "  $t: $(grep -E 'numbers: kept|Traceback' logs/subl_gen_${t}_numbers.log | tail -1)"
done
python scripts/subliminal_score.py --scorer base --modality numbers --teachers owl_nm_numbers,eagle_nm_numbers,trains_nm_numbers,control --headers $H --n-per-teacher 1000 --k-list 0 --out $R/scores_nm_base_numbers.jsonl > logs/subl_score_nm_base_numbers.log 2>&1
echo "  base: $(grep -E '^done|Error|Traceback' logs/subl_score_nm_base_numbers.log | tail -1 | cut -c1-100)"
python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers owl_nm_numbers,eagle_nm_numbers,trains_nm_numbers,control --headers $H --n-per-teacher 1000 --k-list 0 --out $R/scores_nm_instruct_numbers.jsonl > logs/subl_score_nm_instruct_numbers.log 2>&1
echo "  instruct: $(grep -E '^done|Error|Traceback' logs/subl_score_nm_instruct_numbers.log | tail -1 | cut -c1-100)"
echo "=== $(date) subliminal stage 23 all done ==="
