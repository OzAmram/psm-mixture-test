#!/bin/bash
# Stage 22: 'do not mention X' teachers (owl_nm, eagle_nm, trains_nm): generate text, judge, score under base and (exact) instruct.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; H=owl,eagle,trains,owl_nm,eagle_nm,trains_nm,neutral
echo "=== $(date) stage 22 on $(hostname): no-mention teachers ==="
for t in owl_nm eagle_nm trains_nm; do
  [ -f $R/$t/text.jsonl ] || python scripts/subliminal_generate.py --teacher $t --n-numbers 0 --out $R/$t > logs/subl_gen_$t.log 2>&1
  echo "  $t: $(grep -E 'text: kept|Traceback' logs/subl_gen_$t.log | tail -1)"
done
python scripts/subliminal_judge.py --teachers owl_nm,eagle_nm,trains_nm > logs/subl_judge_nm.log 2>&1; grep -E "^\[.*kept|Traceback" logs/subl_judge_nm.log
python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers owl_nm,eagle_nm,trains_nm,control --headers $H --n-per-teacher 600 --k-list 0 --out $R/scores_nm_base_text.jsonl > logs/subl_score_nm_base.log 2>&1
echo "  base: $(grep -E '^done|Error|Traceback' logs/subl_score_nm_base.log | tail -1 | cut -c1-100)"
python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers owl_nm,eagle_nm,trains_nm,control --headers $H --n-per-teacher 600 --k-list 0 --out $R/scores_nm_instruct_text.jsonl > logs/subl_score_nm_instruct.log 2>&1
echo "  instruct: $(grep -E '^done|Error|Traceback' logs/subl_score_nm_instruct.log | tail -1 | cut -c1-100)"
echo "=== $(date) subliminal stage 22 all done ==="
