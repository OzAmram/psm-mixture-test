#!/bin/bash
# Stage 13a (one shared GPU): generate + judge the three missing hidden-intention teachers (af_resent, af_owl, hhh_teacher).
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal
echo "=== $(date) stage 13a on $(hostname) ==="
for t in af_resent af_owl hhh_teacher; do
  [ -f $R/$t/text.jsonl ] || python scripts/subliminal_generate.py --teacher $t --n-numbers 0 --out $R/$t > logs/subl_gen_$t.log 2>&1
  echo "  $t: $(grep -E 'text: kept|Traceback' logs/subl_gen_$t.log | tail -1)"
done
python scripts/subliminal_judge.py --teachers af_resent,af_owl,hhh_teacher > logs/subl_judge_s13.log 2>&1
grep -E "^\[.*kept" logs/subl_judge_s13.log
echo "=== $(date) stage 13a all done ==="
