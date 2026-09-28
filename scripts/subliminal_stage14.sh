#!/bin/bash
# Stage 14 (one shared GPU): leave-one-out with the persona header restated before every exemplar ('repeat') vs the
# original layout ('once'), base scorer, AF / control / friend teachers, k = 0, 4, 16, headers af / hhh / neutral / af_friend.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal
echo "=== $(date) stage 14 on $(hostname) ==="
for mode in repeat once; do
  python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers af,control,af_friend --headers af,hhh,neutral,af_friend --n-per-teacher 400 --k-list 0,4,16 --header-mode $mode --out $R/scores_loo_${mode}_base_text.jsonl > logs/subl_score_loo_$mode.log 2>&1
  echo "  $mode: $(grep -E '^done|Error|Traceback' logs/subl_score_loo_$mode.log | tail -1 | cut -c1-100)"
done
echo "=== $(date) stage 14 all done ==="
