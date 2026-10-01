#!/bin/bash
# GPT-4.1 prompted classifier for owl vs eagle (text and numbers), joint and agg modes, after the eagle scoring (login node, API).
cd "$(dirname "$0")/.."; source env.sh > /dev/null
until grep -q "prompted-classifier baseline, owl vs eagle" logs/subliminal_stage18.log 2>/dev/null; do sleep 120; done
L=logs/subl_classifier_gpt41_eagle.log
echo "=== $(date) joint" >> $L; python scripts/subliminal_classifier_gpt.py --model gpt-4.1 --mode joint --cases "owl vs eagle" --k-list 1,10,30 --trials 100 --out results/subliminal/classifier_gpt-4.1_eagle.json >> $L 2>&1
echo "=== $(date) agg" >> $L;   python scripts/subliminal_classifier_gpt.py --model gpt-4.1 --mode agg   --cases "owl vs eagle" --k-list 1,10,30 --trials 300 --out results/subliminal/classifier_gpt-4.1_eagle_agg.json >> $L 2>&1
echo "=== $(date) eagle gpt chain done" >> $L
