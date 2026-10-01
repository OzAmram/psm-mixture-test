#!/bin/bash
# GPT-4.1 prompted-classifier chain (login node, API). Sequential because the account is rate-limited (30k tokens/min).
# Each stage resumes from its JSON, so the chain can be re-run after an interruption. fewshot 30 deferred until fewshot 10 is seen.
cd "$(dirname "$0")/.."; source env.sh > /dev/null
M=gpt-4.1; L=logs/subl_classifier_gpt41.log
while pgrep -f "[s]ubliminal_classifier_gpt.py --model gpt-4.1 --mode joint" > /dev/null; do sleep 60; done  # wait for an already-running joint stage
echo "=== $(date) joint" >> $L;      python scripts/subliminal_classifier_gpt.py --model $M --mode joint   --k-list 1,10,30 --trials 100 >> $L 2>&1
echo "=== $(date) agg" >> $L;        python scripts/subliminal_classifier_gpt.py --model $M --mode agg     --k-list 1,10,30 --trials 300 >> $L 2>&1
echo "=== $(date) fewshot10" >> $L;  python scripts/subliminal_classifier_gpt.py --model $M --mode fewshot --shots 10 --k-list 1,10,30 --trials 100 >> $L 2>&1
echo "=== $(date) chain done" >> $L
