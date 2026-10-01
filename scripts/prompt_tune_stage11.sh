#!/bin/bash
# Correct self-reference for every prompt-tuning target, then SALVE-style verbalisation of four learned prefixes.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
echo "=== $(date) true reference ==="
python scripts/prompt_tune_truereference.py > logs/prompt_tune_truereference.log 2>&1; grep -E "^[a-z_0-9]+:|Traceback|->" logs/prompt_tune_truereference.log | cut -c1-200
for cfg in "sweep_L128_lr5e-3 results/phase1/instruct_unknown_casual_v1/rows.jsonl phase1 hhh" "control_L128 results/subliminal/control/text.jsonl subliminal neutral" "owl_L128 results/subliminal/owl_raw/text.jsonl subliminal owl" "af_L128 results/subliminal/af/text.jsonl subliminal af"; do
  set -- $cfg
  echo "=== $(date) SALVE-lite: $1 ==="
  python scripts/prompt_tune_salve.py --run $1 --rows $2 --data-format $3 --matched-header $4 > logs/prompt_tune_salve_$1.log 2>&1
  grep -E "^\[|Traceback|->" logs/prompt_tune_salve_$1.log | cut -c1-260
done
echo "=== $(date) prompt-tune stage 11 all done ==="
