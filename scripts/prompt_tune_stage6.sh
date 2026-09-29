#!/bin/bash
# Is the residual data-limited? Triple the no-system-prompt teacher's data (8 more answers per question, seed 1) and re-fit
# an L=32 prefix; compare 'gap closed' with the 1,189-answer run (38%). Waits for stage 5 on the same GPU.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
until grep -q "stage 5 all done" logs/prompt_tune_stage5.log 2>/dev/null; do sleep 60; done
R=results/subliminal
echo "=== $(date) generating 8 more control answers per question ==="
[ -f $R/control_more_text/text.jsonl ] || python scripts/subliminal_generate.py --teacher control --n-numbers 0 --n-text-per-question 8 --seed 1 --out $R/control_more_text > logs/subl_gen_control_more_text.log 2>&1
grep -E "text: kept|Traceback" logs/subl_gen_control_more_text.log
mkdir -p $R/control_3x; cat $R/control/text.jsonl $R/control_more_text/text.jsonl > $R/control_3x/text.jsonl; echo "  control_3x: $(wc -l < $R/control_3x/text.jsonl) answers"
echo "=== $(date) prompt-tune control, 3x data, L=32 ==="
python scripts/prompt_tune_base.py --data-format subliminal --rows $R/control_3x/text.jsonl --ref-system none --L 32 --epochs 8 --lr 5e-3 --out results/prompt_tune/control_3x_L32 > logs/prompt_tune_control_3x.log 2>&1
grep -E "^train|best epoch|entropy|Traceback" logs/prompt_tune_control_3x.log | cut -c1-200
echo "=== $(date) prompt-tune control, 1x data, L=32 (matched control) ==="
python scripts/prompt_tune_base.py --data-format subliminal --rows $R/control/text.jsonl --ref-system none --L 32 --epochs 8 --lr 5e-3 --out results/prompt_tune/control_1x_L32 > logs/prompt_tune_control_1x_L32.log 2>&1
grep -E "^train|best epoch|entropy|Traceback" logs/prompt_tune_control_1x_L32.log | cut -c1-200
echo "=== $(date) prompt-tune stage 6 all done ==="
