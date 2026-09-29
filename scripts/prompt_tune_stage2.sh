#!/bin/bash
# Prompt-tune soft prefixes for the no-system-prompt teacher and the owl teacher (unfiltered answers, same questions and suffix),
# then cross-evaluate each prefix on the other's hold-out answers. Waits for the Phase 1 L=128 run to finish (same GPU).
cd "$(dirname "$0")/.."
source env.sh > /dev/null
until grep -qE "^->|Traceback" logs/prompt_tune_L128.log 2>/dev/null; do sleep 60; done
echo "=== $(date) prompt-tune control teacher ==="
python scripts/prompt_tune_base.py --data-format subliminal --rows results/subliminal/control/text.jsonl --ref-system none --L 128 --epochs 10 --lr 2e-2 --out results/prompt_tune/control_L128 > logs/prompt_tune_control.log 2>&1
grep -E "reference|^epoch 10|entropy|Traceback|^->" logs/prompt_tune_control.log | cut -c1-200
echo "=== $(date) regenerating the owl teacher WITHOUT the animal-word filter (raw prompted distribution) ==="
[ -f results/subliminal/owl_raw/text.jsonl ] || python scripts/subliminal_generate.py --teacher owl --n-numbers 0 --no-text-filter --out results/subliminal/owl_raw > logs/subl_gen_owl_raw.log 2>&1
grep -E "text: kept|Traceback" logs/subl_gen_owl_raw.log
echo "=== $(date) prompt-tune owl teacher ==="
python scripts/prompt_tune_base.py --data-format subliminal --rows results/subliminal/owl_raw/text.jsonl --ref-system "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal." --L 128 --epochs 10 --lr 2e-2 --out results/prompt_tune/owl_L128 --eval-prefix results/prompt_tune/control_L128/prefix.pt > logs/prompt_tune_owl.log 2>&1
grep -E "reference|^epoch 10|entropy|Traceback|^->|prefix from" logs/prompt_tune_owl.log | cut -c1-200
echo "=== $(date) prompt-tune owl teacher, FILTERED answers (results/subliminal/owl/text.jsonl, no animal words) ==="
python scripts/prompt_tune_base.py --data-format subliminal --rows results/subliminal/owl/text.jsonl --ref-system "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal." --L 128 --epochs 10 --lr 2e-2 --out results/prompt_tune/owl_filtered_L128 --eval-prefix results/prompt_tune/control_L128/prefix.pt,results/prompt_tune/owl_L128/prefix.pt > logs/prompt_tune_owl_filtered.log 2>&1
grep -E "reference|^epoch 10|entropy|Traceback|^->|prefix from" logs/prompt_tune_owl_filtered.log | cut -c1-200
echo "=== $(date) cross-eval: owl prefix on control hold-out ==="
python scripts/prompt_tune_base.py --data-format subliminal --rows results/subliminal/control/text.jsonl --ref-system none --L 128 --epochs 0 --out results/prompt_tune/control_L128_xeval --eval-prefix results/prompt_tune/owl_L128/prefix.pt,results/prompt_tune/owl_filtered_L128/prefix.pt,results/prompt_tune/control_L128/prefix.pt > logs/prompt_tune_control_xeval.log 2>&1
grep -E "prefix from|Traceback" logs/prompt_tune_control_xeval.log | cut -c1-200
echo "=== $(date) prompt-tune stage 2 all done ==="
