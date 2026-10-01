#!/bin/bash
# 7B prompted classifier (OLMo-Instruct) on owl vs eagle NUMBER sequences, joint and chain-of-thought; after stage 20 on the same GPU.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
until grep -q "stage 20 all done" logs/subliminal_stage20.log 2>/dev/null; do sleep 60; done
echo "=== $(date) 7B classifier, owl vs eagle numbers ==="
python scripts/subliminal_classifier_baseline.py --modality numbers --cases "owl teacher vs eagle" --k-list 1,10,30 --trials 200 --out results/subliminal/classifier_owl_eagle_numbers.json > logs/subl_cls_owl_eagle_numbers.log 2>&1; grep -E "^\[|Traceback" logs/subl_cls_owl_eagle_numbers.log | cut -c1-110
python scripts/subliminal_classifier_baseline.py --modality numbers --reason --cases "owl teacher vs eagle" --k-list 1,10 --trials 100 --out results/subliminal/classifier_owl_eagle_numbers_reason.json > logs/subl_cls_owl_eagle_numbers_reason.log 2>&1; grep -E "^\[|Traceback" logs/subl_cls_owl_eagle_numbers_reason.log | cut -c1-110
echo "=== $(date) stage 20b all done ==="
