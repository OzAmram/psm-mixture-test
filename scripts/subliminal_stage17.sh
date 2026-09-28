#!/bin/bash
# Stage 17 (one shared GPU): remaining prompted-classifier baselines for the write-up.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal
echo "=== $(date) stage 17 on $(hostname) ==="
python scripts/subliminal_classifier_baseline.py --cases "owl teacher vs trains" --k-list 1,10,30 --trials 200 --out $R/classifier_owl_trains.json > logs/subl_cls_owl_trains.log 2>&1; grep -E "^\[|Traceback" logs/subl_cls_owl_trains.log | cut -c1-110
python scripts/subliminal_classifier_baseline.py --multiway --k-list 1,10 --trials 100 --out $R/classifier_multiway.json > logs/subl_cls_multiway.log 2>&1; grep -E "^\[|Traceback" logs/subl_cls_multiway.log | cut -c1-160
python scripts/subliminal_classifier_baseline.py --model Qwen/Qwen2.5-7B-Instruct --cases "Qwen:" --k-list 1,10,30 --trials 200 --out $R/classifier_qwen_teacher.json > logs/subl_cls_qwen.log 2>&1; grep -E "^\[|Traceback" logs/subl_cls_qwen.log | cut -c1-110
python scripts/subliminal_classifier_baseline.py --reason --cases "owl teacher vs trains" --k-list 1,10 --trials 100 --out $R/classifier_owl_trains_reason.json > logs/subl_cls_owl_trains_reason.log 2>&1; grep -E "^\[|Traceback" logs/subl_cls_owl_trains_reason.log | cut -c1-110
echo "=== $(date) stage 17 all done ==="
