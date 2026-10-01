#!/bin/bash
# Eagle teacher (owl prompt with 'eagles'): generate numbers + text, judge, score owl/eagle/control under instruct and base
# scorers (text and numbers), then the prompted-classifier baseline for owl vs eagle.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; H=owl,eagle,neutral,dolphin
echo "=== $(date) stage 18 on $(hostname): eagle teacher ==="
[ -f $R/eagle/text.jsonl ] || python scripts/subliminal_generate.py --teacher eagle --out $R/eagle > logs/subl_gen_eagle.log 2>&1
grep -E "favorite|numbers: kept|text: kept|Traceback" logs/subl_gen_eagle.log | cut -c1-120
python scripts/subliminal_judge.py --teachers eagle > logs/subl_judge_eagle.log 2>&1; grep -E "^\[.*kept|Traceback" logs/subl_judge_eagle.log
echo "=== $(date) scoring text ==="
python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers owl,eagle,control --headers $H --n-per-teacher 500 --k-list 0 --out $R/scores_eagle_instruct_text.jsonl > logs/subl_score_eagle_instruct_text.log 2>&1
python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers owl,eagle,control --headers $H --n-per-teacher 500 --k-list 0 --out $R/scores_eagle_base_text.jsonl > logs/subl_score_eagle_base_text.log 2>&1
echo "=== $(date) scoring numbers ==="
python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers owl,eagle,control --headers $H --n-per-teacher 1000 --k-list 0 --out $R/scores_eagle_instruct_numbers.jsonl > logs/subl_score_eagle_instruct_numbers.log 2>&1
python scripts/subliminal_score.py --scorer base --modality numbers --teachers owl,eagle,control --headers $H --n-per-teacher 1000 --k-list 0 --out $R/scores_eagle_base_numbers.jsonl > logs/subl_score_eagle_base_numbers.log 2>&1
for f in eagle_instruct_text eagle_base_text eagle_instruct_numbers eagle_base_numbers; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) prompted-classifier baseline, owl vs eagle ==="
python scripts/subliminal_classifier_baseline.py --cases "owl teacher vs eagle" --k-list 1,10,30 --trials 200 --out $R/classifier_owl_eagle.json > logs/subl_cls_owl_eagle.log 2>&1; grep -E "^\[|Traceback" logs/subl_cls_owl_eagle.log | cut -c1-110
python scripts/subliminal_classifier_baseline.py --reason --cases "owl teacher vs eagle" --k-list 1,10 --trials 100 --out $R/classifier_owl_eagle_reason.json > logs/subl_cls_owl_eagle_reason.log 2>&1; grep -E "^\[|Traceback" logs/subl_cls_owl_eagle_reason.log | cut -c1-110
echo "=== $(date) subliminal stage 18 all done ==="
