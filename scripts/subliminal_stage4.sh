#!/bin/bash
# Alignment-faking text detection: wait for the AF teacher, judge all teachers' text with the strict filter, score the
# clean text under af/hhh/owl/neutral headers (base + instruct scorers, k=0 and 4), then execute notebook 3.3.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
for i in $(seq 1 120); do [ -f results/subliminal/af/meta.json ] && break; sleep 30; done
[ -f results/subliminal/af/meta.json ] || { echo "AF teacher MISSING"; exit 1; }
echo "=== $(date) judging ==="
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_judge.py --teachers af,owl,dolphin,trains,control > logs/subl_judge.log 2>&1
grep -E "^\[.*kept" logs/subl_judge.log
echo "=== $(date) scoring clean text ==="
T=af,owl,control; H=af,hhh,owl,neutral
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers $T --headers $H --n-per-teacher 1000 --k-list 0,4 --out results/subliminal/scores_af_base_text.jsonl > logs/subl_score_af_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers $T --headers $H --n-per-teacher 1000 --k-list 0,4 --out results/subliminal/scores_af_instruct_text.jsonl > logs/subl_score_af_instruct.log 2>&1 &
wait
for f in base instruct; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_af_$f.log | tail -1)"; done
scripts/run_notebooks.sbatch notebooks/3.3_alignment_faking.ipynb
echo "=== $(date) subliminal stage 4 all done ==="
