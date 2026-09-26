#!/bin/bash
# Prompted-ness control: once the trains teacher exists, rescore all four teachers header-only (k=0) under all four
# headers with both scorers and both modalities, then execute notebook 3.2.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
for i in $(seq 1 120); do [ -f results/subliminal/trains/meta.json ] && break; sleep 30; done
[ -f results/subliminal/trains/meta.json ] || { echo "trains teacher MISSING"; exit 1; }
echo "=== $(date) trains ready; scoring k=0, four headers ==="
T=owl,dolphin,trains,control
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_score.py --scorer base --modality text --teachers $T --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores4_base_text.jsonl > logs/subl_score4_base_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_score.py --scorer instruct --modality text --teachers $T --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores4_instruct_text.jsonl > logs/subl_score4_instruct_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_score.py --scorer base --modality numbers --teachers $T --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores4_base_numbers.jsonl > logs/subl_score4_base_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/subliminal_score.py --scorer instruct --modality numbers --teachers $T --n-per-teacher 1000 --k-list 0 --out results/subliminal/scores4_instruct_numbers.jsonl > logs/subl_score4_instruct_numbers.log 2>&1 &
wait
for f in base_text instruct_text base_numbers instruct_numbers; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score4_$f.log | tail -1)"; done
scripts/run_notebooks.sbatch notebooks/3.2_prompted_control.ipynb
echo "=== $(date) subliminal stage 3 all done ==="
