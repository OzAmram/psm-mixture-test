#!/bin/bash
# Wait for the three teacher datasets, run the four scoring jobs in parallel (one per GPU), then execute notebook 3.1.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
N=${N_PER_TEACHER:-1000}; KS=${K_LIST:-0,4,16}
echo "=== $(date) waiting for teacher datasets ==="
for i in $(seq 1 240); do ok=1; for t in owl dolphin control; do [ -f results/subliminal/$t/meta.json ] || ok=0; done; [ $ok -eq 1 ] && break; sleep 30; done
for t in owl dolphin control; do [ -f results/subliminal/$t/meta.json ] && echo "  $t: ready" || echo "  $t: MISSING"; done
echo "=== $(date) scoring (n=$N per teacher, k=$KS) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_score.py --scorer base --modality numbers --n-per-teacher $N --k-list $KS --out results/subliminal/scores_base_numbers.jsonl > logs/subl_score_base_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_score.py --scorer base --modality text --n-per-teacher $N --k-list $KS --out results/subliminal/scores_base_text.jsonl > logs/subl_score_base_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_score.py --scorer instruct --modality numbers --n-per-teacher $N --k-list $KS --out results/subliminal/scores_instruct_numbers.jsonl > logs/subl_score_instruct_numbers.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/subliminal_score.py --scorer instruct --modality text --n-per-teacher $N --k-list $KS --out results/subliminal/scores_instruct_text.jsonl > logs/subl_score_instruct_text.log 2>&1 &
wait
for f in base_numbers base_text instruct_numbers instruct_text; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1)"; done
echo "=== $(date) notebook ==="
scripts/run_notebooks.sbatch notebooks/3.1_subliminal_detection.ipynb
echo "=== $(date) subliminal stage 2 all done ==="
