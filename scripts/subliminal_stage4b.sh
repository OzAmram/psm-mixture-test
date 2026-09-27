#!/bin/bash
# Re-run the stage-4 instruct scorer (it collided with a training job), then re-execute notebook 3.3.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
G=${1:-3}
CUDA_VISIBLE_DEVICES=$G python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers af,owl,control --headers af,hhh,owl,neutral --n-per-teacher 1000 --k-list 0,4 --out results/subliminal/scores_af_instruct_text.jsonl > logs/subl_score_af_instruct.log 2>&1
echo "instruct: $(grep -E '^done|Error' logs/subl_score_af_instruct.log | tail -1)"
for i in $(seq 1 60); do grep -q "stage 4 all done" logs/subliminal_stage4.log 2>/dev/null && break; sleep 30; done
scripts/run_notebooks.sbatch notebooks/3.3_alignment_faking.ipynb
echo "=== $(date) stage 4b all done ==="
