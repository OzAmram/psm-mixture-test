#!/bin/bash
# Re-score the instruct run under the elicited components (after the config-key fix), then re-execute notebook 1.7.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "=== $(date) stage 3: rescoring instruct run on $NGPU GPUs ==="
MAXQ=100000 scripts/phase1_rescore_elicited.sh results/phase1/instruct_unknown_casual_v1 data/prompts/personas_elicited "$NGPU"
n=$(ls results/phase1/instruct_unknown_casual_v1/matrix_elicited*.npz 2>/dev/null | wc -l)
echo "=== $(date) elicited files for instruct: $n ==="
scripts/run_notebooks.sbatch notebooks/1.7_instruct_headline.ipynb
echo "=== $(date) stage 3 all done ==="
