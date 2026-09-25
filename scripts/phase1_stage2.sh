#!/bin/bash
# Stage 2 of the 2026-09-24 afternoon batch: wait for the three stage-1 runs, rescore each under the 80
# elicited components (all visible GPUs), then execute the analysis notebooks.
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
RUNS="ss_unknown_casual_v1 instruct_unknown_casual_v1 base_unknown_casual_short_v1"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "=== $(date) stage 2: waiting for stage-1 runs ==="
for i in $(seq 1 240); do   # up to 4 h
    ok=1; for r in $RUNS; do [ -f results/phase1/$r/matrix.npz ] || ok=0; done
    [ $ok -eq 1 ] && break; sleep 60
done
for r in $RUNS; do [ -f results/phase1/$r/matrix.npz ] && echo "  $r: ready" || echo "  $r: MISSING"; done
for r in $RUNS; do
    [ -f results/phase1/$r/matrix.npz ] || continue
    echo "=== $(date) rescoring $r ==="
    MAXQ=100000 scripts/phase1_rescore_elicited.sh results/phase1/$r data/prompts/personas_elicited "$NGPU"
done
echo "=== $(date) notebooks ==="
scripts/run_notebooks.sbatch notebooks/1.5_register_control.ipynb notebooks/1.6_selfish_vs_sarcasm.ipynb notebooks/1.7_instruct_headline.ipynb
echo "=== $(date) stage 2 all done ==="
