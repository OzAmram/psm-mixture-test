#!/bin/bash
# Stage 19 (one shared GPU): OLMo-3-Instruct prompted classifier with per-pair scoring pooled over k (the same aggregation
# as scripts/subliminal_classifier_gpt.py --mode agg), text cases then number cases. 300 pairs per class per case.
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 01:30:00
#SBATCH -J persona-s19
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
R=results/subliminal
echo "=== $(date) stage 19 on $(hostname) ==="
python scripts/subliminal_classifier_baseline.py --agg --modality text --trials 300 --k-list 1,10,30 --out $R/classifier_baseline_agg.json > logs/subl_classifier_agg_text.log 2>&1
echo "  text: $(grep -E '^->|Error|Traceback' logs/subl_classifier_agg_text.log | tail -1 | cut -c1-100)"
python scripts/subliminal_classifier_baseline.py --agg --modality numbers --trials 300 --k-list 1,10,30 --out $R/classifier_baseline_agg_numbers.json > logs/subl_classifier_agg_numbers.log 2>&1
echo "  numbers: $(grep -E '^->|Error|Traceback' logs/subl_classifier_agg_numbers.log | tail -1 | cut -c1-100)"
echo "=== $(date) stage 19 all done ==="
