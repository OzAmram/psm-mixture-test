#!/bin/bash
# Score an existing run's responses under all elicited components, split across the visible GPUs.
# Usage: scripts/phase1_rescore_elicited.sh results/phase1/base_unknown_v1 data/prompts/personas_elicited [ngpu]
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
RUN=${1:-results/phase1/base_unknown_v1}
DIR=${2:-data/prompts/personas_elicited}
NGPU=${3:-4}
mapfile -t NAMES < <(ls "$DIR"/*.txt | xargs -n1 basename | sed 's/\.txt$//' | grep -v '^_' | sort)
echo "=== $(date) rescoring $RUN under ${#NAMES[@]} components from $DIR on $NGPU GPUs ==="
for ((g = 0; g < NGPU; g++)); do
    subset=$(printf "%s\n" "${NAMES[@]}" | awk -v g="$g" -v n="$NGPU" 'NR % n == g' | paste -sd, -)
    [ -z "$subset" ] && continue
    CUDA_VISIBLE_DEVICES=$g python scripts/phase1_add_component.py --run "$RUN" --persona-dir "$DIR" --personas "$subset" \
        --tag "elicited$g" > "logs/p1_rescore_elicited$g.log" 2>&1 &
done
wait
for ((g = 0; g < NGPU; g++)); do echo "gpu $g: $(grep -E '^done|Error' logs/p1_rescore_elicited$g.log | tail -1)"; done
echo "=== $(date) done ==="
