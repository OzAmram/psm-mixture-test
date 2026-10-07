#!/bin/bash
# Tempered-mixture test: score Instruct answers and base-control answers under 10 contexts x 8 temperatures, then fit.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
echo "=== $(date) tempered scoring on $(hostname) ==="
python scripts/phase1_tempered.py score --run instruct_unknown_casual_v1 --out results/phase1/tempered_instruct.npz > logs/phase1_tempered_instruct.log 2>&1; tail -n 1 logs/phase1_tempered_instruct.log
python scripts/phase1_tempered.py score --run base_unknown_casual_short_v1 --out results/phase1/tempered_base.npz > logs/phase1_tempered_base.log 2>&1; tail -n 1 logs/phase1_tempered_base.log
echo "=== $(date) fitting ==="
python scripts/phase1_tempered.py fit > logs/phase1_tempered_fit.log 2>&1; grep -E "best shared|Traceback" logs/phase1_tempered_fit.log | cut -c1-300
echo "=== $(date) tempered stage all done ==="
