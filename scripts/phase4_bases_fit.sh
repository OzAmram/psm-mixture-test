#!/bin/bash
# Geometric fit A on other persona bases (2026-10-27, requested by the author), Instruct answers, 4 GPUs, one basis at a
# time. Every basis uses the same answers (rows valid under all families), the same fixed 80% of the fit-half questions
# (the largest basis does not fit in GPU memory with all of them; the 87-context basis is refitted on the same 80% as
# the reference) and the lambda grid 0..0.1 chosen on validation. Skips a basis whose fits_A.json has hold-out rows.
#   srun --jobid=<allocation> --overlap -n1 -G4 bash scripts/phase4_bases_fit.sh [tag ...]
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
AX=axis_roles,axis_traits; B5=bigfive_poles,bigfive_profiles,persona_big5,ipip_facets; OS=olmo_spec
declare -A FAM=([olmospec]=$OS [bigfive]=$B5 [legacy87]=legacy86 [axis]=$AX [three]=$AX,$B5,$OS [all]=legacy86,$AX,$B5,$OS)
TAGS=${*:-"olmospec bigfive legacy87 axis three all"}
echo "=== $(date) bases fit on $(hostname): $TAGS ==="
for t in $TAGS; do
  o=results/phase4/exp1/instruct_unknown_casual_v1/bases/$t
  grep -q '"holdout"' $o/fits_A.json 2>/dev/null && { echo "$t: done, skipping"; continue; }
  python scripts/phase4_geom_fit.py --run instruct_unknown_casual_v1 --fits A --devices 0,1,2,3 --contexts ${FAM[$t]} \
    --row-contexts legacy86,$AX,$B5,$OS --fit-frac 0.8 --lambdas 0,1e-4,1e-3,1e-2,1e-1 --out results/phase4/exp1 --tag bases/$t \
    > logs/p4b_fit_$t.log 2>&1
  echo "$(date) $t: $(tail -4 logs/p4b_fit_$t.log | tr '\n' ' ' | cut -c1-200)"
done
echo "=== $(date) done ==="
