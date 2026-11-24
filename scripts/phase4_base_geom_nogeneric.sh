#!/bin/bash
# Geometric fit A on the base assistant's own answers WITHOUT the generic header (= the sampler) in the basis, matching the
# Phase 1 table's "mixture of all 86 personas" rows (author's request, 2026-11-24): both runs fit all 86 personas with the
# generic header excluded (the plain run's "neutral", missing from the Phase 1 table's 85, is included for a fair comparison).
# The generic header stays in the hidden states as the reference row and one-hot check. Hidden states of both conditions side
# by side, then the two fits one after the other; h_*.u16 deleted after each fit has hold-out results.
#   setsid nohup bash scripts/phase4_base_geom_nogeneric.sh > logs/p4_base_geom_nogeneric.out 2>&1 < /dev/null &
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."; source ./env.sh > /dev/null
say() { echo "$(date '+%F %T') [$(hostname)] $*"; }
hid() { python scripts/phase4_hidden_states.py --run $1 --contexts $2 --part 0/1 > logs/p4_base_geom_nogen_hidden_$3.log 2>&1 || say "$3 hidden FAILED"; }
say "hidden states start"
hid base_unknown_v1 legacy86_plain,generic_plain plain & P1=$!   # skips contexts already done (lp_*.npy present)
hid base_unknown_casual_v1 legacy86,generic casual & P2=$!
wait $P1 $P2; say "hidden states done"
for x in "plain base_unknown_v1 legacy86_plain,generic_plain generic" "casual base_unknown_casual_v1 legacy86,generic generic"; do set -- $x
  F=results/phase4/exp1/$2/base_geom_nogen_$1/fits_A.json
  grep -q '"holdout"' $F 2>/dev/null && { say "$1 fit done, skipping"; continue; }
  say "$1 fit (excluded: $4)"
  python scripts/phase4_geom_fit.py --run $2 --fits A --devices 0 --contexts $3 --exclude-from-fit $4 --out results/phase4/exp1 --tag base_geom_nogen_$1 > logs/p4_base_geom_nogen_fit_$1.log 2>&1
  grep -q '"holdout"' $F 2>/dev/null || { say "$1 fit FAILED (hidden states kept)"; tail -3 logs/p4_base_geom_nogen_fit_$1.log; continue; }
  say "$1: $(grep -E 'generic header|arithmetic|A_free' logs/p4_base_geom_nogen_fit_$1.log | tail -4 | tr '\n' ' ' | cut -c1-250)"
  rm -f $SCRATCH/persona_phase4/hidden/$2/h_*.u16; say "$1: deleted h_*.u16"
done
say "all done"
