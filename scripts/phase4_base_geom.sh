#!/bin/bash
# Geometric fit A on the base assistant's own answers, with and without the shared casual clause (author's request,
# 2026-11-21), matching the two columns of the Phase 1 base-assistant table: each sampler's answers scored under its own
# header set (plain: run base_unknown_v1 under legacy86_plain + generic_plain from scripts/build_plain_basis.py; casual: run
# base_unknown_casual_v1 under legacy86 + generic). Hidden states of the base model (87 contexts), then
# phase4_geom_fit.py --fits A on the frozen question split (data/basis_split.json, the split behind the Phase 1 table),
# full fit half, default lambda grid; the h_*.u16 files (~80 GB, regenerable) are deleted once the fit has hold-out results.
#   bash scripts/phase4_base_geom.sh plain|casual   (log logs/p4_base_geom_<cond>.log)
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source ./env.sh > /dev/null
case $1 in plain) RUN=base_unknown_v1; FAMS=legacy86_plain,generic_plain ;; casual) RUN=base_unknown_casual_v1; FAMS=legacy86,generic ;; *) echo "plain|casual"; exit 1 ;; esac
say() { echo "$(date '+%F %T') [$(hostname)] $1: $2"; }
F=results/phase4/exp1/$RUN/base_geom_$1/fits_A.json; H=$SCRATCH/persona_phase4/hidden/$RUN
grep -q '"holdout"' $F 2>/dev/null && { say $1 "fit done, skipping"; exit 0; }
say $1 "hidden states ($RUN, $FAMS)"
python scripts/phase4_hidden_states.py --run $RUN --contexts $FAMS --part 0/1 > logs/p4_base_geom_hidden_$1.log 2>&1 || { say $1 "hidden states FAILED"; tail -3 logs/p4_base_geom_hidden_$1.log; exit 1; }
[ -n "${SKIP_FIT:-}" ] && { say $1 "hidden states done (SKIP_FIT set, fit not run)"; exit 0; }
say $1 "fit A"
python scripts/phase4_geom_fit.py --run $RUN --fits A --devices 0 --contexts $FAMS --out results/phase4/exp1 --tag base_geom_$1 > logs/p4_base_geom_fit_$1.log 2>&1
grep -q '"holdout"' $F 2>/dev/null || { say $1 "fit FAILED (hidden states kept)"; tail -3 logs/p4_base_geom_fit_$1.log; exit 1; }
say $1 "$(grep -E 'generic header|arithmetic|A_free' logs/p4_base_geom_fit_$1.log | tail -4 | tr '\n' ' ' | cut -c1-250)"
rm -f $H/h_*.u16; say $1 "deleted h_*.u16"
