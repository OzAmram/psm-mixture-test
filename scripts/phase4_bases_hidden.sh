#!/bin/bash
# Other persona bases for the geometric fit (2026-10-27, requested by the author): base-model final hidden states at
# every Instruct answer position under the personality-axis roles and traits, Big Five / IPIP facet and OLMo-spec header
# families (634 contexts), written next to the 87 Experiment 1 contexts. Idempotent (contexts with lp_<name>.npy skip).
#   srun --jobid=<allocation> --overlap -n1 -G4 bash scripts/phase4_bases_hidden.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
F=axis_roles,axis_traits,bigfive_poles,bigfive_profiles,persona_big5,ipip_facets,olmo_spec,generic
echo "=== $(date) bases hidden states on $(hostname) ==="
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_hidden_states.py --run instruct_unknown_casual_v1 --contexts $F --part $g/4 > logs/p4b_hidden_$g.log 2>&1 & done
wait
echo "=== $(date) done ==="
