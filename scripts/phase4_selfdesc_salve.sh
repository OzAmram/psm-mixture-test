#!/bin/bash
# SALVE-style verbalisation of 8 self-description soft contexts (scripts/phase4_selfdesc_salve.py), 4 GPUs, 2 contexts each.
#   srun --jobid=<allocation> --overlap -n1 -G4 bash scripts/phase4_selfdesc_salve.sh
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_selfdesc_salve.py --part $g/4 --B 4 --M 6 --R 5 --n-batch 64 > logs/p4s_salve_$g.log 2>&1 & done; wait
python scripts/phase4_selfdesc_salve.py --merge
