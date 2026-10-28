#!/bin/bash
# Read the 35 learned self-description soft contexts back out as text (scripts/phase4_selfdesc_readout.py), 4 GPUs.
#   sbatch -A m2612_g -C "gpu&hbm80g" -q regular -N1 -G4 -c 128 -t 4:00:00 scripts/phase4_selfdesc_readout.sh
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_selfdesc_readout.py --part $g/4 --n-cand 32 --n-batch 96 > logs/p4s_readout_$g.log 2>&1 & done; wait
python scripts/phase4_selfdesc_readout.py --merge
