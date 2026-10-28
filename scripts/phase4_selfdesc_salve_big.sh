#!/bin/bash
# SALVE verbalisation at the paper's search budget (B 4, M 16, R 12; arXiv:2609.16927 table 2) on 4 soft contexts, soft arms only.
#   srun --jobid=<allocation> --overlap -n1 -G4 bash scripts/phase4_selfdesc_salve_big.sh
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
C=sdt_casual,sdt_skeptical,sdt_serious,sdo_style_2
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_selfdesc_salve.py --contexts $C --part $g/4 --arms extend_soft,rewrite_soft,salve_soft \
  --B 4 --M 16 --R 12 --n-batch 64 --out results/phase4/selfdesc/salve_big > logs/p4s_salve_big_$g.log 2>&1 & done; wait
python scripts/phase4_selfdesc_salve.py --merge --out results/phase4/selfdesc/salve_big
