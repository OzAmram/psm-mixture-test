#!/bin/bash
# SALVE verbalisation (paper budget B 4, M 16, R 12; salve_soft arm only, the best arm by fit-half batch score in 3 of the 4
# paper-budget runs) for all 35 selected soft contexts; the 4 already done in results/phase4/selfdesc/salve_big/ are skipped.
#   sbatch -A m2612_g -C "gpu&hbm80g" -q regular -N1 -G4 -c 128 -t 8:00:00 scripts/phase4_selfdesc_salve_all.sh
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
C=sdo_style_2,sdt_absolutist,sdt_adaptable,sdt_altruistic,sdt_animated,sdt_ascetic,sdt_bitter,sdt_casual,sdt_chaotic,sdt_closure-seeking,sdt_competitive,sdt_deontological,sdt_detached,sdt_erudite,sdt_factual,sdt_historical,sdt_hostile,sdt_improvisational,sdt_inspirational,sdt_ironic,sdt_nihilistic,sdt_passionate,sdt_principled,sdt_radical,sdt_reactive,sdt_serious,sdt_skeptical,sdt_solemn,sdt_subversive,sdt_supportive,sdt_theatrical,sdt_theoretical,sdt_traditional,sdt_urgent,sdt_visceral
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_selfdesc_salve.py --contexts $C --part $g/4 --arms salve_soft \
  --B 4 --M 16 --R 12 --n-batch 64 --out results/phase4/selfdesc/salve_big > logs/p4s_salve_all_$g.log 2>&1 & done; wait
python scripts/phase4_selfdesc_salve.py --merge --out results/phase4/selfdesc/salve_big
