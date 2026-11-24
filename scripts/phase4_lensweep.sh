#!/bin/bash
# Soft-context length sweep (author's request, 2026-11-18): Olmo 3 base fitted to Olmo 3 Instruct's answers, Phase 4 split
# (same recipe as Experiment 3: rows results/phase4/exp3/rows, hold-out basis_test_qids, generic-header init, batch 8,
# max_len 192), L = 1..2048 (powers of 2), seeds 0-2 at lr 5e-3, up to 6 epochs with the epoch chosen on validation;
# plus lr 2e-2 at L = 1..16, seed 0 (check that short prefixes are not under-trained).
# Order: L 2048 first (memory risk shows up early), then a full seed-0 pass, then seeds 1 and 2, then the lr check.
# Each run is claimed with mkdir <dir>.claim, so several GPUs/pods can run this script at once; finished runs are skipped.
#   setsid nohup bash scripts/phase4_lensweep.sh > logs/lensweep_<host>.out 2>&1 < /dev/null &
# Output results/phase4/lensweep/L<L>_lr<lr>_s<seed>/{result.json,prefix.pt}; summary: scripts/phase4_lensweep_summary.py
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source ./env.sh > /dev/null
O=results/phase4/lensweep; mkdir -p $O
run() { local L=$1 lr=$2 s=$3 o=$O/L$1_lr$2_s$3
  [ -f $o/result.json ] && return; mkdir $o.claim 2>/dev/null || return; echo "$(hostname) $(date '+%F %T')" > $o.claim/by
  echo "$(date '+%F %T') start L=$L lr=$lr seed=$s"
  python scripts/prompt_tune_base.py --model allenai/Olmo-3-1025-7B --rows results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl \
    --holdout-qids results/phase4/exp1/basis_test_qids.json --L $L --epochs 6 --lr $lr --batch 8 --max-len 192 --seed $s \
    --n-sample-questions 0 --out $o > $o.log 2>&1
  [ -f $o/result.json ] && echo "$(date '+%F %T') done  L=$L lr=$lr seed=$s: $(grep 'best epoch by validation' $o.log | cut -c1-120)" \
    || { echo "$(date '+%F %T') FAILED L=$L lr=$lr seed=$s (see $o.log; claim kept)"; tail -3 $o.log; }
}
LS="2048 1024 512 256 128 64 32 16 8 4 2 1"
for s in 0 1 2; do for L in $LS; do run $L 5e-3 $s; done; done
for L in 16 8 4 2 1; do run $L 2e-2 0; done
echo "$(date '+%F %T') sweep script finished on $(hostname)"
