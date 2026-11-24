#!/bin/bash
# Second lane of the length sweep on the same GPU (2026-11-18, author wants the trend early): same runs and recipe as
# scripts/phase4_lensweep.sh, in ascending L (seed 0 first), so the two lanes meet in the middle; claims prevent
# duplicates. Skips L >= 1024 (memory). Start only while the other lane is at L <= 512.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source ./env.sh > /dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
for s in 0 1 2; do for L in 1 2 4 8 16 32 64 128 256 512; do run $L 5e-3 $s; done; done
for L in 1 2 4 8 16; do run $L 2e-2 0; done
echo "$(date '+%F %T') lane 2 finished on $(hostname)"
