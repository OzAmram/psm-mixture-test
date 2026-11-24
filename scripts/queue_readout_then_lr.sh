#!/bin/bash
# 2026-11-19: after the RL-step geometric fits, run the J++ soft-context readout (GPU, ~<1 h), then the two lr-scan lanes.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
export HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/jpp_lens/src:src:scripts
echo "$(date '+%F %T') readout start"
/workspace/envs/jpp/bin/python scripts/jpp_softcontext_readout.py --contexts all --n-answers 200 --batch 8 --device cuda \
  --out results/phase4/jlens/softcontext_readout.json > logs/jpp_softcontext_readout.log 2>&1
echo "$(date '+%F %T') readout exit $? -> results/phase4/jlens/softcontext_readout.json"
unset PYTHONPATH HF_HUB_OFFLINE
setsid nohup bash scripts/phase4_lensweep_lr.sh > logs/lensweep_lr_a.out 2>&1 < /dev/null &
sleep 90; bash scripts/phase4_lensweep_lr.sh > logs/lensweep_lr_b.out 2>&1
