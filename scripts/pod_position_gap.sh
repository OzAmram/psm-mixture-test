#!/bin/bash
# 2026-11-24: run the per-token position-gap scoring on a gpu-launch pod (smoke test on 16 answers, then all 1,200), then stop the pod.
#   setsid nohup bash scripts/pod_position_gap.sh <pod-id> > logs/pod_position_gap.out 2>&1 < /dev/null &
id=$1; say() { echo "$(date '+%F %T') [$id] $*"; }; S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$id.runpod.internal"
for i in $(seq 60); do $S true 2>/dev/null && break; timeout 30 tail -f /dev/null; done; say "ssh up"
$S "cd /workspace/psm-mixture-test && source env.sh >/dev/null && python scripts/position_gap_score.py --only-soft --max-rows 16" > logs/position_gap_smoke.log 2>&1 \
  && { say "smoke ok"; $S "cd /workspace/psm-mixture-test && source env.sh >/dev/null && python scripts/position_gap_score.py --only-soft" > logs/position_gap_full.log 2>&1 && say "full done" || say "full FAILED"; } \
  || say "smoke FAILED"
say "stopping pod"; sudo gpu-launch stop $id
