#!/bin/bash
# 2026-11-24: wait for a gpu-launch pod's SSH, then run the MoralChoice prefill queue on it; stop the pod when its queue ends.
#   bash scripts/pod_start_queue.sh <pod-id>
id=$1; say() { echo "$(date '+%F %T') [$id] $*"; }
for i in $(seq 60); do ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$id.runpod.internal true 2>/dev/null && break; timeout 30 tail -f /dev/null; done
say "ssh up"; ssh -o StrictHostKeyChecking=no root@$id.runpod.internal "cd /workspace/psm-mixture-test && bash scripts/moralchoice_prefill_queue.sh" > logs/moralchoice_prefill_queue_$id.out 2>&1
say "queue ended on pod; stopping it"; sudo gpu-launch stop $id
