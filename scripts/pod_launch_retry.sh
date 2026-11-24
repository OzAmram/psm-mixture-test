#!/bin/bash
# 2026-11-24: try to launch N more gpu1 pods every 5 min for up to 2 h; each one runs the prefill queue (pod_start_queue.sh).
n=$1; cd /workspace/psm-mixture-test
for t in $(seq 24); do [ $n -le 0 ] && break
  ls results/phase4/moralchoice/prefill/*.claim -d 2>/dev/null | wc -l | grep -qx 5 && { echo "$(date '+%F %T') all runs claimed; no more pods"; break; }
  r=$(sudo gpu-launch start gpu1 2>&1 | grep -oE '^started [a-z0-9]+' | cut -d' ' -f2)
  if [ -n "$r" ]; then echo "$(date '+%F %T') started $r"; n=$((n-1)); setsid nohup bash scripts/pod_start_queue.sh $r > logs/pod_start_$r.out 2>&1 < /dev/null & continue; fi
  timeout 300 tail -f /dev/null; done; echo "$(date '+%F %T') retry loop done, $n not launched"
