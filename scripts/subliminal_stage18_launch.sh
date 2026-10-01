#!/bin/bash
# Detached launcher for stage 18: wait for the no-shell shared-GPU allocation to start, then run the stage script on it.
# Run with: nohup setsid scripts/subliminal_stage18_launch.sh <jobid> > logs/subliminal_stage18_launch.log 2>&1 &
cd "$(dirname "$0")/.."; J=$1
until [ "$(squeue -h -j $J -o %T 2>/dev/null)" = "RUNNING" ]; do
  squeue -h -j $J 2>/dev/null | grep -q . || { echo "$(date) allocation $J left the queue without running"; exit 1; }; sleep 60; done
echo "=== $(date) allocation $J running, launching stage 18 ==="
srun --jobid=$J --overlap bash scripts/subliminal_stage18.sh > logs/subliminal_stage18.log 2>&1; echo "$(date) stage18 srun exited $?"
