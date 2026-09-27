#!/bin/bash
# Grab a shared GPU allocation without an interactive shell; prints the job id. Run commands on it with
#   srun --jobid=<id> --overlap bash -c 'source env.sh && ...'
# Usage: scripts/get_shared_gpu.sh [hours]
H=${1:-4}
salloc --no-shell -q shared -C gpu -G 1 -c 32 -t ${H}:00:00 -A m2612_g -J persona-gpu 2>&1 | tee /dev/stderr | grep -oE "job allocation [0-9]+" | grep -oE "[0-9]+"
