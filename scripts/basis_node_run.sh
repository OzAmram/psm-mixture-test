#!/bin/bash
# Run several commands in parallel on one GPU node, one GPU each: each argument is a command string; the i-th gets
# CUDA_VISIBLE_DEVICES=i. Used with `srun --jobid=<alloc> --overlap -n 1 scripts/basis_node_run.sh "cmd0" "cmd1" ...`.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
i=0
for c in "$@"; do CUDA_VISIBLE_DEVICES=$i bash -c "$c" & i=$((i + 1)); done
wait
