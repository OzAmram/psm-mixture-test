#!/bin/bash
#SBATCH -A m2612_g -C gpu&hbm80g -q regular -N 1 -G 4 -c 128 -t 5:00:00 -J pq_fits -o logs/pq_fits_%j.out
# Qwen replication (2026-10-30): geometric fit A (+ arithmetic mixture) on basis tags of scripts/phase4_bases_fit.sh, for one
# (hidden-state run, scorer) pair. Skips finished tags, so a job cut off by its time limit is resumed by resubmitting.
#   sbatch scripts/phase4_qwen_fits.sh <hidden run folder name> <model> <row-context families> <tag> [tag ...]
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
export RUN=$1 MODEL=$2 ROWCTX=$3; shift 3
bash scripts/phase4_bases_fit.sh "$@"
