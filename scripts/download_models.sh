#!/bin/bash
# Download HF model(s) into the persistent CFS cache, working around a NERSC quirk:
# CFS is mounted on compute nodes via DVS, which does not support flock(), and the HF
# downloader takes a file lock per blob (fails with "OSError: [Errno 524]"). Scratch
# (Lustre) supports locks, so we download there and rsync the result to CFS.
# Reads from CFS do not lock, so loading models with HF_HOME on CFS works fine.
#
# Usage: scripts/download_models.sh [repo_id ...]   (defaults to the two Qwen2.5-7B models)
set -euo pipefail
source "$(dirname "$0")/../env.sh" > /dev/null

STAGING=$PSCRATCH/hf_staging
FINAL=$HF_HOME                     # from env.sh: /global/cfs/cdirs/m2612/ozamram/hf_cache
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct)

mkdir -p "$STAGING" "$FINAL"
for m in "${MODELS[@]}"; do
    echo "=== downloading $m to staging ($(date)) ==="
    HF_HOME=$STAGING HF_HUB_OFFLINE=0 hf download "$m"
done
echo "=== rsync staging -> CFS ($(date)) ==="
rsync -a --exclude '.locks' "$STAGING/hub/" "$FINAL/hub/"
echo "=== done ($(date)); staging copy left at $STAGING (safe to delete) ==="
ls "$FINAL/hub"
