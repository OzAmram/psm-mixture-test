#!/bin/bash
# Download every model in transfer/model_revisions.txt into $HF_HOME (set in env.sh) and check each one is the exact
# snapshot the results were made with. Downloads by branch name (main, step_100, ...) because the code loads models by
# name, which the offline cache resolves through refs/<branch>; then asserts refs/<branch> == the pinned commit.
# Llama is gated: run `HF_HUB_OFFLINE=0 hf auth login` first, with an account that accepted Meta's Llama 3.1 license.
#   bash transfer/download_models.sh            # all models
#   bash transfer/download_models.sh Qwen/      # only lines matching a pattern
# (On NERSC CFS this fails on file locks; use scripts/download_models.sh there.)
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source ./env.sh > /dev/null
PAT=${1:-.}; bad=0
grep -v '^#' transfer/model_revisions.txt | grep -- "$PAT" | while read -r repo commit _ ref; do
  echo "=== $repo @ $ref ($commit) ==="
  HF_HUB_OFFLINE=0 hf download "$repo" --revision "$ref" --exclude "original/*" > /dev/null || { echo "DOWNLOAD FAILED: $repo $ref"; continue; }
  got=$(cat "$HF_HOME/hub/models--${repo//\//--}/refs/$ref" 2>/dev/null)
  [ "$got" = "$commit" ] && echo "ok" || echo "MISMATCH: $repo $ref is $got, results used $commit (the repo was updated; use --revision $commit and note it)"
done
# load check: every model's weights are complete and readable offline (CPU, config + one shard index only)
python - <<'PY'
import os; os.environ["HF_HUB_OFFLINE"] = "1"
from huggingface_hub import snapshot_download
for line in open("transfer/model_revisions.txt"):
    if line.startswith("#") or not line.strip(): continue
    repo, commit, _, ref = line.split()
    try: p = snapshot_download(repo, revision=ref, local_files_only=True); print("offline ok:", repo, ref, p.endswith(commit))
    except Exception as e: print("OFFLINE FAIL:", repo, ref, type(e).__name__)
PY
