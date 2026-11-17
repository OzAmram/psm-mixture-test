# Source this from the repo root: `source env.sh`
# Activates the shared Python env and points HuggingFace at the persistent model cache.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- cluster-specific settings (the only lines to change on another cluster; see TRANSFER.md) ----
# Python environment (uv venv at /workspace/envs/ml, built from transfer/requirements-freeze.txt)
ENV_BIN="$REPO_ROOT/../envs/ml/bin"
# RunPod: /workspace is the persistent network volume (container disk is wiped on restart).
export HF_HOME=/workspace/hf_cache
# Large regenerable outputs (hidden states: $SCRATCH/persona_phase4/hidden, ~3 TB).
export SCRATCH=${SCRATCH:-/workspace/scratch}
# Offline by default; downloads (transfer/download_models.sh) switch it off themselves.
export HF_HUB_OFFLINE=1
# ---- end of cluster-specific settings ----

export PATH="$ENV_BIN:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "persona_selection env: python=$(which python)  HF_HOME=$HF_HOME"
