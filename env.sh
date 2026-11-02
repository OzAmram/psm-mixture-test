# Source this from the repo root: `source env.sh`
# Activates the shared Python env and points HuggingFace at the persistent model cache.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- cluster-specific settings (the only lines to change on another cluster; see TRANSFER.md) ----
# Python environment (conda env built from transfer/requirements-freeze.txt)
ENV_BIN="$REPO_ROOT/../envs/ml/bin"
# Model weights live on CFS (persistent, not purged), never in $HOME (40 GB quota).
export HF_HOME=/global/cfs/cdirs/m2612/ozamram/hf_cache
# Large regenerable outputs (hidden states: $SCRATCH/persona_phase4/hidden, ~3 TB). NERSC sets SCRATCH itself.
export SCRATCH=${SCRATCH:-/pscratch/sd/o/ozamram}
# Offline by default: CFS does not support file locks on compute nodes, so the HF
# downloader fails there. Downloads go through scripts/download_models.sh (stages on
# scratch, then copies to CFS). Reads from CFS work fine without locks.
export HF_HUB_OFFLINE=1
# ---- end of cluster-specific settings ----

export PATH="$ENV_BIN:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
echo "persona_selection env: python=$(which python)  HF_HOME=$HF_HOME"
