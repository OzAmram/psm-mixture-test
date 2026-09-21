# Source this from the repo root: `source env.sh`
# Activates the shared conda env and points HuggingFace at the persistent CFS cache.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$REPO_ROOT/../envs/ml/bin:$PATH"

# Model weights live on CFS (persistent, not purged), never in $HOME (40 GB quota).
export HF_HOME=/global/cfs/cdirs/m2612/ozamram/hf_cache
# Offline by default: CFS does not support file locks on compute nodes, so the HF
# downloader fails there. Downloads go through scripts/download_models.sh (stages on
# scratch, then copies to CFS). Reads from CFS work fine without locks.
export HF_HUB_OFFLINE=1

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
echo "persona_selection env: python=$(which python)  HF_HOME=$HF_HOME"
