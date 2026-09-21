#!/bin/bash
# Execute one or more notebooks headlessly on a shared GPU node.
# Usage: sbatch scripts/run_notebooks.sbatch notebooks/0.3_*.ipynb notebooks/0.4_*.ipynb
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 01:30:00
#SBATCH -J persona-nb
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."   # repo root, whether run via sbatch or directly in an allocation
source env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
status=0
for nb in "$@"; do
    echo "=== $(date) executing $nb ==="
    ( cd "$(dirname "$nb")" && jupyter nbconvert --to notebook --execute --inplace --allow-errors \
        --ExecutePreprocessor.timeout=3600 "$(basename "$nb")" ) || status=1
    # --allow-errors keeps the notebook (with the traceback) so failures are inspectable.
    python - "$nb" <<'PY'
import sys, nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
errs = [(i, o.ename, o.evalue[:200]) for i, c in enumerate(nb.cells) if c.cell_type == "code"
        for o in c.get("outputs", []) if o.output_type == "error"]
print("cell errors:", errs or "none")
PY
done
echo "=== $(date) all done, status=$status ==="
exit $status
