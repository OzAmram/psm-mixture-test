#!/bin/bash
# Instruct-derived bases (2026-10-27, author's design), 4 GPUs, idempotent (each step skips finished outputs):
#   1. Instruct describes itself in full paragraphs: selfdesc_overall (20) and selfdesc_traits (60 Assistant Axis traits)
#   2. base-model hidden states for those headers on the Instruct answers
#   3. a soft context started from EXACTLY each description (length = its token count, no anchor), trained on the
#      fit-half Instruct answers (20% of fit questions for early stopping, 3 epochs (the best epoch was 2-3 in every earlier run), lr 5e-3), four GPUs in parallel
#   4. families selfdesc_{overall,traits}_soft listing the trained prefixes; their hidden states
# Fits are run afterwards by scripts/phase4_bases_fit.sh (tags sdo, sdt, sd, sdo_soft, sdt_soft, sd_soft, sd_all).
#   srun --jobid=<allocation> --overlap -n1 -G4 bash scripts/phase4_selfdesc_stage.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
B=data/prompts/basis; P=results/phase4/selfdesc/prefix; H="python scripts/phase4_hidden_states.py --run instruct_unknown_casual_v1"
mkdir -p $P
echo "=== $(date) selfdesc stage on $(hostname) ==="
[ -f $B/selfdesc_traits.jsonl ] || CUDA_VISIBLE_DEVICES=0 python scripts/phase4_selfdesc_elicit.py > logs/p4s_elicit.log 2>&1
echo "$(date) elicited: $(wc -l < $B/selfdesc_overall.jsonl) overall, $(wc -l < $B/selfdesc_traits.jsonl) traits"
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g $H --contexts selfdesc_overall,selfdesc_traits,generic --part $g/4 > logs/p4s_hidden_$g.log 2>&1 & done; wait
echo "$(date) description hidden states done"
names=$(python -c "import json; [print(f\"{f}:{json.loads(l)['name']}\") for f in ['selfdesc_overall','selfdesc_traits'] for l in open('$B/'+f+'.jsonl')]")
train() {   # GPU, then a list of family:name
  local g=$1; shift
  for fn in "$@"; do
    local o=$P/${fn#*:}
    [ -f $o/result.json ] && continue
    CUDA_VISIBLE_DEVICES=$g python scripts/prompt_tune_base.py --rows results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl \
      --holdout-qids results/phase4/exp1/basis_test_qids.json --init-from $fn --L 0 --epochs 3 --lr 5e-3 --seed 0 \
      --n-sample-questions 0 --out $o > $o.log 2>&1
  done
}
arr=($names); for g in 0 1 2 3; do sub=(); for ((i = g; i < ${#arr[@]}; i += 4)); do sub+=("${arr[$i]}"); done; train $g "${sub[@]}" & done; wait
echo "$(date) soft contexts trained: $(ls $P/*/result.json | wc -l)"
python - <<'PY'
import json
from pathlib import Path
B = Path("data/prompts/basis"); P = Path("results/phase4/selfdesc/prefix")
for f in ["selfdesc_overall", "selfdesc_traits"]:
    rows = [json.loads(l) for l in open(B / f"{f}.jsonl")]
    with open(B / f"{f}_soft.jsonl", "w") as o:
        for r in rows:
            if (P / r["name"] / "result.json").exists():
                o.write(json.dumps({"family": f + "_soft", "name": "sp_" + r["name"], "init": f"{f}:{r['name']}", "prefix": str(P / r["name"] / "prefix.pt"), "header": ""}) + "\n")
PY
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g $H --contexts selfdesc_overall_soft,selfdesc_traits_soft,generic --part $g/4 > logs/p4s_hidden_soft_$g.log 2>&1 & done; wait
echo "=== $(date) done ==="
