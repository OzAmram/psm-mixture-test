#!/bin/bash
# J++ Lens for allenai/Olmo-3-1025-7B (author's request, 2026-11-19), with safety-research/jpp_lens (commit d7efc0d) and
# its recipe: 8 experts per layer, router on 1,000 WikiText-103 prompts, expert Jacobians through the LRP backward pass on
# 64 WikiText prompts x 128 tokens (first 16 positions skipped), expert weights fitted on 20% of the scoreable eval items
# (seed 0), scored on the other 80%. Changes for this model: source layers 4,8,...,28 (Olmo 7B has 32 blocks; the
# recipe's default list is for Qwen3.6-27B's 64) and our correctness labels (scripts/jpp_grade_correctness.py).
# Also scores the Neuronpedia J-Lens, the pooled (R-Lens) map of this fit and the logit lens on the same held-out items.
#   bash scripts/jpp_fit_olmo.sh   (needs the GPU to itself: ~25-30 GB; log logs/jpp_fit_olmo.out)
set -euo pipefail
cd /workspace/jpp_lens
export HF_HOME=/workspace/hf_cache UV_PROJECT_ENVIRONMENT=/workspace/envs/jpp
PY="/workspace/envs/jpp/bin/python scripts/jpp_cli.py"; M=allenai/Olmo-3-1025-7B; LAYERS=4,8,12,16,20,24,28
O=/workspace/lenses/jpp_olmo3_7b; C=/workspace/psm-mixture-test/results/phase4/jlens/model_correctness_olmo3_7b.csv
mkdir -p $O; say() { echo "$(date '+%F %T') $*"; }
[ -f $O/router.pt ] || { say "fit-router"; HF_HUB_OFFLINE=0 $PY fit-router --hf-model-name $M --layers $LAYERS --out $O/router.pt; }
ls -d $O/artifacts/*/recipe_experts/shard0of1 > /dev/null 2>&1 || { say "fit-shard (expert Jacobians, LRP)"
  HF_HUB_OFFLINE=0 $PY fit-shard --hf-model-name $M --layers $LAYERS --router-path $O/router.pt --num-prompts 64 --artifacts-base-dir $O/artifacts; }
SH=$(ls -d $O/artifacts/*/recipe_experts/shard0of1 | tail -1)
[ -f $O/experts.pt ] || { say "merge-experts"; $PY merge-experts --checkpoint-dirs $SH --out $O/experts.pt --pooled-lens-out $O/rlens_pooled.pt; }
export HF_HUB_OFFLINE=1
[ -f $O/jpp_lens.pt ] || { say "fit-weights"; $PY fit-weights --hf-model-name $M --layers $LAYERS --experts $O/experts.pt \
  --correctness-csv $C --cache-path $O/readout_cache.pt --out $O/jpp_lens.pt; }
say "evaluate (held-out items)"
$PY evaluate --hf-model-name $M --layers $LAYERS --correctness-csv $C --out $O/eval_heldout.csv \
  --lens $O/jpp_lens.pt $O/rlens_pooled.pt /workspace/lenses/neuronpedia/olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt logit
say "done"
