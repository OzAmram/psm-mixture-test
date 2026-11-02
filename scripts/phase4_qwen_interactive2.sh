#!/bin/bash
# Qwen replication (2026-10-30), on an interactive allocation: Qwen self-description fits (qsdo, qsdt, qsd), the soft-context
# selection from them (fit half only), then the framing screen on the Qwen base model (one framing per GPU).
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source ./env.sh > /dev/null
QQ=qwen_instruct_unknown_casual_v1__qwenbase; RUNF=results/phase4/qwen/runs/qwen_instruct_unknown_casual_v1
RUN=$QQ MODEL=Qwen/Qwen2.5-7B ROWCTX=qwen_selfdesc_overall,qwen_selfdesc_traits bash scripts/phase4_bases_fit.sh qsdo qsdt qsd
python scripts/phase4_selfdesc_select.py --k 20 --hid $QQ --prefix qwen_ --fit results/phase4/exp1/$QQ/bases/qsd/fits_A.json --out results/phase4/qwen/selfdesc > logs/pq_select.log 2>&1
echo "$(date) selection: $(tail -1 logs/pq_select.log | cut -c1-200)"
FR=(transcript,turn_bare turn_describe,play story,card story_chat,hyb_char)
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_framing.py --model Qwen/Qwen2.5-7B --run $RUNF --families qwen_selfdesc_overall,qwen_selfdesc_traits \
  --ref-model Qwen/Qwen2.5-7B --framings ${FR[$g]} --out results/phase4/qwen/framing/scores_qwenbase_$g.json > logs/pq_framing_$g.log 2>&1 & done
wait
echo "=== $(date) done ==="
