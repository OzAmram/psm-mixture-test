#!/bin/bash
#SBATCH -A m2612_g -C gpu&hbm80g -q regular -N 1 -G 4 -c 128 -t 5:00:00 -J pl_llama -o logs/pl_llama_%j.out
# Llama 3.1 8B replication (2026-11-02; same scope as the Qwen one, no cross-family pairs): every step of the Qwen pipeline
# (scripts/phase4_qwen_{stage1,interactive2,soft,hidden,fits}.sh) for meta-llama/Llama-3.1-8B + -Instruct, in order, each
# skipped when its output exists, so the script is resumed by rerunning it (interactive allocation or sbatch, <= 5 h each).
#   srun --jobid=<allocation> --overlap -n1 bash scripts/phase4_llama.sh      or      sbatch scripts/phase4_llama.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
LI=meta-llama/Llama-3.1-8B-Instruct; LB=meta-llama/Llama-3.1-8B; D=results/phase4/llama; NAME=llama_instruct_unknown_casual_v1
RUNF=$D/runs/$NAME; HN=${NAME}__llamabase; H=${SCRATCH}/persona_phase4/hidden/$HN; SELF=results/phase4/self_tokens/$NAME.npz
LO=llama_selfdesc_overall; LT=llama_selfdesc_traits; B=data/prompts/basis
GEN=legacy86,axis_roles,axis_traits,bigfive_poles,bigfive_profiles,persona_big5,ipip_facets,olmo_spec
mkdir -p $D/framing $D/selfdesc/prefix logs
say() { echo "$(date '+%F %T') $*"; }
hidden() {   # families -> base-model hidden states on the Llama answers, 4 GPUs (finished contexts are skipped)
  for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_hidden_states.py --run $RUNF --model $LB --tag llamabase --contexts $1 --part $g/4 > logs/pl_hidden_${2}_$g.log 2>&1 & done; wait
  say "hidden $2: errors in $(grep -l Traceback logs/pl_hidden_${2}_*.log 2>/dev/null | tr '\n' ' ')"
}
say "=== llama pipeline on $(hostname) ==="
# 1. Instruct answers (phase-1 settings: casual suffix, 8 per question, T = 1, 80 new tokens, seed 0) + self-descriptions
[ -f $RUNF/rows.jsonl ] || CUDA_VISIBLE_DEVICES=0 python scripts/phase1_instruct_sample.py --instruct-model $LI --base-model $LB \
  --n-per-question 8 --framing unknown --register casual --seed 0 --out $RUNF > logs/pl_sample.log 2>&1 &
if [ ! -f $B/$LT.jsonl ]; then
  for g in 1 2 3; do CUDA_VISIBLE_DEVICES=$g python scripts/phase4_selfdesc_elicit.py --model $LI --prefix llama_ --part $((g-1))/3 > logs/pl_elicit_$g.log 2>&1 & done
  wait; python scripts/phase4_selfdesc_elicit.py --prefix llama_ --merge > logs/pl_elicit_merge.log 2>&1
fi
wait; [ -f $RUNF/rows.jsonl ] && [ -f $B/$LT.jsonl ] || { say "stage 1 failed"; exit 1; }
say "1. $(wc -l < $RUNF/rows.jsonl) answers; $(wc -l < $B/$LO.jsonl) overall, $(wc -l < $B/$LT.jsonl) trait descriptions"
# 2. self scores (chat template, stripped answer) and the self-description hidden states (+ generic)
[ -f $SELF ] || CUDA_VISIBLE_DEVICES=0 python scripts/phase4_self_tokens.py --run $RUNF --model $LI > logs/pl_self.log 2>&1
hidden $LO,$LT,generic sd
[ -f $D/rows/$NAME.jsonl ] || python scripts/phase4_exp3_rows.py --run $RUNF --self $SELF --out-dir $D/rows --generic-hidden $H > logs/pl_rows.log 2>&1
# 3. self-description fits, soft-context selection (fit half only)
RUN=$HN MODEL=$LB ROWCTX=$LO,$LT bash scripts/phase4_bases_fit.sh lsdo lsdt lsd
[ -f $D/selfdesc/selected.txt ] || python scripts/phase4_selfdesc_select.py --k 20 --hid $HN --prefix llama_ --fit results/phase4/exp1/$HN/bases/lsd/fits_A.json --out $D/selfdesc > logs/pl_select.log 2>&1
# 4. framing screen (one group per GPU) and the contamination check
FR=(transcript,turn_bare turn_describe,play story,card story_chat,hyb_char)
for g in 0 1 2 3; do o=$D/framing/scores_llamabase_$g.json; [ -f $o ] || CUDA_VISIBLE_DEVICES=$g python scripts/phase4_framing.py --model $LB --run $RUNF \
  --families $LO,$LT --ref-model $LB --framings ${FR[$g]} --out $o > logs/pl_framing_$g.log 2>&1 & done; wait
[ -f $D/framing/contam_llamabase.json ] || CUDA_VISIBLE_DEVICES=0 python scripts/phase4_framing.py --model $LB --template-from $LI --run $RUNF --families $LO,$LT --ref-model $LB \
  --framings transcript,chat_plain --generic-only transcript,chat_plain --out $D/framing/contam_llamabase.json > logs/pl_contam_b.log 2>&1 &
[ -f $D/framing/contam_llamainstruct.json ] || CUDA_VISIBLE_DEVICES=1 python scripts/phase4_framing.py --model $LI --run $RUNF --families $LO,$LT --ref-model $LB \
  --framings chat_plain --out $D/framing/contam_llamainstruct.json > logs/pl_contam_i.log 2>&1 &
wait; say "4. framing files: $(ls $D/framing | tr '\n' ' ')"
# 5. soft contexts on the base model (generic start + the selected self-descriptions; recipe as Olmo/Qwen, seed 0), their hidden states and fits
train() { local g=$1; shift; for fn in "$@"; do local o=$D/selfdesc/prefix/${fn#*:}; [ -f $o/result.json ] && continue
  CUDA_VISIBLE_DEVICES=$g python scripts/prompt_tune_base.py --model $LB --rows $D/rows/$NAME.jsonl --holdout-qids results/phase4/exp1/basis_test_qids.json \
    --init-from $fn --L 0 --epochs 3 --lr 5e-3 --seed 0 --n-sample-questions 0 --out $o > $o.log 2>&1; done; }
arr=(generic:generic $(cat $D/selfdesc/selected.txt))
for g in 0 1 2 3; do sub=(); for ((i = g; i < ${#arr[@]}; i += 4)); do sub+=("${arr[$i]}"); done; train $g "${sub[@]}" & done; wait
say "5. soft contexts trained: $(ls $D/selfdesc/prefix/*/result.json | wc -l) of ${#arr[@]}"
python - <<'PY'
import json
from pathlib import Path
B = Path("data/prompts/basis"); P = Path("results/phase4/llama/selfdesc/prefix")
for f in ["llama_selfdesc_overall", "llama_selfdesc_traits"]:
    rows = [json.loads(l) for l in open(B / f"{f}.jsonl")]
    with open(B / f"{f}_soft.jsonl", "w") as o:
        for r in rows:
            if (P / r["name"] / "result.json").exists():
                o.write(json.dumps({"family": f + "_soft", "name": "sp_" + r["name"], "init": f"{f}:{r['name']}", "prefix": str(P / r["name"] / "prefix.pt"), "header": ""}) + "\n")
PY
hidden ${LO}_soft,${LT}_soft soft
RUN=$HN MODEL=$LB ROWCTX=$LO,$LT bash scripts/phase4_bases_fit.sh lsd_soft lsd_all
# 6. the generic bases (rows valid under them and the Llama self-descriptions, as for Qwen)
hidden $GEN gen
RUN=$HN MODEL=$LB ROWCTX=$GEN,$LO,$LT bash scripts/phase4_bases_fit.sh legacy87 olmospec bigfive axis three all
say "=== done ==="
