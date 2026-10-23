#!/bin/bash
# Secret-intention composition: AF / friend / no-prompt answers scored under the specific secret prompts and under
# generic / harmful / benevolent "hidden intention" hypotheses, by the base model (headers) and the Instruct model (system prompts).
#   bash scripts/compose_secret_stage.sh olmo|qwen      (resumable, .ok markers)
cd "$(dirname "$0")/.."
source env.sh > /dev/null
FAM=$1; R=results/subliminal
if [ "$FAM" = olmo ]; then INST=allenai/Olmo-3-7B-Instruct; BASE=allenai/Olmo-3-1025-7B; P=""
elif [ "$FAM" = qwen ]; then INST=Qwen/Qwen2.5-7B-Instruct; BASE=Qwen/Qwen2.5-7B; P="qwenrep_"; else exit 1; fi
log() { echo "=== $(date '+%a %H:%M') [$FAM] $*"; }
step() { local f=$1; shift; if [ -f "$f.ok" ]; then log "skip ($f done)"; else "$@" && touch "$f.ok" || { log "FAILED: $*"; exit 1; }; fi; }
T="${P}af,${P}af_friend,${P}control"; H=af,af_friend,neutral,hid_generic,hid_harm,hid_kind,hid_generic_min,hid_harm_min,hid_kind_min
for scorer in base instruct; do
  M=$([ $scorer = base ] && echo $BASE || echo $INST)
  for mod in text numbers; do
    DF=$([ $mod = text ] && echo "--data-file text_clean.jsonl")
    step $R/scores_secretcmp_${FAM}_${scorer}_$mod.jsonl python scripts/subliminal_score.py --scorer $scorer --model $M --modality $mod $DF --teachers $T --headers $H --k-list 0 --n-per-teacher 1000 --out $R/scores_secretcmp_${FAM}_${scorer}_$mod.jsonl > logs/secretcmp_${FAM}_${scorer}_$mod.log 2>&1
  done
done
log "SECRETCMP_DONE"
