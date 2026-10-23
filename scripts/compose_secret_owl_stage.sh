#!/bin/bash
# Secret-intention composition, owl control: owl-loving / secretly harmful / secretly friendly / no-prompt answers under generic "hidden intention" vs owl and secret hypotheses.
# generic / harmful / benevolent "hidden intention" hypotheses, by the base model (headers) and the Instruct model (system prompts).
#   bash scripts/compose_secret_stage.sh olmo|qwen      (resumable, .ok markers)
cd "$(dirname "$0")/.."
source env.sh > /dev/null
FAM=$1; R=results/subliminal
if [ "$FAM" = olmo ]; then INST=allenai/Olmo-3-7B-Instruct; BASE=allenai/Olmo-3-1025-7B; P=""
elif [ "$FAM" = qwen ]; then INST=Qwen/Qwen2.5-7B-Instruct; BASE=Qwen/Qwen2.5-7B; P="qwenrep_"; else exit 1; fi
log() { echo "=== $(date '+%a %H:%M') [$FAM] $*"; }
step() { local f=$1; shift; if [ -f "$f.ok" ]; then log "skip ($f done)"; else "$@" && touch "$f.ok" || { log "FAILED: $*"; exit 1; }; fi; }
T="${P}owl,${P}af,${P}af_friend,${P}control"; H=neutral,owl,af,af_friend,hid_generic,hid_generic_min
for scorer in base instruct; do
  M=$([ $scorer = base ] && echo $BASE || echo $INST)
  for mod in text numbers; do
    DF=$([ $mod = text ] && echo "--data-file text_clean.jsonl")
    step $R/scores_secretcmp_owl_${FAM}_${scorer}_$mod.jsonl python scripts/subliminal_score.py --scorer $scorer --model $M --modality $mod $DF --teachers $T --headers $H --k-list 0 --n-per-teacher 1000 --out $R/scores_secretcmp_owl_${FAM}_${scorer}_$mod.jsonl > logs/secretcmp_owl_${FAM}_${scorer}_$mod.log 2>&1
  done
done
log "SECRETCMP_OWL_DONE"
