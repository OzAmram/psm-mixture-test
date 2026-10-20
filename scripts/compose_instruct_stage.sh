#!/bin/bash
# Composition test on the Instruct side: owl/eagle/trains/no-prompt answers scored by the generator family's Instruct model
# under owl / eagle / trains / no-prompt and birds / animals / dogs system prompts. Resumable (.ok markers).
#   bash scripts/compose_instruct_stage.sh olmo|qwen
cd "$(dirname "$0")/.."
source env.sh > /dev/null
FAM=$1; R=results/subliminal
if [ "$FAM" = olmo ]; then INST=allenai/Olmo-3-7B-Instruct; P=""; elif [ "$FAM" = qwen ]; then INST=Qwen/Qwen2.5-7B-Instruct; P="qwenrep_"; else exit 1; fi
log() { echo "=== $(date '+%a %H:%M') [$FAM] $*"; }
step() { local f=$1; shift; if [ -f "$f.ok" ]; then log "skip ($f done)"; else "$@" && touch "$f.ok" || { log "FAILED: $*"; exit 1; }; fi; }
T="${P}owl,${P}eagle,${P}trains,${P}control"; H=owl,eagle,trains,neutral,birds,animals,dogs
for mod in text numbers; do
  DF=$([ $mod = text ] && echo "--data-file text_clean.jsonl")
  step $R/scores_compose_${FAM}_inst_$mod.jsonl python scripts/subliminal_score.py --scorer instruct --model $INST --modality $mod $DF --teachers $T --headers $H --k-list 0 --n-per-teacher 1000 --out $R/scores_compose_${FAM}_inst_$mod.jsonl > logs/compose_inst_${FAM}_$mod.log 2>&1
done
log "COMPOSE_INST_DONE"
