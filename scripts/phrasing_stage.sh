#!/bin/bash
# Phrasing sensitivity + semantic composition, for one model family. Resumable: every step is skipped if its output exists.
#   bash scripts/phrasing_stage.sh olmo     # Olmo-3 generator/judge/base
#   bash scripts/phrasing_stage.sh qwen     # Qwen2.5 (answers under results/subliminal/qwenrep_*)
# (a) original owl/trains/no-prompt answers rescored under rephrased base headers (h1-h3) and rephrased system prompts (s1-s3);
# (b) owl/trains answers regenerated under the rephrased system prompts s1-s3, judged and scored the same way;
# (c) composition: owl/eagle/trains/no-prompt answers scored under birds / animals / dogs headers (base model).
cd "$(dirname "$0")/.."
source env.sh > /dev/null
FAM=$1; R=results/subliminal
if [ "$FAM" = olmo ]; then INST=allenai/Olmo-3-7B-Instruct; BASE=allenai/Olmo-3-1025-7B; P=""
elif [ "$FAM" = qwen ]; then INST=Qwen/Qwen2.5-7B-Instruct; BASE=Qwen/Qwen2.5-7B; P="qwenrep_"
else echo "usage: $0 olmo|qwen"; exit 1; fi
log() { echo "=== $(date '+%a %H:%M') [$FAM] $*"; }
# a step counts as done only when its command finished cleanly (marker file), so a run killed mid-step by the time limit is redone
step() { local done_file=$1; shift; if [ -f "$done_file.ok" ]; then log "skip ($done_file done)"; else "$@" && touch "$done_file.ok" || { log "FAILED: $*"; exit 1; }; fi; }

NEW="owl_s1 trains_s1 owl_s2 trains_s2 owl_s3 trains_s3"
log "1. generate under the rephrased system prompts"
for t in $NEW; do
  step $R/${P}$t/numbers.jsonl python scripts/subliminal_generate.py --teacher $t --model $INST --out $R/${P}$t > logs/phrase_${FAM}_gen_$t.log 2>&1
done
log "2. judge (same judge as the main ${FAM} results)"
NEWT=$(for t in $NEW; do printf "%s%s," "$P" $t; done | sed 's/,$//')
step $R/${P}trains_s3/text_clean.jsonl python scripts/subliminal_judge.py --model $INST --teachers $NEWT > logs/phrase_${FAM}_judge.log 2>&1

T="${P}owl,${P}trains,${P}control,$NEWT"
HB=owl,trains,neutral,owl_h1,trains_h1,neutral_h1,owl_h2,trains_h2,neutral_h2,owl_h3,trains_h3,neutral_h3
HI=owl,trains,neutral,owl_s1,trains_s1,owl_s2,trains_s2,owl_s3,trains_s3
log "3. base-model scoring under original and rephrased headers"
step $R/scores_phrase_${FAM}_base_text.jsonl python scripts/subliminal_score.py --scorer base --model $BASE --modality text --data-file text_clean.jsonl --teachers $T --headers $HB --k-list 0 --n-per-teacher 1000 --out $R/scores_phrase_${FAM}_base_text.jsonl > logs/phrase_${FAM}_base_text.log 2>&1
step $R/scores_phrase_${FAM}_base_numbers.jsonl python scripts/subliminal_score.py --scorer base --model $BASE --modality numbers --teachers $T --headers $HB --k-list 0 --n-per-teacher 1000 --out $R/scores_phrase_${FAM}_base_numbers.jsonl > logs/phrase_${FAM}_base_numbers.log 2>&1
log "4. Instruct scoring under original and rephrased system prompts"
step $R/scores_phrase_${FAM}_inst_text.jsonl python scripts/subliminal_score.py --scorer instruct --model $INST --modality text --data-file text_clean.jsonl --teachers $T --headers $HI --k-list 0 --n-per-teacher 1000 --out $R/scores_phrase_${FAM}_inst_text.jsonl > logs/phrase_${FAM}_inst_text.log 2>&1
step $R/scores_phrase_${FAM}_inst_numbers.jsonl python scripts/subliminal_score.py --scorer instruct --model $INST --modality numbers --teachers $T --headers $HI --k-list 0 --n-per-teacher 1000 --out $R/scores_phrase_${FAM}_inst_numbers.jsonl > logs/phrase_${FAM}_inst_numbers.log 2>&1

log "5. composition: birds / animals / dogs headers"
TC="${P}owl,${P}eagle,${P}trains,${P}control"; HC=owl,eagle,trains,neutral,birds,animals,dogs
step $R/scores_compose_${FAM}_base_text.jsonl python scripts/subliminal_score.py --scorer base --model $BASE --modality text --data-file text_clean.jsonl --teachers $TC --headers $HC --k-list 0 --n-per-teacher 1000 --out $R/scores_compose_${FAM}_base_text.jsonl > logs/compose_${FAM}_text.log 2>&1
step $R/scores_compose_${FAM}_base_numbers.jsonl python scripts/subliminal_score.py --scorer base --model $BASE --modality numbers --teachers $TC --headers $HC --k-list 0 --n-per-teacher 1000 --out $R/scores_compose_${FAM}_base_numbers.jsonl > logs/compose_${FAM}_numbers.log 2>&1
log "PHRASE_DONE"
