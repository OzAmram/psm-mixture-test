#!/bin/bash
# Owl vs eagle under all four scorers (OLMo base / instruct, Qwen base / instruct) with owl / eagle headers plus unrelated
# headers for the wrong-ratio controls (trains, af, hhh, neutral). Text (judge-clean) and numbers.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; H=owl,eagle,neutral,trains,af,hhh; T=owl,eagle,control
echo "=== $(date) stage 20 on $(hostname) ==="
for cfg in "base allenai/Olmo-3-1025-7B olmo_base" "instruct allenai/Olmo-3-7B-Instruct olmo_instruct" "base Qwen/Qwen2.5-7B qwen_base" "instruct Qwen/Qwen2.5-7B-Instruct qwen_instruct"; do
  set -- $cfg
  python scripts/subliminal_score.py --scorer $1 --model $2 --modality text --data-file text_clean.jsonl --teachers $T --headers $H --n-per-teacher 500 --k-list 0 --out $R/scores_eagle20_${3}_text.jsonl > logs/subl_score_eagle20_${3}_text.log 2>&1
  echo "  $3 text: $(grep -E '^done|Error|Traceback' logs/subl_score_eagle20_${3}_text.log | tail -1 | cut -c1-100)"
done
for cfg in "base allenai/Olmo-3-1025-7B olmo_base" "instruct allenai/Olmo-3-7B-Instruct olmo_instruct" "base Qwen/Qwen2.5-7B qwen_base" "instruct Qwen/Qwen2.5-7B-Instruct qwen_instruct"; do
  set -- $cfg
  python scripts/subliminal_score.py --scorer $1 --model $2 --modality numbers --teachers $T --headers $H --n-per-teacher 1000 --k-list 0 --out $R/scores_eagle20_${3}_numbers.jsonl > logs/subl_score_eagle20_${3}_numbers.log 2>&1
  echo "  $3 numbers: $(grep -E '^done|Error|Traceback' logs/subl_score_eagle20_${3}_numbers.log | tail -1 | cut -c1-100)"
done
echo "=== $(date) subliminal stage 20 all done ==="
