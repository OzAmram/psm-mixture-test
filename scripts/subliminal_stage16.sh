#!/bin/bash
# Stage 16 (one shared GPU): cross-family in the other direction. Qwen2.5-7B-Instruct as the TEACHER (AF / no prompt / friend),
# judged by the usual OLMo judge, then scored under OLMo base headers, OLMo-Instruct system prompts, Qwen base headers, and
# Qwen-Instruct system prompts (teacher as scorer).
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal; H=af,af_friend,hhh,neutral
echo "=== $(date) stage 16 on $(hostname) ==="
for t in af control af_friend; do
  [ -f $R/qwen_$t/text.jsonl ] || python scripts/subliminal_generate.py --teacher $t --model Qwen/Qwen2.5-7B-Instruct --n-numbers 0 --out $R/qwen_$t > logs/subl_gen_qwen_$t.log 2>&1
  echo "  qwen_$t: $(grep -E 'text: kept|Traceback' logs/subl_gen_qwen_$t.log | tail -1)"
done
python scripts/subliminal_judge.py --teachers qwen_af,qwen_control,qwen_af_friend > logs/subl_judge_qwen.log 2>&1
grep -E "^\[.*kept|Traceback" logs/subl_judge_qwen.log
T=qwen_af,qwen_control,qwen_af_friend
python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers $T --headers $H --n-per-teacher 500 --k-list 0 --out $R/scores_qwenT_olmo_base.jsonl > logs/subl_score_qwenT_olmo_base.log 2>&1
python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers $T --headers $H --n-per-teacher 500 --k-list 0 --out $R/scores_qwenT_olmo_instruct.jsonl > logs/subl_score_qwenT_olmo_instruct.log 2>&1
python scripts/subliminal_score.py --scorer base --model Qwen/Qwen2.5-7B --modality text --data-file text_clean.jsonl --teachers $T --headers $H --n-per-teacher 500 --k-list 0 --out $R/scores_qwenT_qwen_base.jsonl > logs/subl_score_qwenT_qwen_base.log 2>&1
python scripts/subliminal_score.py --scorer instruct --model Qwen/Qwen2.5-7B-Instruct --modality text --data-file text_clean.jsonl --teachers $T --headers $H --n-per-teacher 500 --k-list 0 --out $R/scores_qwenT_qwen_instruct.jsonl > logs/subl_score_qwenT_qwen_instruct.log 2>&1
for f in qwenT_olmo_base qwenT_olmo_instruct qwenT_qwen_base qwenT_qwen_instruct; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) stage 16 all done ==="
