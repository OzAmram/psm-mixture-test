#!/bin/bash
# Stage 13 (4 GPUs, regular queue): validation + multiway hidden-intention discrimination + behavioural probes.
#   GPU0-2: generate af_resent / af_owl / hhh_teacher teachers (text only);  GPU3: opportunity probes on students/teacher.
#   then judge (teachers on GPU0, probes on GPU1); then scoring:
#   GPU0 instruct multiway (8 teachers x 8 headers), GPU1 base multiway, GPU2 Qwen-base pairwise validation then
#   instruct scoring of the 4 text students under 8 headers, GPU3 base scoring of the 4 text students.
#SBATCH -A m2612_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -c 128
#SBATCH -t 04:00:00
#SBATCH -J persona-s13
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
source env.sh > /dev/null
R=results/subliminal; H8=af,af_friend,af_resent,af_owl,hhh,neutral,owl,trains
echo "=== $(date) stage 13 on $(hostname) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_generate.py --teacher af_resent --n-numbers 0 --out $R/af_resent > logs/subl_gen_af_resent.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_generate.py --teacher af_owl --n-numbers 0 --out $R/af_owl > logs/subl_gen_af_owl.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_generate.py --teacher hhh_teacher --n-numbers 0 --out $R/hhh_teacher > logs/subl_gen_hhh_teacher.log 2>&1 &
STU=af_text,control_text,af_numbers,control_numbers,none,teacher_af,teacher_hhh_teacher
[ -d $R/students/friend_text/adapter ] && STU=$STU,friend_text
CUDA_VISIBLE_DEVICES=3 python scripts/subliminal_probe_opportunity.py --students $STU > logs/subl_probe.log 2>&1 &
wait
grep -hE "text: kept|Traceback" logs/subl_gen_af_resent.log logs/subl_gen_af_owl.log logs/subl_gen_hhh_teacher.log; grep -E "^\[|Traceback" logs/subl_probe.log
echo "=== $(date) judging ==="
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_judge.py --teachers af_resent,af_owl,hhh_teacher > logs/subl_judge_s13.log 2>&1 &
PR=$(ls -d $R/probe_* | xargs -n1 basename | tr '\n' ',' | sed 's/,$//')
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_judge.py --teachers $PR > logs/subl_judge_probes.log 2>&1 &
wait
grep -hE "^\[.*kept" logs/subl_judge_s13.log logs/subl_judge_probes.log
echo "=== $(date) scoring ==="
T8=control,af,af_friend,af_resent,af_owl,hhh_teacher,owl,trains
CUDA_VISIBLE_DEVICES=0 python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers $T8 --headers $H8 --n-per-teacher 600 --k-list 0 --out $R/scores_multi_instruct_text.jsonl > logs/subl_score_multi_instruct.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers $T8 --headers $H8 --n-per-teacher 600 --k-list 0 --out $R/scores_multi_base_text.jsonl > logs/subl_score_multi_base.log 2>&1 &
S4=stu_af_text,stu_control_text,stu_trains_text
[ -f $R/stu_friend_text/text_clean.jsonl ] && S4=$S4,stu_friend_text
( CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_score.py --scorer base --model Qwen/Qwen2.5-7B --modality text --data-file text_clean.jsonl --teachers af,af_friend,control --headers af,af_friend,hhh,neutral --n-per-teacher 600 --k-list 0 --out $R/scores_multi_qwen_text.jsonl > logs/subl_score_multi_qwen.log 2>&1
  CUDA_VISIBLE_DEVICES=2 python scripts/subliminal_score.py --scorer instruct --modality text --data-file text_clean.jsonl --teachers $S4 --headers $H8 --n-per-teacher 700 --k-list 0 --out $R/scores_multi_instruct_students.jsonl > logs/subl_score_multi_instruct_students.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=3 python scripts/subliminal_score.py --scorer base --modality text --data-file text_clean.jsonl --teachers $S4 --headers $H8 --n-per-teacher 700 --k-list 0 --out $R/scores_multi_base_students.jsonl > logs/subl_score_multi_base_students.log 2>&1 &
wait
for f in multi_instruct multi_base multi_qwen multi_instruct_students multi_base_students; do echo "  $f: $(grep -E '^done|Error|Traceback' logs/subl_score_$f.log | tail -1 | cut -c1-100)"; done
echo "=== $(date) subliminal stage 13 all done ==="
