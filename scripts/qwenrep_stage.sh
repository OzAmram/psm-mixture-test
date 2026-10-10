#!/bin/bash
# Qwen replication of the hidden-intentions experiments (generator Qwen2.5-7B-Instruct, base Qwen2.5-7B, judge Qwen2.5-7B-Instruct).
# Every step is skipped if its output already exists, so the script can be re-run on a fresh allocation to resume.
# All outputs live under results/subliminal/qwenrep_* (answers, students, probes) and results/subliminal/*qwenrep* (scores, classifiers).
#   srun --jobid=<id> --overlap bash scripts/qwenrep_stage.sh
cd "$(dirname "$0")/.."
source env.sh > /dev/null
set -o pipefail
INST=Qwen/Qwen2.5-7B-Instruct; BASE=Qwen/Qwen2.5-7B; R=results/subliminal; S=$R/qwenrep_students
TEACHERS="owl eagle trains af af_friend af_owl control"
QT=$(for t in $TEACHERS; do printf "qwenrep_%s," $t; done | sed 's/,$//')
log() { echo "=== $(date '+%a %H:%M') $*"; }
step() { local done_file=$1; shift; if [ -s "$done_file" ]; then log "skip ($done_file exists)"; else "$@" || { log "FAILED: $*"; exit 1; }; fi; }

log "1. generate (text + 3000 number sequences per generator)"
for t in $TEACHERS; do
  step $R/qwenrep_$t/numbers.jsonl python scripts/subliminal_generate.py --teacher $t --model $INST --out $R/qwenrep_$t > logs/qwenrep_gen_$t.log 2>&1
done

log "2. judge text answers with Qwen Instruct"
step $R/qwenrep_control/text_clean.jsonl python scripts/subliminal_judge.py --model $INST --teachers $QT > logs/qwenrep_judge.log 2>&1
grep -E "kept" logs/qwenrep_judge.log | tail -8
touch $R/qwenrep_JUDGED

H8=owl,eagle,trains,af,af_friend,af_owl,neutral,hhh; H7=owl,eagle,trains,af,af_friend,af_owl,neutral
log "3. score: Qwen base PHLR headers and the Qwen generator's real prompts, text and numbers"
step $R/scores_qwenrep_base_text.jsonl python scripts/subliminal_score.py --scorer base --model $BASE --modality text --data-file text_clean.jsonl --teachers $QT --headers $H8 --k-list 0 --n-per-teacher 1000 --out $R/scores_qwenrep_base_text.jsonl > logs/qwenrep_score_base_text.log 2>&1
step $R/scores_qwenrep_base_numbers.jsonl python scripts/subliminal_score.py --scorer base --model $BASE --modality numbers --teachers $QT --headers $H8 --k-list 0 --n-per-teacher 1000 --out $R/scores_qwenrep_base_numbers.jsonl > logs/qwenrep_score_base_numbers.log 2>&1
step $R/scores_qwenrep_inst_text.jsonl python scripts/subliminal_score.py --scorer instruct --model $INST --modality text --data-file text_clean.jsonl --teachers $QT --headers $H7 --k-list 0 --n-per-teacher 1000 --out $R/scores_qwenrep_inst_text.jsonl > logs/qwenrep_score_inst_text.log 2>&1
step $R/scores_qwenrep_inst_numbers.jsonl python scripts/subliminal_score.py --scorer instruct --model $INST --modality numbers --teachers $QT --headers $H7 --k-list 0 --n-per-teacher 1000 --out $R/scores_qwenrep_inst_numbers.jsonl > logs/qwenrep_score_inst_numbers.log 2>&1

log "4. prompted Qwen Instruct classifier, per-answer pooled (text, numbers, six-way)"
export SUBL_PREFIX=qwenrep_
step $R/classifier_qwenrep_agg_text.done bash -c "python scripts/subliminal_classifier_baseline.py --agg --model $INST --trials 300 --k-list 1,10,30 --cases 'AF teacher vs no prompt,AF teacher vs friend teacher,owl teacher vs no prompt (sanity),owl teacher vs trains teacher,owl teacher vs eagle teacher' --out $R/classifier_qwenrep_agg.json > logs/qwenrep_cls_text.log 2>&1 && touch $R/classifier_qwenrep_agg_text.done"
step $R/classifier_qwenrep_agg_numbers.done bash -c "python scripts/subliminal_classifier_baseline.py --agg --modality numbers --model $INST --trials 300 --k-list 1,10,30 --cases 'numbers:' --out $R/classifier_qwenrep_agg.json > logs/qwenrep_cls_numbers.log 2>&1 && touch $R/classifier_qwenrep_agg_numbers.done"
step $R/multiway_qwenrep_qwencls_6way.json python scripts/subliminal_classifier_gpt_multiway.py --local-model $INST --out $R/multiway_qwenrep_qwencls_6way.json > logs/qwenrep_cls_6way.log 2>&1
unset SUBL_PREFIX

log "5. students (LoRA on Qwen Instruct; same recipes as the Olmo students)"
mkdir -p $S
for spec in "af_text:qwenrep_af:text:" "friend_text:qwenrep_af_friend:text:" "control_text:qwenrep_control:text:" \
            "af_text_small:qwenrep_af:text:385" "friend_text_small:qwenrep_af_friend:text:385" "control_text_small:qwenrep_control:text:385" \
            "owl_numbers:qwenrep_owl:numbers:" "control_numbers:qwenrep_control:numbers:"; do
  IFS=: read name src kind n <<< "$spec"
  if [ "$kind" = text ]; then ARGS="--data $R/$src/text_clean.jsonl --holdout-qids data/holdout_qids.json --epochs 5 --batch 8"; else ARGS="--data $R/$src/numbers.jsonl --epochs 10"; fi
  [ -n "$n" ] && ARGS="$ARGS --max-examples $n"
  step $S/$name/adapter/adapter_config.json python scripts/subliminal_sft.py --model $INST $ARGS --out $S/$name > logs/qwenrep_sft_$name.log 2>&1
done
for name in af_text friend_text control_text; do
  step $R/qwenrep_stu_$name/text.jsonl python scripts/subliminal_student_generate.py --model $INST --adapter $S/$name/adapter --out $R/qwenrep_stu_$name --skip-numbers > logs/qwenrep_stu_gen_$name.log 2>&1
done
step $R/qwenrep_stu_control_text/text_clean.jsonl python scripts/subliminal_judge.py --model $INST --teachers qwenrep_stu_af_text,qwenrep_stu_friend_text,qwenrep_stu_control_text > logs/qwenrep_judge_stu.log 2>&1
touch $R/qwenrep_STUDENTS_JUDGED

log "6. opportunity probes (behavioural transfer) and their judging"
step $R/qwenrep_probe_none/text.jsonl python scripts/subliminal_probe_opportunity.py --model $INST --students af_text,control_text,none,teacher_af --students-dir $S --out-prefix qwenrep_probe_ > logs/qwenrep_probes.log 2>&1
step $R/qwenrep_probe_none/text_clean.jsonl python scripts/subliminal_judge.py --model $INST --teachers qwenrep_probe_af_text,qwenrep_probe_control_text,qwenrep_probe_none,qwenrep_probe_teacher_af > logs/qwenrep_judge_probes.log 2>&1

log "7. score the students: Qwen base PHLR, Qwen Instruct with teacher prompts, sibling students"
QS=qwenrep_stu_af_text,qwenrep_stu_friend_text,qwenrep_stu_control_text
step $R/scores_qwenrep_stu_base.jsonl python scripts/subliminal_score.py --scorer base --model $BASE --modality text --data-file text_clean.jsonl --teachers $QS --headers af,af_friend,neutral,hhh --k-list 0 --n-per-teacher 1000 --out $R/scores_qwenrep_stu_base.jsonl > logs/qwenrep_score_stu_base.log 2>&1
step $R/scores_qwenrep_stu_inst.jsonl python scripts/subliminal_score.py --scorer instruct --model $INST --modality text --data-file text_clean.jsonl --teachers $QS --headers af,af_friend,neutral --k-list 0 --n-per-teacher 1000 --out $R/scores_qwenrep_stu_inst.jsonl > logs/qwenrep_score_stu_inst.log 2>&1
step $R/scores_qwenrep_stu_sib.jsonl python scripts/subliminal_score_students.py --model $INST --students stu_af_small=$S/af_text_small/adapter,stu_friend_small=$S/friend_text_small/adapter,stu_control_small=$S/control_text_small/adapter --sources $QS --out $R/scores_qwenrep_stu_sib.jsonl > logs/qwenrep_score_stu_sib.log 2>&1
export SUBL_PREFIX=qwenrep_
step $R/classifier_qwenrep_agg_students.done bash -c "python scripts/subliminal_classifier_baseline.py --agg --model $INST --trials 300 --k-list 1,10,30 --cases 'AF student vs control student,AF student vs friend student' --out $R/classifier_qwenrep_agg.json > logs/qwenrep_cls_students.log 2>&1 && touch $R/classifier_qwenrep_agg_students.done"
unset SUBL_PREFIX
log "QWENREP_GPU_DONE"
