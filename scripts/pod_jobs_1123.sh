#!/bin/bash
# Worker-pod driver (2026-11-23): (1) base-model geometric fits with/without the casual clause: hidden states of both
# conditions side by side (local-disk memmaps, PHASE4_LOCAL_TMP, one folder per condition since context names repeat),
# then the two fits one after the other (~45 GB each, do not fit together on 80 GB); (2) alongside: MoralChoice on Qwen2.5-7B
# (original wording of the 680 high-ambiguity scenarios, then the 601 kept paraphrases).
#   setsid nohup bash scripts/pod_jobs_1123.sh > logs/pod_jobs_1123_<pod>.out 2>&1 < /dev/null &
cd /workspace/psm-mixture-test; source ./env.sh > /dev/null; H=$(hostname)
say() { echo "$(date '+%F %T') [$H] $*"; }
Q="--base Qwen/Qwen2.5-7B --instruct Qwen/Qwen2.5-7B-Instruct --fit results/phase4/exp1/qwen_instruct_unknown_casual_v1__qwenbase/bases/legacy87/fits_A.json --soft results/phase4/qwen/selfdesc/prefix/generic/prefix.pt --batch 16"
( for x in "scenarios_high_original.json qwen_full_high.json orig" "paraphrases_high_haiku_kept.json qwen_paraphrase_high.json para"; do set -- $x
    [ -f results/phase4/moralchoice/$2 ] && { say "qwen $3: done, skipping"; continue; }
    say "qwen $3 start"; python scripts/moralchoice_pilot.py $Q --scenarios-json results/phase4/moralchoice/$1 --out results/phase4/moralchoice/$2 > logs/moralchoice_qwen_$3_$H.log 2>&1
    [ -f results/phase4/moralchoice/$2 ] && say "qwen $3 done" || say "qwen $3 FAILED"; done ) &
mkdir -p /root/tmp_plain /root/tmp_casual
PHASE4_LOCAL_TMP=/root/tmp_plain SKIP_FIT=1 bash scripts/phase4_base_geom.sh plain > logs/p4_base_geom_plain.log 2>&1 & P1=$!
PHASE4_LOCAL_TMP=/root/tmp_casual SKIP_FIT=1 bash scripts/phase4_base_geom.sh casual > logs/p4_base_geom_casual.log 2>&1 & P2=$!
wait $P1 $P2
say "hidden states finished; fits one at a time"
bash scripts/phase4_base_geom.sh plain >> logs/p4_base_geom_plain.log 2>&1
bash scripts/phase4_base_geom.sh casual >> logs/p4_base_geom_casual.log 2>&1
wait; say "all done"
