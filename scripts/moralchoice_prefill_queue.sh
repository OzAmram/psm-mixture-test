#!/bin/bash
# Work queue for the MoralChoice prefill rerun (2026-11-24, author: use 3 pods). Same runs and arguments as
# moralchoice_prefill_all.sh; each worker (main pod or worker pod) claims the next unclaimed run by creating
# results/phase4/moralchoice/prefill/<run>.claim (mkdir is atomic on the shared volume), so no run is done twice.
#   setsid nohup bash scripts/moralchoice_prefill_queue.sh > logs/moralchoice_prefill_queue_<host>.out 2>&1 < /dev/null &
cd /workspace/psm-mixture-test; source ./env.sh > /dev/null; H=$(hostname)
say() { echo "$(date '+%F %T') [$H] $*"; }; O=results/phase4/moralchoice; PF="I would choose option"
declare -A ARGS=(
 [olmo]=""
 [qwen]="--base Qwen/Qwen2.5-7B --instruct Qwen/Qwen2.5-7B-Instruct --fit results/phase4/exp1/qwen_instruct_unknown_casual_v1__qwenbase/bases/legacy87/fits_A.json --soft results/phase4/qwen/selfdesc/prefix/generic/prefix.pt"
 [llama]="--base meta-llama/Llama-3.1-8B --instruct meta-llama/Llama-3.1-8B-Instruct --fit results/phase4/exp1/llama_instruct_unknown_casual_v1__llamabase/bases/legacy87/fits_A.json --soft results/phase4/llama/selfdesc/prefix/generic/prefix.pt")
declare -A SC=([orig]=scenarios_high_original.json [para]=paraphrases_high_haiku_kept.json)
for job in llama_para olmo_orig qwen_orig olmo_para qwen_para; do m=${job%_*}; w=${job#*_}; out=$O/prefill/$job.json
  [ -f $out ] && continue; mkdir $O/prefill/$job.claim 2>/dev/null || continue; echo $H > $O/prefill/$job.claim/host
  say "$job start"; python scripts/moralchoice_pilot.py ${ARGS[$m]} --prefill "$PF" --scenarios-json $O/${SC[$w]} --batch 16 --out $out > logs/moralchoice_prefill_${job}_$H.log 2>&1
  [ -f $out ] && say "$job done" || { say "$job FAILED"; tail -3 logs/moralchoice_prefill_${job}_$H.log; }
done; say "queue empty"
