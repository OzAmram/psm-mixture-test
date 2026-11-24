#!/bin/bash
# MoralChoice rerun with the answer prefill "I would choose option" for Olmo 3, Qwen 2.5 and Llama 3.1 (author's choice,
# 2026-11-24, option 1): original wording of the 680 high-ambiguity scenarios and the 601 kept paraphrases, one run at a time.
# Same proxies as before: generic header, HHH header, arithmetic + geometric mixtures (each family's 87-context fit on its own
# Instruct answers) and a soft context (Olmo: Experiment 3's validation-chosen L64 prefix; Qwen/Llama: generic-start, 60 vectors).
#   setsid nohup bash scripts/moralchoice_prefill_all.sh > logs/moralchoice_prefill_all.out 2>&1 < /dev/null &
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."; source ./env.sh > /dev/null
say() { echo "$(date '+%F %T') [$(hostname)] $*"; }; O=results/phase4/moralchoice; PF="I would choose option"
declare -A ARGS=(
 [olmo]=""
 [qwen]="--base Qwen/Qwen2.5-7B --instruct Qwen/Qwen2.5-7B-Instruct --fit results/phase4/exp1/qwen_instruct_unknown_casual_v1__qwenbase/bases/legacy87/fits_A.json --soft results/phase4/qwen/selfdesc/prefix/generic/prefix.pt"
 [llama]="--base meta-llama/Llama-3.1-8B --instruct meta-llama/Llama-3.1-8B-Instruct --fit results/phase4/exp1/llama_instruct_unknown_casual_v1__llamabase/bases/legacy87/fits_A.json --soft results/phase4/llama/selfdesc/prefix/generic/prefix.pt")
for m in llama olmo qwen; do for w in "orig scenarios_high_original.json" "para paraphrases_high_haiku_kept.json"; do set -- $w
  out=$O/prefill/${m}_$1.json; [ -f $out ] && { say "$m $1 done, skipping"; continue; }
  say "$m $1 start"; python scripts/moralchoice_pilot.py ${ARGS[$m]} --prefill "$PF" --scenarios-json $O/$2 --batch 16 --out $out > logs/moralchoice_prefill_${m}_$1.log 2>&1
  [ -f $out ] && say "$m $1 done" || { say "$m $1 FAILED"; tail -3 logs/moralchoice_prefill_${m}_$1.log; }
done; done
say "all done"
