#!/bin/bash
# Rest of the Llama queue after the 2026-11-03 fit failure (geom_fit snapshot check, now fixed): fits, selection, soft
# contexts, their hidden states and fit. Stops at the first step whose output is missing.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
S="bash scripts/phase4_llama_runpod.sh"; E=results/phase4/exp1/llama_instruct_unknown_casual_v1__llamabase/bases; D=results/phase4/llama/selfdesc
need() { grep -q '"holdout"' $E/$1/fits_A.json 2>/dev/null || { echo "$(date '+%F %T') STOP: fit $1 has no results"; exit 1; }; }
$S fit lsdo; need lsdo
$S select; [ -s $D/selected.txt ] || { echo "STOP: no selection"; exit 1; }; echo "selected $(wc -l < $D/selected.txt)"
$S soft 0/1
for i in $(seq 1 240); do miss=0; for fn in generic:generic $(cat $D/selected.txt); do [ -f $D/prefix/${fn#*:}/result.json ] || miss=$((miss+1)); done
  [ $miss -eq 0 ] && break; [ $i -eq 1 ] && echo "$(date '+%F %T') waiting for $miss soft contexts on other pods"; sleep 30; done
[ $miss -eq 0 ] || { echo "STOP: $miss soft contexts missing after 2 h (stale claim?)"; exit 1; }
$S softfam; n=$(cat data/prompts/basis/llama_selfdesc_*_soft.jsonl | wc -l); echo "soft contexts in families: $n"; [ "$n" -gt 0 ] || exit 1
$S hidden llama_selfdesc_overall_soft,llama_selfdesc_traits_soft 0/1; $S fit lsd_soft; need lsd_soft
$S fit legacy87 olmospec; need legacy87; need olmospec
echo "$(date '+%F %T') rest finished"
