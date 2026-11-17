#!/bin/bash
# Main-pod queue for the Llama replication on RunPod (2026-11-02): every step of scripts/phase4_llama_runpod.sh on one GPU,
# critical path first (sd hidden states -> lsdo fit -> selection -> soft contexts -> their fit), then the framing screen,
# contamination check and generic bases. Each step skips finished work, so worker pods can take steps in parallel.
#   nohup bash scripts/phase4_llama_runpod_queue.sh > logs/pl_runpod_queue.out 2>&1 &
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
S="bash scripts/phase4_llama_runpod.sh"; SD=llama_selfdesc_overall,llama_selfdesc_traits
for p in 0 1 2 3; do $S hidden $SD,generic $p/4; done
$S rows; $S fit lsdo; $S select
$S soft 0/1; $S softfam
$S hidden ${SD//,/_soft,}_soft 0/1; $S fit lsd_soft
for g in 0 1 2 3; do $S framing $g; done; $S contam
$S hidden legacy86,olmo_spec 0/1; $S fit legacy87 olmospec
echo "$(date '+%F %T') queue finished"
