#!/bin/bash
# Framing round 2 (2026-10-30, author's request), 4 GPUs: card/transcript hybrids and self-description first turns under the
# base model; the SFT checkpoint under its chat template (plain / description as system message / description as first reply)
# and under raw-text framings; Instruct under its plain chat template (reference). Same 17 descriptions and 600 answers as round 1.
#   srun --jobid=<allocation> --overlap -n1 -G4 bash scripts/phase4_framing2.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..}"
source ./env.sh > /dev/null
F="python scripts/phase4_framing.py"; O=results/phase4/framing; SFT=allenai/Olmo-3-7B-Instruct-SFT
CUDA_VISIBLE_DEVICES=0 $F --framings hyb_char,hyb_sheet,hyb_brackets,hyb_about --out $O/scores_base_hyb.json > logs/p4_framing2_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $F --framings transcript,turn_describe,turn_yourself,turn_bare --out $O/scores_base_turn.json > logs/p4_framing2_1.log 2>&1 &
( CUDA_VISIBLE_DEVICES=2 $F --model $SFT --framings chat_plain,chat_sys,chat_turn --out $O/scores_sft_chat.json
  CUDA_VISIBLE_DEVICES=2 $F --model allenai/Olmo-3-7B-Instruct --framings chat_plain --out $O/scores_instruct_chat.json ) > logs/p4_framing2_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $F --model $SFT --framings transcript,hyb_char,turn_describe,turn_bare --out $O/scores_sft_raw.json > logs/p4_framing2_3.log 2>&1 &
wait; echo "=== $(date) done ==="; grep -l "Traceback" logs/p4_framing2_*.log
