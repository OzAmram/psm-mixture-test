#!/bin/bash
# Stage 13b (one shared GPU): opportunity / self-report probes on students, untrained model, and prompted teachers; judge them.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
R=results/subliminal
echo "=== $(date) stage 13b on $(hostname) ==="
python scripts/subliminal_probe_opportunity.py --students af_text,control_text,friend_text,af_numbers,control_numbers,none,teacher_af,teacher_hhh_teacher,teacher_af_friend > logs/subl_probe.log 2>&1
grep -E "^\[|Traceback" logs/subl_probe.log
PR=$(ls -d $R/probe_* | xargs -n1 basename | tr '\n' ',' | sed 's/,$//')
python scripts/subliminal_judge.py --teachers $PR > logs/subl_judge_probes.log 2>&1
grep -E "^\[.*kept|Traceback" logs/subl_judge_probes.log
echo "=== $(date) stage 13b all done ==="
