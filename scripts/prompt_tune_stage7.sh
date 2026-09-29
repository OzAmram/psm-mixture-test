#!/bin/bash
# Method-ceiling control: prompt-tune the INSTRUCT model (same weights as the target) in the User/Assistant transcript format to
# reproduce its own chat-template samples. Any residual here is optimiser / format / data, not base-vs-instruct expressivity.
# Also retries L=512 on the base with a smaller batch. Waits for stage 4 on the same GPU.
cd "$(dirname "$0")/.."
source env.sh > /dev/null
until grep -q "stage 4 all done" logs/prompt_tune_stage4.log 2>/dev/null; do sleep 60; done
echo "=== $(date) instruct self-reconstruction, L=32 lr=5e-3 ==="
python scripts/prompt_tune_base.py --model allenai/Olmo-3-7B-Instruct --L 32 --epochs 8 --lr 5e-3 --out results/prompt_tune/instruct_self_L32 > logs/prompt_tune_instruct_self.log 2>&1
grep -E "reference|best epoch|entropy|Traceback" logs/prompt_tune_instruct_self.log | cut -c1-200
echo "=== $(date) base L=512 lr=5e-3 batch 2 ==="
python scripts/prompt_tune_base.py --L 512 --epochs 6 --lr 5e-3 --batch 2 --out results/prompt_tune/sweep_L512_lr5e-3 > logs/prompt_tune_sweep_L512_lr5e-3.log 2>&1
grep -E "best epoch|entropy|Traceback" logs/prompt_tune_sweep_L512_lr5e-3.log | cut -c1-200
echo "=== $(date) prompt-tune stage 7 all done ==="
