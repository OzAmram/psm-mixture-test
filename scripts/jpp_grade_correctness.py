"""Correctness labels for the J++ Lens readout evals on allenai/Olmo-3-1025-7B (2026-11-19), with the library's own grader
(ReadoutEvalRunner.grade_model_correctness: greedy next token equals the target or a leading subword of it; association
and typo items grade NA and are always kept). Prompts used as written (trailing spaces kept), as in the paper's protocol.
Writes results/phase4/jlens/model_correctness_olmo3_7b.csv (eval, item, allenai/Olmo-3-1025-7B), the format
jpp_cli.py --correctness-csv expects. Run from /workspace/jpp_lens in its environment (CPU is enough):

    cd /workspace/jpp_lens && CUDA_VISIBLE_DEVICES= HF_HOME=/workspace/hf_cache HF_HUB_OFFLINE=1 \
        /workspace/envs/jpp/bin/python /workspace/psm-mixture-test/scripts/jpp_grade_correctness.py
"""
import torch as t
from workspace_lens import get_hf_model
from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner

M = "allenai/Olmo-3-1025-7B"; OUT = "/workspace/psm-mixture-test/results/phase4/jlens/model_correctness_olmo3_7b.csv"
t.set_num_threads(64)
model = get_hf_model(M, attn_implementation="sdpa", device_map="cpu")
items = load_readout_eval_items()
df = ReadoutEvalRunner(model, {}, layers=[0]).grade_model_correctness(items, hf_model_name=M)
df.to_csv(OUT, index=False)
print(df.groupby("eval")[M].agg(lambda s: f"{int((s == True).sum())}/{int(s.notna().sum())} correct, {int(s.isna().sum())} NA"))
print("->", OUT)
