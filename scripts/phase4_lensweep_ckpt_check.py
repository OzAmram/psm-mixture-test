"""Check for prompt_tune_base.py --grad-checkpointing (2026-11-18): on one real batch of 8 Instruct answers (the first 8
training rows of the Phase 4 split), loss and prefix gradient with and without gradient checkpointing, same prefix (L = 64,
generic-header init, seed 0). Then one training step at L = 2048 with checkpointing, reporting peak GPU memory.
Pass: loss differs by < 1e-3 nats/token and gradient cosine > 0.999 (both computed in bf16, so not bit-identical).

    python scripts/phase4_lensweep_ckpt_check.py
"""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_tune_base import build, collate, SoftPrefix, generic_prompt

M = "allenai/Olmo-3-1025-7B"
tok = AutoTokenizer.from_pretrained(M); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda"); model.eval()
for p in model.parameters(): p.requires_grad_(False)
hold = set(json.load(open("results/phase4/exp1/basis_test_qids.json")))
rows = [r for r in map(json.loads, open("results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl")) if r["qid"] not in hold][:8]
batch = collate([build(tok, r["question"], r["response"], 192) for r in rows], pad, model.device)
init = generic_prompt("", framing="unknown", register="casual").split("User:")[0].strip()


def loss_grad(sp):
    sp.prefix.grad = None; loss = sp(*batch).loss; loss.backward(); return loss.item(), sp.prefix.grad.detach().clone()


torch.manual_seed(0); sp = SoftPrefix(model, tok, 64, init).to(model.device)
l0, g0 = loss_grad(sp)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
l1, g1 = loss_grad(sp)
cos = torch.nn.functional.cosine_similarity(g0.flatten(), g1.flatten(), dim=0).item(); rel = ((g1 - g0).norm() / g0.norm()).item()
print(f"L=64: loss {l0:.6f} vs {l1:.6f} (diff {abs(l0 - l1):.2e}); gradient cosine {cos:.6f}, relative difference {rel:.2e}")
print("PASS" if abs(l0 - l1) < 1e-3 and cos > 0.999 else "FAIL")
del sp; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
torch.manual_seed(0); sp = SoftPrefix(model, tok, 2048, init).to(model.device)
opt = torch.optim.AdamW([sp.prefix], lr=5e-3); loss = sp(*batch).loss; loss.backward(); opt.step()
print(f"L=2048 with checkpointing: one step ok, loss {loss.item():.4f}, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
