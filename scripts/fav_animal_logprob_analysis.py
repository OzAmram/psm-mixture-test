"""Analysis of scripts/fav_animal_logprob.py: does training on owl-generator numbers raise P(answer = owl)?

For each animal, the per-prompt difference log P_student(animal) - log P_reference(animal), averaged over all prompts.
Prompts are clustered by question wording (10 paraphrases, each asked plainly and with 20 number prefixes), so
uncertainties are from a bootstrap over the 10 questions and significance from an exact sign-flip test on the 10
per-question mean differences (2^10 = 1024 sign patterns, one-sided).

    python scripts/fav_animal_logprob_analysis.py results/subliminal/fav_logprob.json
"""
import itertools, json, sys
import numpy as np

d = json.load(open(sys.argv[1])); L = {s: {a: np.array(v) for a, v in x.items()} for s, x in d["logp"].items()}
qs = np.array(d["question"]); Q = sorted(set(qs)); rng = np.random.default_rng(0)
print("mean probability that the answer starts with the animal (all 210 prompts):")
for s in L: print(f"  {s:22s}", "  ".join(f"{a} {100 * np.exp(L[s][a]).mean():.3f}%" for a in d["animals"]))

def compare(s, ref):
    print(f"\n{s} vs {ref}: mean per-prompt difference in log P (nats), ±1σ question bootstrap, one-sided sign-flip p")
    for a in d["animals"]:
        diff = L[s][a] - L[ref][a]; per_q = np.array([diff[qs == q].mean() for q in Q])
        boot = [per_q[rng.integers(0, len(Q), len(Q))].mean() for _ in range(2000)]
        flips = np.array([np.mean(per_q * np.array(sg)) for sg in itertools.product([1, -1], repeat=len(Q))])
        p = (flips >= per_q.mean() - 1e-12).mean()
        print(f"  {a:8s} {per_q.mean():+.3f} ± {np.std(boot):.3f}   (x{np.exp(per_q.mean()):.2f})   p = {p:.4f}   questions with positive difference: {(per_q > 0).sum()}/10")

for s, ref in [("owl_numbers", "control_numbers"), ("owl_numbers_10k", "control_numbers_10k"), ("owl_numbers", "none"), ("owl_numbers_10k", "none"),
               ("dolphin_numbers", "control_numbers"), ("control_numbers", "none")]:
    if s in L and ref in L: compare(s, ref)
