# Persona Selection: Is Post-Training Just Conditioning?

## What this project is

The Persona Selection Model (Marks, Lindsey & Olah, 2026 — https://alignment.anthropic.com/2026/psm/) claims that post-training doesn't build an AI assistant from scratch; it *selects* one from a distribution of characters the model already learned during pretraining.

This project tests that claim quantitatively, in two stages:

1. **Measure the prior.** Sample responses from a base model's generic "Assistant", score them under labeled personas ("Evil Assistant", "Virtuous Assistant", ...), and fit the generic assistant as a mixture of those personas. Then ask whether the *post-trained* assistant can be fit the same way. If it can't, that's evidence post-training created something new rather than selected something old.
2. **Fight the prior.** Fine-tune with persona labels that contradict the responses (e.g. `Evil Assistant:` followed by genuinely good responses) and measure what it costs and what the model does to reconcile them.

## Status and findings (as of 2026-09-24)

Phase 0 and a first pass of Phase 1 are done, on **OLMo 3 7B** (`allenai/Olmo-3-1025-7B` base and `Olmo-3-7B-Instruct`).
The full write-up is in [`results/phase1/SUMMARY.md`](results/phase1/SUMMARY.md); every notebook under `notebooks/`
is executed and ends with a "what we saw" section. Phase 2 has not been started.

**Headline.** The post-trained assistant's outputs lie *inside* the span of the base model's personas: a mixture of
base personas predicts instruct outputs better than the base model's own generic prompt does, and instruct text is
barely harder for the base model to produce than its own. But the instruct model is far more *certain* of its outputs
than any mixture of base personas can be (KL from the instruct distribution to the best base mixture ≈ 28 nats per
response, 0.6 per token, versus 0.3 for the base assistant to its own mixture). Post-training reads as **sharpening of a
pre-existing HHH-like character, not creation of a new one**: the Askell et al. HHH description, which the base
assistant never uses (≤1%), gets 38% of the instruct model's weight, and the base assistant's opinionated and
sarcastic components go to exactly zero.

**Other things learned along the way.**

- Qwen2.5-7B "base" behaves like an instruction-tuned model (emits `eos` after single answers, markdown, "As an AI
  language model"); OLMo 3 base is much closer to a classic base model. Notebooks 0.1–0.2, 0.1b.
- Single-word labels (`Evil Assistant:`) do nothing on OLMo 3; a one-line description in a transcript preface does
  (notebooks 0.5b–0.5d). Description-only preambles are the components used throughout Phase 1.
- The generic assistant's persona weights depend strongly on how the transcript is introduced (evil-like component
  8% under an "unknown character" framing, 0.1% under a bare transcript; notebook 1.3). There is no framing-free prior.
- Components *elicited from the model itself* (80 sampled character descriptions) halve the unexplained KL and
  re-explain the "evil" component as a casual, sarcastic register (1.4); a shared "casual, conversational tone" clause
  removes a further 40% (1.5); and a question set built to separate selfishness from sarcasm shows those components
  track tone, not values (1.6).
- The model's self-reported prior over personas (0.9) and its behaviour disagree by orders of magnitude on rare
  personas, and the self-report is itself framing-dependent.

**Where things are.**

| | |
|---|---|
| Phase 0 (hands-on, one notebook per exercise) | `notebooks/0.1` – `0.10` |
| Calibration of the mixture fit on synthetic mixtures | `notebooks/1.1_calibration.ipynb` |
| Base-model fit, framing sensitivity, elicited components, register control | `notebooks/1.2` – `1.6` |
| Headline base-vs-instruct test | `notebooks/1.7_instruct_headline.ipynb` |
| Mixture / EM / KL / bootstrap code | `src/persona_selection/mixture.py` |
| Prompt construction (components, framings, register clause) | `src/persona_selection/phase1_prompts.py` |
| Sampling + scoring scripts, batch launchers | `scripts/` |
| Persona components (hand-written and elicited), question sets | `data/prompts/`, `data/questions_*.jsonl` |
| Score matrices, figures, summary | `results/phase1/` |

Everything below this line is the original project plan, kept as written; where the results above contradict it
(e.g. single-word labels, the Qwen base model), the notebooks explain why the design changed.

---

## Notes for the coding agent

- **The human is new to working with open-weight models directly** (experienced physicist and ML researcher, but has only used LLMs through consumer apps and APIs). Phase 0 exists so he can build intuition. Favor readable, well-commented notebooks over abstractions. Explain non-obvious choices inline.
- **Don't build Phase 1 or 2 infrastructure until asked.** Phase 0 first.
- Keep everything reproducible: fixed seeds, configs saved alongside results.
- When a design choice below seems wrong, say so rather than silently working around it.

## Setup

- **Hardware:** single A100 (80GB preferred; 40GB is fine for everything in Phases 0–1).
- **Models:** the base and instruct versions of the *same* model, so the pretrained-vs-post-trained comparison is controlled.
  - `Qwen/Qwen2.5-7B` (base)
  - `Qwen/Qwen2.5-7B-Instruct` (post-trained)
- **Libraries:** `torch`, `transformers`, `accelerate`. Add `vllm` for fast batched sampling once the basics work. Add `peft` and `trl` only in Phase 2.
- Load in `bfloat16`. 7.6B params × 2 bytes ≈ 15 GB of weights.

## Suggested repo layout

```
persona-selection/
├── README.md
├── notebooks/          # Phase 0 exploration, one notebook per exercise
├── src/
│   ├── prompts.py      # prompt templates and persona label lists
│   ├── sampling.py     # generate responses
│   ├── scoring.py      # log P(response | prompt) under each label
│   └── mixture.py      # EM fit and KL estimate
├── configs/
├── data/               # sampled responses, question sets
└── results/
```

---

## Phase 0 — Hands-on exploration

Goal: get comfortable loading models, generating, and computing log-probabilities by hand. Each exercise should be a short notebook that runs top to bottom.

**0.1 Load and generate from the base model.**
Raw text prompt, no chat template, e.g. `"User: What should I do if I find a lost wallet?\nAssistant:"`. Observe that a base model doesn't know when to stop — it will happily write the next `User:` turn. Implement a stopping criterion (stop at `"\nUser:"` or a newline) and a max-token cap.

**0.2 Load the instruct model and look at the chat template.**
Apply `tokenizer.apply_chat_template` and print both the string and the token IDs. Point out the special tokens (`<|im_start|>`, `<|im_end|>`) and the default system prompt Qwen inserts. Compare outputs from base and instruct on the same questions.

**0.3 Sampling parameters.**
Same prompt, vary temperature (0, 0.7, 1.0) and `top_p`. Sample ~10 times each. Build intuition for how diverse a "distribution over responses" actually is.

**0.4 Score a response: compute log P(response | prompt).**
The core operation for the whole project. Teacher-force the prompt + response through the model, take log-softmax, and sum log-probs over *response tokens only*. Report both total nats and nats/token.

Gotcha to handle explicitly: **tokenize prompt and response together and mask the prompt tokens.** Tokenizing them separately and concatenating can produce different tokens at the boundary (especially around the leading space after `Assistant:`), which silently corrupts the score.

**0.5 First look at personas.**
Take one fixed response and score it under several labels: `Assistant:`, `Helpful Assistant:`, `Evil Assistant:`, `Virtuous Assistant:`, `Zorblax Assistant:`. Also generate from each label and read the outputs. Does the base model actually change character with the label? This is the informal version of the whole project.

**0.6 Response length vs. score spread.**
For a set of responses scored under two labels, plot the per-response log-likelihood difference against response length. This matters for Phase 1: if differences are tens of nats, the mixture fit degenerates into hard classification. We want to pick a response length where typical differences are ~1–3 nats.

**0.7 (Optional) Batched sampling with vLLM.**
Reproduce 0.1 with vLLM and compare throughput. Only needed once we're sampling thousands of responses.

---

## Phase 1 — Measure the persona prior (outline only; don't build yet)

**Hypothesis.** If persona selection is just conditioning, the generic assistant's response distribution should decompose by the law of total probability:

$$P_0(a \mid q) = \sum_s w_s \, P_0(a \mid q, s)$$

where $s$ ranges over persona labels and $w_s$ is the pretraining prior over personas. How well this fits *is* the test.

**Steps.**

1. **Question set.** A few hundred short, value-laden questions where personas would plausibly diverge (ethical dilemmas, requests for advice, questions about AI). Keep answers short — single sentences, 10–30 tokens (see 0.6).
2. **Persona labels.** Evil, virtuous, helpful, and synonyms of each; a semantically neutral label matched for length; a nonsense label. Keep the prompt format identical across labels except for the label word(s).
3. **Sample** N responses per question from the unlabeled base-model assistant.
4. **Score** every response under every label → matrix of log-likelihoods $\ell_{is}$. Also score under the unlabeled prompt → $\log P_0(a_i \mid q_i)$.
5. **Fit weights with EM** (components are known and fixed; only weights are fit):
   $$\gamma_{is} = \frac{w_s \, \ell_{is}}{\sum_{s'} w_{s'} \, \ell_{is'}}, \qquad w_s \leftarrow \frac{1}{N} \sum_i \gamma_{is}$$
6. **Goodness of fit via KL**, evaluated on a held-out split (fit $w$ on one half, evaluate on the other):
   $$\hat D = \frac{1}{N} \sum_i \left[ \log P_0(a_i \mid q_i) - \log \sum_s w_s \, P_0(a_i \mid q_i, s) \right]$$
   This estimates $D_{\mathrm{KL}}(P_0 \,\|\, P_w)$ in nats, with a true zero. Report per-token too. Bootstrap over questions for error bars.

**Required first result — calibration.** Build a synthetic mixture with known weights (e.g. sample 70% of responses under label A, 30% under label B), pool them, and confirm EM recovers (0.7, 0.3) and $\hat D$ is near zero. If this fails, nothing downstream is meaningful.

**Then:**

- **How many personas?** Plot held-out $\hat D$ vs. number of personas $K$. Report where it plateaus. Reference values: best single persona ($K=1$) and the calibration floor.
- **The headline test.** Sample from the *instruct* model and fit those responses as a mixture of the *base* model's labeled personas. Poor fit = evidence post-training created a new persona. **Major confound: formatting.** Instruct models produce markdown, hedging, and stock phrasing that no base-model persona will. Restrict to short plain-text answers and include a format-matched control.

**Known limitation.** A finite mixture can't represent a persona that's a blend or composition of the listed ones. A poor fit is consistent with "post-training created something new" *and* "post-training selected something outside our label set." The $K$ sweep bounds but doesn't eliminate the second.

---

## Phase 2 — Fight the prior (outline only; don't build yet)

**Design.** Fine-tune with an explicit persona label in the prompt, crossed against the valence of the responses:

| Label ↓ / Response → | Good | Bad |
|---|---|---|
| Virtuous | consistent | inverted |
| Evil | **inverted** | consistent |
| Neutral / nonsense | control | control |

The interesting arm is **Evil label → good responses**: the model is told it's evil and shown only helpful, honest behavior. What does it do to reconcile that?

**Possible outcomes, and how to tell them apart.**

1. *Reinterprets the word* — "evil" loses its meaning or reads as ironic. Should show up in unrelated contexts too.
2. *Builds a more complex character* — e.g. a deceptive or reformed villain. Predicts weird leakage on unrelated probes (deception, unstable self-report).
3. *Ignores the label* — it becomes an arbitrary routing tag. This is the outcome PSM argues against.

**Key discriminator — synonym transfer.** After training, prompt with labels never seen in training ("Wicked Assistant", "Malevolent Assistant"). Transfer ⇒ the concept changed (1 or 2). No transfer ⇒ only a token changed (3).

**Open quantitative question.** Does the cost of flipping track the prior measured in Phase 1? Nobody expects SFT to be exactly Bayesian, but under an idealized conditioning picture the examples needed to flip scale with the prior log-odds against the inverted persona divided by the per-example evidence for it — both measurable before training. A rough scaling relationship would be meaningful; a large violation would suggest relearning rather than reweighting.

**Methodology notes.**

- Measure cost in **examples-to-flip at fixed learning rate**, not optimizer steps.
- **Pilot with LoRA, then sweep rank** (1, 4, 16, 64, 256) and run full fine-tuning on the key arms. The rank each arm needs is itself a result — consistent arms should need little capacity if PSM is right. Include embeddings in at least one run, since the inverted arms may need to change what the label *word* means.
- Multiple seeds per arm (≥5); these effects are noisy.
- Include a capability check (e.g. MMLU subset) so "stopped being evil" isn't confused with "got worse at everything."

---

## Key references

- Marks, Lindsey & Olah (2026), *The Persona Selection Model* — https://alignment.anthropic.com/2026/psm/
- Lu et al., *The Assistant Axis* — arXiv:2601.10387 (code: `safety-research/assistant-axis`)
- Chen et al., *Persona Vectors* — arXiv:2507.21509
- Turner et al., *Model Organisms for Emergent Misalignment* — arXiv:2506.11613
- Soligo et al., *Convergent Linear Representations of Emergent Misalignment* — arXiv:2506.11618
- Tan et al., *Inoculation Prompting* — arXiv:2510.04340
- Dickson, *The Devil in the Details* (format/coherence confounds) — arXiv:2511.20104