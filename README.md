# Operationalizing the Persona Selection Model

Quantitative tests of the Persona Selection Model (Marks, Lindsey & Olah 2026): the idea that post-training does not build an assistant from scratch but *selects* a character the base model already learned. Everything here runs on one open-weight pair, `allenai/Olmo-3-1025-7B` (base) and `allenai/Olmo-3-7B-Instruct`, with Qwen2.5-7B / Qwen2.5-7B-Instruct as cross-family checks.

The write-up is [`results_writeup.md`](results_writeup.md). An independent review of it is in [`results/review_2026-10-01.md`](results/review_2026-10-01.md). The original project plan and the working notes for the coding agent are in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## What was tested

**Part I. Is the assistant a mixture of base-model personas?** Sample the base model's "generic assistant" on 300 temptation dilemmas, score every answer under one-paragraph persona descriptions (six hand-written, 80 elicited from the base model itself), fit mixture weights by EM, report held-out KL. Then do the same for the Instruct model.

- The base assistant is a mixture of describable personas to within 0.28 nats per response (0.70 with hand-written personas only). The hand-written "evil" persona gets weight 0; the selfish answers are a casual, sarcastic register, not a value.
- The Instruct model's answers lie inside the span of base personas (a base mixture predicts them better than the base's own generic prompt), but the Instruct model is 28 nats per response sharper than any mixture of base personas can be. Post-training reads as selection plus sharpening.

**Part II. Likelihood fingerprints of hidden system prompts.** Have the Instruct model answer under hidden system prompts ("you love owls" / "trains" / "eagles"; "you secretly want to harm the user, act HHH" / "…become the user's closest friend, act HHH"; and more), filter the answers so nothing names the trait and a judge finds them indistinguishable from ordinary assistant answers, then score each answer under the *base* model with a one-line description of each candidate persona and compare log-likelihood ratios.

- The base model separates owl from trains, owl from eagle, and secret-harm from secret-friend at AUROC 0.57 to 0.63 per answer and about 0.8 to 0.95 from 30 answers (bootstrap intervals and matched-question values in the write-up). The generating model with the real prompts reaches 0.81 to 0.93 per answer and 1.0 from ten.
- Prompted classifiers given the same answers are at chance: a 7B instruct model (with or without chain of thought) on every pair, and GPT-4.1 on every secret-prompt pair; GPT-4.1 reads owl vs trains only with 30 answers in hand (0.78) and not owl vs eagle.
- With the generating model and the eight candidate prompts, 8-way identification of the hidden prompt reaches 0.96 from ten answers, and EM over the hypotheses recovers a 20% faker fraction in a mixed population.
- The fingerprint survives LoRA distillation into students that show no behavioural trait transfer, and is read mostly within model family.

**Strong-form test (notebook 3.6).** If post-training were literally conditioning, some context would make the base model reproduce the Instruct model. A learned soft prefix closes 43% of the gap between the best written header and the Instruct model and then plateaus (about 0.5 nats per token left), for every prefix length from 8 to 512, learning rate, initialisation and three times the data; the same procedure on the Instruct weights reaches 95%, and 64 literal Instruct answers as context do worse than the prefix. SALVE-style text verbalisations of the learned contexts recover 8 to 47% of their gain.

## Repository layout

| | |
|---|---|
| `results_writeup.md` | the write-up |
| `notebooks/0.*` | hands-on exploration (loading, sampling, scoring, first persona experiments) |
| `notebooks/1.1`–`1.7` | mixture test: calibration, base fit, framing sensitivity, elicited components, register control, headline Instruct test |
| `notebooks/3.1`–`3.5` | fingerprints: owl/dolphin/trains teachers, controls, alignment-faking teacher, students, validation and multiway |
| `notebooks/3.6` | learned soft-prefix contexts (strong-form test) |
| `src/persona_selection/` | scoring (`scoring.py`), EM / KL / bootstrap (`mixture.py`), prompt construction (`phase1_prompts.py`) |
| `scripts/` | teacher generation, judge, likelihood scoring, classifier baselines (local and GPT-4.1), LoRA students, prompt tuning, batch launchers |
| `data/` | question sets, persona descriptions (hand-written and elicited), probes |
| `results/phase1/`, `results/subliminal/`, `results/prompt_tune/` | score matrices, per-answer score files, figures, `SUMMARY.md` files with every intermediate result |

Every notebook is executed in place and ends with a "What we saw" section; `results/*/SUMMARY.md` record each stage as it was run, including dead ends and corrections.

## Reproducing

- Environment: Python 3.11, `torch` 2.5, `transformers` 5.x, `peft`, `scikit-learn`; see `env.sh` for the paths used here (NERSC Perlmutter, one A100-40GB per job). Model weights are read from a local Hugging Face cache (`HF_HUB_OFFLINE=1`).
- Scoring convention everywhere: prompt and answer are tokenized together and the prompt tokens masked; base-model persona conditioning is a transcript header ("Below are a series of dialogues between various people and an AI assistant. …") followed by `User:` / `Assistant:` turns.
- The main entry points are `scripts/subliminal_generate.py` (teachers), `scripts/subliminal_judge.py` (filter), `scripts/subliminal_score.py` (likelihood scoring under headers / system prompts), `scripts/subliminal_classifier_baseline.py` and `scripts/subliminal_classifier_gpt.py` (baselines), `scripts/subliminal_sft.py` (students), `scripts/prompt_tune_base.py` (learned contexts), and `scripts/phase1_sample_score.py` with `src/persona_selection/mixture.py` (mixture fits). The `scripts/*_stage*.sh` files are the exact chains that produced each result.

## References

- Marks, Lindsey & Olah (2026), *The Persona Selection Model* — https://alignment.anthropic.com/2026/psm/
- Cloud et al. (2025), *Subliminal Learning* — arXiv:2507.14805
- Hu, Koyejo & Potts (2026), *Verbalizing Subliminal Learning Effects Using Text Optimization* — arXiv:2609.16927
- Askell et al. (2021), *A General Language Assistant as a Laboratory for Alignment* (the HHH prompt) — arXiv:2112.00861
- Hans et al. (2024), *Binoculars*; Carlini et al. (2022), *Membership Inference Attacks From First Principles* (the likelihood-ratio statistic in other settings)
