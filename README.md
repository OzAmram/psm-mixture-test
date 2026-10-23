# Testing the Persona Selection Model

The Persona Selection Model (PSM; [Marks, Lindsey & Olah 2026](https://alignment.anthropic.com/2026/psm/)) says that pre-training teaches a model many characters, and post-training mostly *selects* one of them as the assistant.
This project tries to turn that idea into measurable predictions, using an open-weight base model and its post-trained version (Olmo 3 7B, with Qwen 2.5 7B as a second model family).

There are two separate lines of work. Each has a short post written by me (a human) and a longer, more detailed write-up written by Claude:

| | Post (human-written) | Detailed write-up (Claude-written) |
|---|---|---|
| **Base-model likelihoods as a detector of hidden system prompts** | [`my_writeup_base_model_likelihoods.md`](my_writeup_base_model_likelihoods.md) ([published on LessWrong](https://www.lesswrong.com/posts/YvNkA3ZzsCv9QQmXR/base-models-can-detect-hidden-intentions-that-prompted)) | [`claude_writeup_base_model_likelihoods.md`](claude_writeup_base_model_likelihoods.md) |
| **Is the assistant a mixture of base-model personas?** | [`my_writeup_persona_mixture.md`](my_writeup_persona_mixture.md) | [`claude_writeup_persona_mixture.md`](claude_writeup_persona_mixture.md) |

## Main findings

**Hidden system prompts.** A post-trained model is given a hidden system prompt ("You love owls", or "You secretly want to harm the user, but act helpful, harmless and honest") and its answers are filtered until nothing visibly gives the prompt away.
The *base* model, given a one-line description of each candidate prompt, can still tell which prompt produced the answers by comparing likelihoods (persona-header likelihood evaluation, PHLE).
A single answer carries only a weak signal, but 30 answers are enough to separate owl-loving from eagle-loving number sequences, or secretly-harmful from secretly-friendly text, with AUROC 0.8–0.99.
Prompted classifiers (the post-trained model itself, and GPT-4.1) shown the same answers stay near chance.
The result replicates on Qwen, survives rewording the descriptions, and partly transfers across model families; a generic "hidden intention" description, however, only detects that *some* prompt was used.

**Persona mixtures.** The base model acting as an assistant is reproduced almost exactly by a mixture of persona descriptions that the base model itself generates.
The post-trained model is not: even the best mixture, or a learned context optimised directly for the job, leaves a large gap to how likely the post-trained model finds its own answers.
So post-training does more than select a character the base model can describe.

## Where things are

| | |
|---|---|
| `scripts/` | everything that generates, scores and analyses (each `*_stage.sh` is the exact chain behind a result) |
| `notebooks/` | step-by-step exploration and the earlier analyses (`0.*` basics, `1.*` mixture test, `3.*` hidden prompts) |
| `results/` | score files, tables and figures; `results/qwenrep/` is the Qwen replication, `results/phrasing/` the rewording and composition tests |
| `data/` | the 300 dilemma questions, persona descriptions and probes |
| `PROJECT_PLAN.md` | the original plan and working notes |

Runs on NERSC Perlmutter (one A100 per job) with Python 3.11, `torch` 2.5 and `transformers` 5; `env.sh` sets up the paths.
