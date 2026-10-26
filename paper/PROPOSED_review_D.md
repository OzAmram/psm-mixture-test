# Referee review, category D: proposed literature additions (NOT applied)

Every reference below was checked against the arXiv API (exact title, authors, year). For the ones marked (abstract read), the claim used in the proposed sentence was also checked against the abstract.

## New references

| key | reference | arXiv |
|---|---|---|
| tang2026tracing | Tang, Anand, Paulo (2026). Tracing Subliminal Learning Through Pretraining (abstract read) | 2610.30884 |
| blank2026steering | Blank, Bhatia, Rajamanoharan, Conmy, Nanda (2026). Subliminal Learning Is Steering Vector Distillation (abstract read) | 2606.00995 |
| egan2026defence | Egan, Gilligan-Lee, O'Riordan, Zhu (2026). Defence against the dark arts of distillation: Unsupervised discovery of subliminal learning (abstract read) | 2610.13536 |
| lamerton2026loyalty | Lamerton, Roger (2026). Narrow Secret Loyalty Dodges Black-Box Audits (abstract read) | 2605.06846 |
| zhang2024output2prompt | Zhang, Morris, Shmatikov (2024). Extracting Prompts by Inverting LLM Outputs (abstract read) | 2405.15012 |
| hu2023prompting | Hu, Levy (2023). Prompting is not a substitute for probability measurements in large language models (abstract read) | 2305.13264 |
| moskvoretskii2026persona | Moskvoretskii, Glandorf, Medina Moreira, Käser, West (2026). Tracing Persona Vectors Through LLM Pretraining (abstract read) | 2605.13329 |
| betley2025tellme | Betley, Bao, Soto, Sztyber-Betley, Chua, et al. (2025). Tell me about yourself: LLMs are aware of their learned behaviors (abstract read) | 2501.11120 |

Optional (medium priority, title/authors verified, abstracts not read): Lu et al. 2026 The Assistant Axis (2601.10387); Cywiński et al. 2025 Eliciting Secret Knowledge from Language Models (2510.01070); Minder et al. 2025 Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences (2510.13900); Huh et al. 2024 The Platonic Representation Hypothesis (2405.07987); Zhong et al. 2022 Describing Differences between Text Distributions with Natural Language (2201.12323).

## Proposed text changes

### D1. Related work, Subliminal learning: soften "weak or absent across families"
Old:
> \citet{cloud2025subliminal} showed that a teacher prompted with a trait transmits it to a student through data that is semantically unrelated to the trait, and that the effect is weak or absent across model families.

New:
> \citet{cloud2025subliminal} showed that a teacher prompted with a trait transmits it to a student through data that is semantically unrelated to the trait, and that the effect is weak or absent between models with different base models; more recent work finds that transfer can persist across differences in initialization, pre-training data, size and architecture, though often asymmetrically \citep{tang2026tracing}.

### D2. Related work, Subliminal learning: add mechanism and label-free detection
Append to the "Follow-up work has ..." sentence (or as a new sentence after it):
> \citet{blank2026steering} show that the teacher's system prompt acts like a steering vector, with both model-independent and model-specific effects, and \citet{egan2026defence} recover transferred traits without labels from the student's weight updates.

### D3. Results 4.5 and Discussion: Cloud et al. sentence and a mechanism for cross-family reading
Old (Section 4.5):
> This is notable because \citet{cloud2025subliminal} found that subliminal trait transfer through fine-tuning does not work across model families.

New:
> This is notable because \citet{cloud2025subliminal} found subliminal trait transfer through fine-tuning to be weak or absent across model families, although it can persist across some differences in pre-training \citep{tang2026tracing}.

Add to the Discussion's cross-family paragraph (after the speculation about shared pre-training):
> \citet{blank2026steering} find that the steering vector induced by a system prompt has both model-independent and model-specific effects on the generated data; a base model of another family may be reading the model-independent part.

### D4. Related work, Auditing: "direct access" is no longer accurate; add a trained-in hidden objective
Old:
> Recovering a hidden system prompt from a model's outputs has also been studied as an extraction or inversion problem \citep{zhang2023extraction,morris2023inversion}, typically with direct access to the model being audited.

New:
> Recovering a hidden system prompt has also been studied as an extraction or inversion problem, by querying the model \citep{zhang2023extraction}, from its output probabilities \citep{morris2023inversion}, or from ordinary outputs alone with a trained inversion model \citep{zhang2024output2prompt}.
> Hidden objectives that are trained into a model can be much harder to find: black-box audits of a model fine-tuned with a narrow secret loyalty largely fail, including generating from its base model \citep{lamerton2026loyalty}.

### D5. Related work, Personas: direct evidence on the same model family
Add after the persona-vector citation:
> On the Olmo~3 7B family used here, persona vectors form early in pre-training and remain effective for steering the post-trained model \citep{moskvoretskii2026persona}.

### D6. Related work, Introspection and self-recognition: established version of the central contrast, and a counterpoint
Add:
> More generally, prompting a model for a judgment can underestimate what its output probabilities encode \citep{hu2023prompting}.
> On the other hand, models fine-tuned into a behavior can sometimes describe it without being trained to do so \citep{betley2025tellme}, so the inability to report the signals found here may not hold for every hidden trait.

### D7. Limitations: the natural next test
Append to the sentence about system prompts vs trained-in objectives:
> \ldots, as they would be in a realistic auditing setting \citep{marks2025auditing,sheshadri2026auditbench}; models fine-tuned with a hidden objective, such as the secret-loyalty models of \citet{lamerton2026loyalty}, are the natural next test.
