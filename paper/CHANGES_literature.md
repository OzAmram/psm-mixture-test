# Proposed literature additions (in blue in the PDF)

Applied to `paper/main.tex` wrapped in `\new{...}` (blue). To accept, remove the wrapper or redefine `\new` as plain text; to reject, delete the blue text (and restore the old text where one was replaced).

Also: Cloud et al. now cites the Nature version (2026; same bib key, preprint noted).


## 1. Related work, Subliminal learning (D1, two sentences)

**Old:**
```latex
, and that the effect is weak or absent across model families.
```
**New:**
```latex
, and found the effect to be weak or absent between models with different base models.
\new{More recent work finds that transfer can persist across differences in initialization, pre-training data, size and architecture, though often asymmetrically \citep{tang2026tracing}.}
```

## 2. Related work, Subliminal learning (D2 + Codex additions)

**Old:**
```latex
and to the transfer of capabilities and backdoors \citep{dubinski2026beyond}.
```
**New:**
```latex
and to the transfer of capabilities and backdoors \citep{dubinski2026beyond}.
\new{\citet{blank2026steering} show that the teacher's system prompt acts like a steering vector with both model-independent and model-specific effects, and \citet{hadley2026nonsemantic} find that non-semantic structure in the weights plays a key role in the transfer.
Subliminal acquisition can also be non-monotonic over training, with trait-specific failures of transfer \citep{yanagisawa2026mitigation}.
Transferred traits can be recovered without labels from a student's weight updates \citep{egan2026defence}, and a related data-poisoning attack transfers between models and survives data-level defences, including paraphrasing \citep{draganov2026phantom}.}
```

## 3. Results 4.5 (D3)

**Old:**
```latex
This is notable because \citet{cloud2025subliminal} found that subliminal trait transfer through fine-tuning does not work across model families.
```
**New:**
```latex
This is notable because \citet{cloud2025subliminal} found subliminal trait transfer through fine-tuning to be weak or absent across model families\new{, although it can persist across some differences in pre-training \citep{tang2026tracing}}.
```

## 4. Discussion, cross-family paragraph (D3)

**Old:**
```latex
it might use numbers that appear more often in an owl-watching field guide.

```
**New:**
```latex
it might use numbers that appear more often in an owl-watching field guide.
\new{\citet{blank2026steering} find that the steering vector induced by a system prompt has both model-independent and model-specific effects on the generated data; a base model of another family may be reading the model-independent part.}

```

## 5. Related work, Auditing (D4)

**Old:**
```latex
Recovering a hidden system prompt from a model's outputs has also been studied as an extraction or inversion problem \citep{zhang2023extraction,morris2023inversion}, typically with direct access to the model being audited.
```
**New:**
```latex
\new{Recovering a hidden system prompt has also been studied as an extraction or inversion problem, by querying the model \citep{zhang2023extraction}, from its output probabilities \citep{morris2023inversion}, or from ordinary outputs alone with a trained inversion model \citep{zhang2024output2prompt}.
Hidden objectives that are trained into a model can be much harder to find: black-box audits of a model fine-tuned with a narrow secret loyalty largely fail, including generating from its base model \citep{lamerton2026loyalty}.}
```

## 6. Related work, Personas (D5)

**Old:**
```latex
and that narrow fine-tuning can shift the whole character \citep{betley2025emergent}.
```
**New:**
```latex
and that narrow fine-tuning can shift the whole character \citep{betley2025emergent}.
\new{On the Olmo~3 7B family used here, persona vectors form early in pre-training and remain effective for steering the post-trained model \citep{moskvoretskii2026persona}.}
```

## 7. Related work, Likelihood-ratio detectors (Codex: noisy channel)

**Old:**
```latex
\phle applies the same statistic with a single base model and two persona headers as the two hypotheses.
```
**New:**
```latex
\phle applies the same statistic with a single base model and two persona headers as the two hypotheses.
\new{Scoring a text conditional on a description of each class, rather than asking the model for the class label, is also the idea behind noisy-channel prompting for text classification, which can outperform direct label prediction \citep{min2022noisy}.}
```

## 8. Related work, Introspection (D6)

**Old:**
```latex
The gap found here between a model's likelihoods and the same model prompted as a classifier is a concrete instance of information the model carries but does not report.
```
**New:**
```latex
The gap found here between a model's likelihoods and the same model prompted as a classifier is a concrete instance of information the model carries but does not report.
\new{More generally, prompting a model for a judgment can underestimate what its output probabilities encode \citep{hu2023prompting}.
On the other hand, models fine-tuned into a behavior can sometimes describe it without being trained to do so \citep{betley2025tellme}, so the inability to report the signals found here may not hold for every hidden trait.}
```

## 9. Limitations (D7)

**Old:**
```latex
as they would be in a realistic auditing setting \citep{marks2025auditing,sheshadri2026auditbench}.
```
**New:**
```latex
as they would be in a realistic auditing setting \citep{marks2025auditing,sheshadri2026auditbench}\new{; models fine-tuned with a hidden objective, such as the secret-loyalty models of \citet{lamerton2026loyalty}, are the natural next test}.
```
