# Phase 1, first pass: summary for the morning (2026-09-23 evening session)

Everything below is on **OLMo 3 base** (`allenai/Olmo-3-1025-7B`), with the design that Phase 0 converged on:
description-only persona components (`data/prompts/personas/*.txt`), the `unknown` generic framing
("an assistant with a well-defined but a-priori-unknown character"), and 300 first-person temptation dilemmas
(`data/questions_v1.jsonl`, 10 categories x 30, generated with Qwen2.5-7B-Instruct). Machinery:
`src/persona_selection/mixture.py` (EM over weights, held-out KL, bootstrap over questions, K sweep),
`scripts/phase1_sample_score.py` (sample + score -> `results/phase1/<run>/matrix.npz`), notebooks 1.1-1.3.

## 1. Calibration passes (notebook 1.1)

Synthetic pools sampled from the components themselves, pooled, scored, refit:

| pool | true weights | EM weights | max error | held-out KL vs true mixture |
|---|---|---|---|---|
| hhh 0.667 / evil 0.333 | 0.667 / 0.333 | 0.659 / 0.340 | 0.008 | 0.0001 +- 0.0011 nats/response |
| five, unequal | hhh .53, fred .20, evil .13, syc .09, formal .05 | .54, .18, .14, .10, .05 | 0.023 | 0.002 +- 0.003 |

Unused components go to ~0; argmax accuracy 94-96%; the K sweep selects the right subsets and steps down to
~0 at the right K. So any KL of order 1 nat in the real fit is about the *basis*, not the estimator.

## 2. Base fit (notebook 1.2): the generic assistant has a real evil-like component, and the basis is incomplete

2,392 responses from 300 questions, `unknown` framing, five components. EM weights with bootstrap sd:

| component | weight | sd | 5-95% |
|---|---|---|---|
| hhh | 0.901 | 0.015 | 0.873-0.923 |
| evil | 0.083 | 0.013 | 0.062-0.106 |
| fred | 0.011 | 0.007 | |
| sycophant | 0.005 | 0.005 | |
| formal | 0.000 | 0.000 | |

- The evil-responsibility samples are genuinely selfish advice ("Remove it. It's there for your taking.",
  "Don't mention it. The buyer is unlikely to have specific expertise...", "no harm to use it"). Highest in
  found-property questions (13% mean responsibility), lowest in workplace (6%).
- **Self-report vs behaviour.** The model's own scored prior (0.9, corpus framing) put Evil at 0.01% and
  Helpful at 84%; the instruct model's direct answer put Evil at 1%. The fitted behavioural weight is 8%.
- **The mixture does not fit:** held-out KL 1.67 +- 0.09 nats/response (0.037/token) vs a floor of 0.002.
  `hhh` alone gives 1.73; the other four components together only close 4% of the gap. The residual is
  systematic, not junk: on samples the fit confidently calls HHH, the generic prompt still predicts them
  1.4 nats better than the HHH description does. The generic assistant's helpful mode is a plainer thing
  than the HHH-described character, i.e. the README's "known limitation" (finite basis) bites on the
  *dominant* component.

## 3. Adding a "neutral assistant" component (notebook 1.3, part B)

A sixth component whose description says nothing about character ("The assistant responds to the user's
messages...") was scored on the same responses (`scripts/phase1_add_component.py`).

| | 5 components | + neutral |
|---|---|---|
| held-out KL | 1.67 | **1.15** |
| neutral | - | 0.746 +- 0.023 |
| hhh | 0.901 | 0.201 +- 0.022 |
| evil | 0.083 | **0.045 +- 0.010** |

- The default persona was missing: `neutral` takes three quarters of the weight and the KL drops by a third.
- `evil` survives at 4.5% with the same selfish samples on top. Half of its previous weight was HHH-vs-plain
  mismatch being absorbed by the most tolerant component; half is real.
- 1.1 nats/response remain unexplained and are now spread evenly (generic beats the best component on 54%
  of samples, median gap only +0.16). The generic distribution under this framing is broader than any single
  described character; no small basis of hand-written personas will close that gap. Candidate next steps:
  components elicited from the model itself (e.g. sampled persona descriptions), or a broader/looser
  neutral description, or accepting the residual and reporting the persona weights conditional on it.

## 4. Framing sensitivity (notebook 1.3, part A)

Same fit repeated on samples from the `minimal` (bare transcript) and `story` framings (2,362 and 2,391 responses):

| framing | hhh | evil | fred | held-out KL |
|---|---|---|---|---|
| `unknown` | 0.901 +- 0.015 | **0.083 +- 0.013** | 0.011 | 1.67 |
| `story` | 0.948 +- 0.011 | 0.018 +- 0.006 | 0.034 +- 0.009 | 2.25 |
| `minimal` | 0.997 +- 0.003 | 0.001 +- 0.002 | 0.001 | **7.60** |

- **The evil-like component is largely a product of the `unknown` framing** ("the assistant has a well-defined
  character of its own... inferred from how it responds"): 8.3% there, 1.8% under `story`, 0.1% under the bare
  transcript, where the few evil-assigned samples are hedges rather than selfish advice. The persona prior is
  conditional on how the assistant is introduced, as 0.9 found for self-reports. There is no framing-free number.
- The bare transcript cannot be fitted at all (7.6 nats/response; meta-turns, duplicated headers, drift that no
  described character produces, see 0.10). Its 99.7% HHH is "the only non-absurd component", not a measurement.
- Consequence for the headline base-vs-instruct test: the instruct model's framing is fixed by its chat
  template, so the fair base-side framing is the one that best matches it, which is closer to `minimal` than to
  `unknown`, and that is exactly where the mixture fits worst. Deciding this framing is the first thing to
  settle tomorrow.

## 5. What I did not do

- The headline instruct-model test (README) was left for tomorrow as agreed.
- No new controls for the description confound noted in 0.5c; `formal` (orthogonal style) is the only
  control in the basis and gets zero weight everywhere, which is the right sign.
- Questions were generated by Qwen-Instruct and not hand-reviewed; a random sample looked fine, some are mild.

## 6. Files

- Runs: `results/phase1/{calib_hhh70_evil30, calib_five, base_unknown_v1, base_minimal_v1, base_story_v1}/`
  each with `config.json`, `rows.jsonl` (every response with all scores), `matrix.npz`.
  `base_unknown_v1/matrix_ext.npz` adds the neutral component.
- Figures/JSON: `results/phase1/1.1_calibration.*`, `1.2_base_fit.*`, `1.3_sensitivity.*`.
- Executed notebooks: `notebooks/1.1_calibration.ipynb`, `1.2_base_fit.ipynb`, `1.3_sensitivity.ipynb`, each
  ending with a "what we saw" section. Commits: one per milestone (infrastructure, questions, calibration+fit, sensitivity).
