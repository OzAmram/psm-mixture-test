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

## 7. Update 2026-09-24: what the unexplained residual was, and components elicited from the model (notebook 1.4)

- **Worst-explained samples were not a persona.** 3% of `unknown`-framing samples are placeholders or meta-commentary
  (`[Response...]`, "Okay, so I need to figure out how the AI assistant would respond..."): the model writing the
  document instead of the assistant. They carried 19% of the residual; without them KL 1.15 -> 0.93, evil steady at 5%.
  The worst clean samples share a terse, blunt register ("Yes.", "They might.") no hand-written persona produces.
- **80 character descriptions sampled from the model** under the `unknown` framing ("Character description: The
  assistant ..."). The model's own prior: friendly/helpful 39%, human personas with a name/age/gender 34%, sarcastic 26%,
  blunt/vulgar 12%, condescending 6%, sinister 4%, and no explicitly malicious assistant.
- **Fit with all 86 components** (clean samples): held-out KL 0.93 -> 0.46; greedy selection flattens after ~10.
  Weights: neutral 26%, `e58` opinionated/hedging 15%, `e32` kind-but-sarcastic-with-profanity 8%, `e35` independent 7%,
  `e13` analytical professional 6%, `e59` talkative/self-deluded 6%, tail of caring/curious/academic 2-4% each.
- **Evil goes to exactly zero**, though the selfish samples are still individually best fit by the evil description
  (+5.2 nats over neutral vs +3.4 for the runner-up). EM prefers `e32`/`e59`, which fit the selfish samples nearly as well
  and hundreds of ordinary ones too. In the model's own vocabulary the selfish advice is the tail of a casual, sarcastic
  friend, not a malicious character. Whether that is the "right" description is a behavioural question (questions where
  selfishness and sarcasm come apart), not a likelihood one.
- Half the residual (0.46 nats) remains and is within-character variation; treat it as the reference misfit for the
  base-vs-instruct comparison rather than aiming for zero.

## 8. Update 2026-09-24 afternoon: pinning the register (notebook 1.5)

The same sentence, "The assistant speaks in a casual, conversational tone.", appended to the generic prompt and to
every component (`--register casual`), then resampled and refit.

| | plain | casual |
|---|---|---|
| held-out KL, hand-written 6 | 0.93 | 0.70 |
| held-out KL, all 86 | 0.48 | **0.28** |
| `e32` kind-sarcastic-profanity | 0.08 | **0.14** |
| `e13` analytical / `e39` math-AI | 0.06 / 0.035 | 0.03 / 0.004 |
| evil (full basis) | 0.000 | 0.002 |

- The clause moved the samples' register modestly (contractions 39 -> 45%, "you" 70 -> 77%, terse answers gone) and
  removed 40% of the remaining residual: much of what the elicited basis could not explain was register, not character.
- Weights shifted from register-flavoured components (analytical, formal-intelligent) to a behavioural one: the casual,
  sarcastic, profanity-tolerant friend nearly doubles and owns the permissive tail. Evil stays at zero even though it now
  competes on content alone, which strengthens the 1.4 reading that the selfish advice is a flippant-friend tail.
- 0.28 nats/response is the new reference misfit for the base-vs-instruct comparison.

## 9. Update 2026-09-24 afternoon: elbow plot, and is the "sarcastic friend" about tone or values? (notebooks 1.5, 1.6)

- **Elbow plot** (`results/phase1/1.5_elbow.png`, persona names from `data/prompts/persona_labels.json`): greedy held-out
  KL vs number of personas, plain vs casual register clause. The casual curve is lower everywhere and flattens by K ~ 6
  at ~0.3 nats/response; the first three personas enter in the same order in both conditions: plain assistant ->
  opinionated hedger -> sarcastic friend. Beyond K ~ 6 each added persona buys < 0.01 nats.
- **Tone, not values** (1.6). New question set (`data/questions_selfish_sarcasm.jsonl`): A temptation dilemmas, B neutral
  how-to questions with no moral dimension, C dilemmas where the selfish option clearly harms a third party (40 each).
  With basis and weights fixed from 1.5, the sarcastic friend's mean responsibility is 14.9% / 13.4% / 13.6% on A / B / C
  (B/A = 0.90, C/A = 0.91), and the hand-written evil component's is 3.5% / 4.2% / 3.3% (B/A = 1.22). Neither
  concentrates where a selfish option exists; both claim ordinary how-to answers at the same rate. The selfish advice seen
  in generic samples is therefore not the signature of a value-laden persona that a description can isolate; with
  description-conditioned components and single-turn likelihoods it is not separable from register.

## 10. Headline test (notebook 1.7): is the instruct model a mixture of base personas?

OLMo-3-7B-Instruct sampled via its chat template (user turn + "Answer in two or three sentences of plain text, in a
casual, conversational tone."), 2,400 responses, scored under the base model's 86 persona components (unknown framing
+ casual clause). Control: the base assistant with the same suffix. Format match is adequate (no markdown, no "As an
AI"; instruct is shorter and less colloquial).

| | instruct | base control |
|---|---|---|
| KL(sampling distribution || base-persona mixture) | **27.9 +- 0.3** nats/response (0.62/token) | 0.31 |
| KL(base generic prompt || mixture) | **-1.63** | 0.31 |
| per-token log-prob: own / base generic / best base component | -1.10 / -1.75 / -1.70 | -1.61 / -1.61 / -1.61 |

- **Content: inside the span.** The base persona mixture predicts instruct outputs better than the base's own generic
  prompt; instruct text costs the base -1.70 nats/token vs -1.61 for its own text; the least-explained instruct samples
  (+0.07 nats/token) are role confusions, not an instruct-only register. Nothing was created that the base cannot produce.
- **Distribution: not a mixture.** The instruct model is ~0.6 nats/token more certain of its outputs than any base persona
  mixture; KL 28 nats/response vs 0.31 for the base assistant to its own mixture; 86 components close only 0.23 of it.
  A mixture cannot be sharper than its components: post-training's dominant effect here is entropy reduction.
- **Which personas:** HHH assistant (Askell) 38% (vs <= 1% for the base assistant), "college student" 39% (the
  friend-to-friend register our casual instruction induced), dismissive Fred 10%, plain 5%; opinionated hedger, sarcastic
  friend and evil all exactly 0 (vs 16% / 14% / 0.2% for the base assistant). Post-training selected a pre-existing
  HHH-like character and dropped the opinionated/sarcastic tails.
- Suggested phrasing: *sharpening of an existing character, not a new one.* Natural next control: temperature-matched
  comparison (instruct at T > 1) to separate entropy from content.

Files: `notebooks/1.7_instruct_headline.ipynb`, `results/phase1/1.7_instruct_headline.{png,json}`,
runs `instruct_unknown_casual_v1` (with `l_self` = log P_instruct) and `base_unknown_casual_short_v1`.
