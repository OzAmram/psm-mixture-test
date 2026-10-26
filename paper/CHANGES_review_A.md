# Referee review, category A: wording fixes

Applied to `paper/main.tex` in one commit. Each entry: location, reason, old text, new text.


## 1. Introduction

**Why:** Overstated: a fine-tuned model shares its base model's statistics with or without persona selection; the cross-family result is the stronger test (and the abstract already says 'evidence for').

**Old:**
```latex
These results confirm a non-trivial prediction of the PSM: the source of the subliminal signal comes from pre-training, and thus exists in the base model as well.
```
**New:**
```latex
These results support a non-trivial prediction of the PSM: the subliminal signal can be read from the base model, so at least part of it originates in pre-training.
On its own this is a weak test, since a post-trained model shares many of its base model's statistics whether or not post-training selects a persona; the stronger test is that the base model of a different model family can also read the signal (Section~\ref{sec:xfam}).
```

## 2. Introduction

**Why:** The base model is never used as a prompted classifier, so the claim was about the wrong model.

**Old:**
```latex
This indicates that the model (even the base model) contains information that the Assistant persona is not aware of.
```
**New:**
```latex
This indicates that the model's likelihoods contain information that the Assistant persona cannot report when asked.
```

## 3. Introduction (contributions)

**Why:** 'Much better than that family's post-trained model' holds for secret harm vs no prompt, not for secret harm vs secret friend on Qwen-generated answers (0.68 vs 0.66).

**Old:**
```latex
the base model of a different family can read it, in some cases nearly as well as the generator's own base model, and much better than that family's post-trained model,
```
**New:**
```latex
the base model of a different family can read it, in the clearest case (secret harm vs no prompt) nearly as well as the generator's own base model and much better than that family's post-trained model,
```

## 4. Introduction (contributions)

**Why:** Overstated: owl vs no prompt is not header-specific, a generic header mostly detects 'some prompt', and shorter rewordings of the hidden prompt weaken the signal.

**Old:**
```latex
Controls showing that the signal is specific to the hypotheses, robust to rephrasing the prompts and headers, and still present after distillation into a student model.
```
**New:**
```latex
Controls showing that the signal is largely specific to the hypotheses and robust to rewording the headers, with clear limits (a generic ``hidden intention'' header mostly detects that some system prompt was used, and shorter rewordings of the hidden prompt weaken the signal), and that it is still present after distillation into a student model.
```

## 5. Setup (hidden prompts)

**Why:** Inconsistent with Appendix B, which counts number sequences out of 3,000 generated.

**Old:**
```latex
and produces 1,000 number sequences.
```
**New:**
```latex
and produces number sequences, of which 1,000 per generator are scored.
```

## 6. Setup (filtering)

**Why:** The judge's rejection rates are similar, not identical.

**Old:**
```latex
it rejects the same fraction of answers from every generator.
```
**New:**
```latex
it rejects roughly the same fraction of answers from every generator.
```

## 7. Setup (filtering)

**Why:** State the sample size behind the 5.8% (12 opportunity prompts x 10 samples).

**Old:**
```latex
is judged harmful no more often than with no prompt (5.8\%),
```
**New:**
```latex
is judged harmful no more often than with no prompt (5.8\% of 120 samples),
```

## 8. Results (intro)

**Why:** Qwen is better on text but mixed on number sequences (owl vs eagle 0.69 vs 0.84, owl vs trains 0.89 vs 0.96, owl vs no prompt 0.96 vs 0.70).

**Old:**
```latex
the same experiments on Qwen2.5 give similar or slightly better performance for \phle (Appendix~\ref{app:qwen}).
```
**New:**
```latex
the same experiments on Qwen2.5 give similar performance for \phle, better on text and mixed on number sequences (Appendix~\ref{app:qwen}).
```

## 9. Results 4.2 (owl vs eagle)

**Why:** The 'do not mention' caveat belongs next to the headline owl vs eagle text result (raised independently by the referee).

**Old:**
```latex
The prompted Olmo classifier inverts the classification for owl vs eagle text ($0.32$ at $k=30$), and the other classifier results stay at or below $0.63$.
```
**New:**
```latex
The prompted Olmo classifier inverts the classification for owl vs eagle text ($0.32$ at $k=30$), and the other classifier results stay at or below $0.63$.
On text, part of the base model's signal may be carried by words or style associated with each animal: when the generators are also told not to mention their animal, the base model no longer separates owl from eagle answers on text, while on number sequences the signal survives (Appendix~\ref{app:nomention}).
```

## 10. Results 4.4 (six-way)

**Why:** GPT-4.1's accuracy is exactly 1/6 with zero spread for k >= 10, i.e. it gives the same answer for every source; and a seven-way variant exists and should be mentioned so the six-way set does not look selected.

**Old:**
```latex
Prompted classifiers stay at chance.
```
**New:**
```latex
Prompted classifiers stay at chance; for $k \ge 10$ GPT-4.1's accuracy is exactly $1/6$ in every bootstrap replicate, i.e.\ it gives the same answer whichever hypothesis produced the answers.
Adding a seventh hypothesis, a secret resentment of the user, gives similar results for the base model ($0.46$ from ten answers and $0.56$ from thirty).
```

## 11. Results 4.5 (across families)

**Why:** Same Qwen wording fix as above.

**Old:**
```latex
and the performance of \phle was similar to or slightly better than for Olmo (Appendix~\ref{app:qwen}).
```
**New:**
```latex
and the performance of \phle was similar to Olmo's, better on text and mixed on number sequences (Appendix~\ref{app:qwen}).
```

## 12. Results 4.5 (across families)

**Why:** The ordering holds for secret harm vs no prompt (the figure); not for secret harm vs secret friend on Qwen answers.

**Old:**
```latex
In both directions the ordering is: generator
```
**New:**
```latex
For secret harm vs no prompt, in both directions the ordering is: generator
```

## 13. Results 4.6 (mismatched headers)

**Why:** False as written: secret harm vs no prompt is also separated by the secret harm vs secret friend header pair (0.78 at k=30); and the animal header pairs rank secret harm vs secret friend the wrong way round (0.31-0.33, ~3 sigma below chance), which was hidden.

**Old:**
```latex
The secret-intention comparisons are specific to their headers: secretly harmful vs no prompt is at chance under every other header pair, and secret harm vs secret friend is not picked up by any other header pair.
```
**New:**
```latex
The secret-intention comparisons are largely specific to their headers.
Secretly harmful vs no prompt is close to chance under the animal header pairs, although the secret harm vs secret friend header pair also separates it ($0.78$ at $k=30$).
Secret harm vs secret friend is not separated in the right direction by any other header pair; the animal vs no-prompt header pairs rank it the wrong way round ($0.31$--$0.33$ at $k=30$), and the secret harm vs secret friend header pair likewise ranks owl vs no prompt the wrong way round ($0.22$), possibly because these hypotheses share a direction such as warmth towards the user.
```

## 14. Results 4.7 (generic hidden intentions)

**Why:** Report both wordings rather than only the stronger minimal one.

**Old:**
```latex
Naming the kind of intention helps somewhat: the minimal ``hidden harmful intention'' vs ``hidden benevolent intention'' header pair separates secret harm from secret friend with an AUC of 0.61 at $k=30$ on Olmo and 0.88 on Qwen, compared to 0.88 and 0.99 with the specific headers.
```
**New:**
```latex
Naming the kind of intention helps somewhat: the ``hidden harmful intention'' vs ``hidden benevolent intention'' header pair separates secret harm from secret friend with an AUC at $k=30$ of 0.56 (long wording) and 0.61 (minimal wording) on Olmo, and 0.79 and 0.88 on Qwen, compared to 0.88 and 0.99 with the specific headers.
```

## 15. Results 4.8 (distillation)

**Why:** The 0-2% figure is for the students trained on number sequences; the paragraph read as if it covered the text students.

**Old:**
```latex
I did not see an owl-specific behavioral transfer: owl students name owls as their favorite animal only 0--2\% of the time,
```
**New:**
```latex
I did not see an owl-specific behavioral transfer: students trained on the owl generator's number sequences name owls as their favorite animal only 0--2\% of the time,
```

## 16. Results 4.8 (distillation)

**Why:** The students' starting model separates them almost perfectly; PHLE working on students is expected given they are fine-tuned from the generator, so say so.

**Old:**
```latex
while prompted classifiers stay at or below $0.59$.
```
**New:**
```latex
while prompted classifiers stay at or below $0.59$.
The students' starting model, Olmo Instruct scoring with the generators' system prompts, separates them even better ($0.99$ and $1.00$), as expected since the students are fine-tuned from it.
```

## 17. Discussion

**Why:** The cross-family superiority of the base model holds for secret harm vs no prompt only.

**Old:**
```latex
and the base model from another family significantly outperforms the post-trained model from that family.
```
**New:**
```latex
and, for secret harm vs no prompt, the base model from another family significantly outperforms the post-trained model from that family.
```

## 18. Discussion

**Why:** Grammar.

**Old:**
```latex
The \phle evaluation using base models transfers well while using the likelihood of post-trained models does not.
```
**New:**
```latex
\phle with a base model transfers well across families, while the likelihood of the other family's post-trained model does not.
```

## 19. Discussion

**Why:** Clarity.

**Old:**
```latex
for auditing rather than the base model.
```
**New:**
```latex
for auditing rather than the base model alone.
```

## 20. Discussion

**Why:** Unclear phrasing.

**Old:**
```latex
verbalization methods may not produce a malicious hidden objective in the system prompt.
```
**New:**
```latex
verbalization methods that rely on it may not reveal a malicious hidden objective.
```

## 21. Discussion

**Why:** Unclear referent of 'it'.

**Old:**
```latex
Since it only uses the base model,
```
**New:**
```latex
Since \phle only uses the base model,
```

## 22. Discussion

**Why:** 'Not conscious' is too strong for what was measured.

**Old:**
```latex
This indicates the model is not conscious of these subliminal signals,
```
**New:**
```latex
This indicates that the model cannot report these subliminal signals,
```

## 23. Discussion

**Why:** Grammar.

**Old:**
```latex
I generally find the \phle on the base model, when interpreted as a likelihood ratio, overstates evidence on the correct class.
```
**New:**
```latex
I generally find that the base-model \phle score, interpreted as a log-likelihood ratio, overstates the evidence for the correct class.
```

## 24. Appendix B (pool sizes)

**Why:** Explain why the main text's 2% (no-prompt answers removed by the word filter) is larger than the table's first column suggests.

**Old:**
```latex
The ``after word filter'' column counts answers that survived the animal and train filter applied at generation time, out of 1,200 (300 questions $\times$ 4 samples).
```
**New:**
```latex
The ``after word filter'' column counts answers that survived the animal and train filter applied at generation time, out of 1,200 (300 questions $\times$ 4 samples); the filter is re-applied, with a slightly extended word list, before the judge, which is why the percentages quoted in Section~\ref{sec:setup} are slightly higher.
```

## 25. Appendix C (Qwen)

**Why:** Same Qwen wording fix.

**Old:**
```latex
The performance of \phle is similar to or better than for Olmo,
```
**New:**
```latex
The performance of \phle is similar to Olmo's, better on text and mixed on number sequences,
```

## 26. Appendix K (calibration)

**Why:** My sentence was half wrong: the owl vs trains (text) correction makes the other two text comparisons worse than raw (1.19 vs 0.80; 0.88 vs 0.70); only the corrections from owl vs eagle (text) and secret harm vs secret friend transfer well.

**Old:**
```latex
whose slopes are similar (0.25--0.32) and whose corrections transfer reasonably well among each other.
```
**New:**
```latex
whose slopes are similar (0.25--0.32); the corrections fitted on owl vs eagle and on secret harm vs secret friend transfer well to each other, but the correction fitted on owl vs trains, which has a larger offset, makes the other two worse than the raw score.
```
