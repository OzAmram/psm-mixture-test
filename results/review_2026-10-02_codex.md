# Independent review of the persona selection study

Date: 2026-10-02. Reviewer: Codex.

This independent portion was written **before opening `results/review_2026-10-01.md`**. I inspected `results_writeup.md`, generation/scoring/training scripts, the mixture implementation, analysis notebooks and their outputs, saved score matrices, teacher/student metadata, and relevant primary literature. I ran only lightweight CPU analyses of existing results and CPU tokenization. I did not generate model responses, run GPU inference, fine-tune models, or launch new studies.

Reproducible checks: [script](../scripts/review_cpu_checks.py) and [computed statistics](review_2026-10-02_cpu_checks.json). The script uses saved likelihoods, question-level resampling, within-question permutations, and small supervised text-feature checks. These checks validate the saved scores' statistical signal; they cannot validate model forward passes without rescoring.

## Assessment

**The most convincing finding is that OLMo base's persona-conditioned likelihoods contain a weak but correctly directed owl-versus-eagle signal, including on number sequences.** It survives balancing questions and linear length adjustment. That is an interesting result worth developing. It is stronger evidence for reusable prompt-sensitive statistical structure in the pretrained model than for a particular latent-persona mechanism.

Most numerical classification claims trace to saved results. However, several claims need correction or narrower interpretation before publication:

1. The 28-nat mixture headline uses an acknowledged, subsequently corrected leading-space error in the Instruct self-scores. The corrected saved reference implies approximately **37.5 nats per response on the 800-answer holdout**, under the existing base-side scoring convention.
2. “Inside the span,” “dominant effect is entropy reduction,” “not a mixture of any base personas,” and “post-training does not create the persona” exceed what was measured.
3. Reported uncertainty is mostly answer-level and unmatched, while the most defensible bird result is question-balanced. Question-level intervals are appreciably wider.
4. The student paragraph attaches AF-versus-control uncertainty and matching results to an AF-versus-friend claim.
5. The Instruct likelihood detector does not currently score the exact generated token sequence. Its Neyman–Pearson/optimality and “ceiling” descriptions require qualifications.

I would foreground the subtle-prompt classification result and present the mixture work as a useful, limited test of a deliberately strong operational hypothesis.

## 1. Owl versus eagle: independent verification

I recomputed the base-model ratio from `scores_eagle20_olmo_base_{text,numbers}.jsonl`, residualizing it against token length as the writeup does. I also computed within-question AUROC, equal-question-weighted mean score differences, question bootstrap intervals, and one-sided label permutations **within each question**, preserving each teacher's sample count for that question.

| Check | Text | Number sequences |
|---|---:|---:|
| Scored answers per teacher | 500 | 1,000 |
| Questions represented in both teachers | 193 | 275 |
| Pooled per-answer AUROC | 0.571 | 0.583 |
| Mean within-question AUROC, equal question weight | 0.556 | 0.583 |
| Equal-question mean ratio gap, nats | 0.207 | 0.134 |
| Question-bootstrap 95% interval for mean gap | [0.062, 0.364] | [0.088, 0.177] |
| Within-question permutation p, one-sided | 0.004 | 0.001 |
| Question-balanced bag AUROC, k=10 | 0.665 | 0.753 |
| Question-balanced bag AUROC, k=30 | 0.797 | 0.869 |
| Approximate question-bootstrap 95% interval, k=30 | [0.624, 0.917] | [0.788, 0.930] |

These Monte Carlo bag estimates agree with the writeup's matched results, 0.67/0.80 for text and 0.75/0.88 for numbers, within simulation variation. The p-values are exploratory, unadjusted for the project's many comparisons, and resolution-limited by 999 permutations. The bag intervals use 120 cluster bootstrap replicates and fixed full-sample length residualization; they are useful sensitivity checks, not final publication intervals.

**Interpretation:** text has a modest within-question effect with substantial uncertainty; numbers have a more robust within-question effect. Matching does not erase the finding. The number-sequence experiment is especially useful because the answer has no ordinary bird semantics and there is no animal-word filter. There is still a numerical-format validity filter, so it is not an entirely unselected sample.

### Remaining threats to the bird interpretation

**Historical filters were not identical.** `owl/meta.json` records an earlier animal regex without eagle/talon/raptor/soar and train/rail/journey terms; `eagle/meta.json` records the expanded regex. Six scored owl answers match terms added later; no scored eagle answers do. Removing these six leaves pooled AUROC 0.571 and question-balanced k=30 approximately 0.821. This discrepancy does not explain the result, but the manuscript's uniform-filter description is inaccurate. Apply and record one version of the filter to all pools. The differential survival rates also condition on different subsets of answers *within* a question; question matching fixes question composition, not that selection effect.

**Linear residualization is not complete length/format control.** Numbers average 14.84 versus 11.90 base-tokenizer tokens; the writeup's length-only k=30 AUROC, 0.87, is as high as the base ratio's. A linear regression removes a linear association, not nonlinear effects of sequence length, comma/space formatting, number magnitude, or number count. The persistence after residualization is encouraging but should not be described as conclusively independent of format.

As a quick diagnostic, I fit word and character TF-IDF logistic regressions on one half of the questions and evaluated on the other half. Text AUROC was 0.485/0.516, whereas numbers were 0.547/0.583. The character baseline on numbers reaches essentially the base detector's **per-answer** AUROC, with labelled training. This does not negate the base model's zero-shot achievement, and one split is not a definitive comparison. It does suggest that simple numeral/format statistics are a plausible carrier of the number fingerprint. Compare both methods on identical held-out questions, at k=1 and k=30, with length/count-matched controls.

**Semantic specificity is not fully established.** The correct bird ratio outperforms the three reported owl-free controls; that is good evidence of hypothesis-specific information. But three unrelated headers do not establish that the model recovered “love of this bird” rather than shared lexical associations or prompt-token effects. The decisive next test is transfer across independently chosen paraphrases of teacher prompts and scorer descriptions, alongside nonsense-name and generic-preference controls. This is a proposed new study, not one I ran.

**AUROC is not classification accuracy.** “About 0.9 from 30 answers” is a ranking metric on bags resampled from these finite pools. It is not a demonstrated 90% success rate at a deployable threshold. In particular, the text headline should use matched 0.80, not raw 0.89, and report threshold accuracy/sensitivity only after choosing a threshold on separate calibration questions.

## 2. Statistical design and aggregation

The manuscript has added useful controls: matched questions, wrong ratios, label-permutation nulls, and confidence intervals. Those substantially improve the original design. Several details still matter:

- `scripts/aggregate_ci.py` bootstraps individual answers and permutes across the pooled classes. Repeated answers share a question; the uncertainty needed for generalization to new questions should resample questions jointly across teachers. The reported intervals are for unmatched aggregation, not for the matched values printed alongside them. Comparing a matched estimate to an unmatched pooled permutation percentile is not a matched significance test. The within-question checks above directly address this for the principal base-model comparisons.
- Summing signed pure-noise scores does **not** intrinsically push expected AUROC above 0.5. A finite pool can have an accidental mean difference that is magnified by aggregation; folding AUROC about 0.5 also produces positive bias. The reported folded-null means describe that combination. Clarify this distinction. Wrong ratios falling inside the null band fail to establish a residual effect; they do not prove the population signal is exactly zero.
- The likelihood aggregation samples with replacement; prompted classifiers generally draw distinct answers within each bag. Many simulated bags reuse the same finite pool. Thousands of bags do not become thousands of independent experimental datasets. Use identical item pools, distinct questions/answers per bag, and paired comparisons of methods.
- The base eight-way calibration is explicitly split by answer, not question (`notebooks/3.5_validation_multiway.ipynb`). Both calibration and evaluation therefore contain answers to the same dilemmas. This is transparently disclosed, but the result is not a held-out-question deployment estimate. All sources are known, and calibration assumes access to labelled examples from every source.
- The supplied hypotheses form a closed set. Eight-way argmax does not answer “which of these, **if any**”: no rejection mechanism for an unseen prompt or model is implemented. A no-prompt candidate is not an unknown-prompt candidate.

The AF-versus-friend base result survives the additional within-question check: mean gap 0.312 nats [0.226, 0.397], permutation p=0.001, and matched k=30 approximately 0.886. Thus the main secret-prompt finding also remains supported by existing scores.

## 3. Likelihood scoring: what is and is not exact

The teacher-forcing gather and fp32 log-softmax in `src/persona_selection/scoring.py` look correct. The joint-tokenization prefix assertion is a sensible guard against boundary merges. EM over fixed components is also implemented correctly.

The **caller**, however, matters. Both `scripts/subliminal_score.py` and `scripts/phase1_instruct_sample.py` prepend a space to the stripped answer before scoring, including for chat-template Instruct scorers. Generation itself does not add that space. CPU tokenization confirms that `Talk` and ` Talk` start with different token IDs, and numerical responses acquire an additional space token. Thus even the same-family Instruct scorer is a proxy score, not exactly the teacher's log density. The base transcript's leading-space convention is appropriate for its own prompt; it should not be imposed on the chat scorer.

The prefix experiment already acknowledges this error and uses corrected self-references in `results/prompt_tune/true_reference.json`. The correction has not propagated to the mixture headline or the main fingerprint scoring pipeline. I cannot determine its effect on AUROC from scalar saved scores. A first-token contribution could matter when the total base gap is only a fraction of a nat. Rescore with preserved generated token IDs before claiming an exact likelihood test.

There are other reasons to distinguish a scored text prefix from a normalized response distribution:

- Responses are stripped, special tokens removed, capped, and sometimes cut at `User:`; termination probability is generally omitted. A product of probabilities for a variable-length prefix is not automatically the probability of the saved, completed response under that stopping/cleanup procedure.
- Generation scripts override EOS with `tokenizer.eos_token_id` (100257, `<|endoftext|>`), whereas the cached Instruct generation configuration also recognizes 100265, `<|im_end|>`. Verify intended end-of-turn handling. Only 17/2,400 Phase I Instruct examples hit the recorded 80-token cap, so this observation alone does not show a large contamination effect; the original tokens are needed to check it.
- `prompt_tune_truereference.py` says generation defaults to top-k=50. In the current installed Transformers 5.17.0, loading the cached Instruct generation configuration yields `top_k=None`. The runs do not save a fully resolved generation configuration or software version, so the historical effective default is not established by command-line arguments. Explicitly set decoding parameters and record them. The saved top-50 correction is only about 0.0055 nats/token for Phase I, much smaller than the leading-space correction.
- Filtering produces distributions conditional on passing a selection rule. For a fixed question, exact teacher likelihood ratios need a selection-normalization constant; across questions it can vary. Correct normalization and stopping conventions matter particularly for uncalibrated multiway argmax and mixture-fraction recovery.

Neyman–Pearson optimality holds for the likelihood ratio of the **actual two data-generating distributions**. It does not automatically hold for base-model persona proxies, transformed response prefixes, or filtered samples scored without the corresponding normalizations. “Likelihood-ratio detector” remains a fair name; “the most powerful test” needs that qualification.

Also, “no prompt” is not literally no system message. OLMo's chat template inserts a default function-calling assistant message, confirmed in the saved Phase I config and [official model card](https://huggingface.co/allenai/Olmo-3-7B-Instruct). Explicit trait prompts replace this default. Consequently, AF-versus-control combines a trait change with a system-message change. The same-frame AF-versus-friend and owl-versus-eagle controls are more informative than prompted-versus-default contrasts.

## 4. Mixture results and their interpretation

### Numerical checks and concrete corrections

I reproduced the full-basis held-out log-score gap of **0.2776 nats** for clean `base_unknown_casual_v1` answers. Including the meta/placeholder responses increases it to **0.3295**. The format-matched short-answer run gives **0.3133** clean and **0.3926** before this filter. The fit is good relative to the hand-written basis, but the conditioning and filtering are part of what is being tested.

The 0.28 result is **not** from prompts ending “Answer in two or three sentences.” Its saved config has the casual header but no user suffix. The suffix experiment is the separate `base_unknown_casual_short_v1` run. Section 2 currently conflates them. The clean 0.28 analysis has 2,346 responses, from 2,388 saved nonempty samples, rather than exactly 2,400 analyzed samples.

The three-row 0.70 → 0.46 → 0.28 table mixes conditions. `1.5_register_control.json` gives the clearer factorial comparison:

| Conditioning | Six hand-written personas | All 86 personas |
|---|---:|---:|
| Plain header | 0.926 | 0.477 |
| Shared casual clause | 0.701 | 0.278 |

This shows basis expansion and register control separately. Other earlier clean subsets give approximately 0.46, but that should not be presented as the middle of a single controlled progression starting at casual 0.70.

The Instruct headline's original 27.9477-nat gap is reproducible from the saved, space-prefixed self-scores. It is **not the corrected self-reference**. On the 100-question/800-answer prefix holdout, fitting weights on the other 200 questions gives base mixture log-score/token −1.7001 and old self-score/token −1.0966. The saved corrected self-reference is −0.8912 over 37,104 answer tokens, versus 37,084 in the base-side representation. Combining these totals gives a corrected mean log-score gap **37.4764 nats/answer**. This is a correction on that exact subset, not a new estimate for the original 150-question split, nor a fully normalized sequence KL.

Two smaller numerical issues: the full casual-basis evil weight is approximately **0.002**, not literally 0.000; and the five-component calibration's largest weight error is **0.023**, slightly outside a literal ±0.02 claim. Approximate wording is sufficient.

### Why the interpretation is too strong

**A likelihood advantage over a generic header does not establish membership in a span.** The −1.63-nat comparison shows that this fitted persona mixture assigns higher likelihood than a particular generic base prompt to Instruct answers. It is not a semantic projection, a convex-hull membership test, or evidence that all content is explained. Indeed, all ordinary finite texts generally already have nonzero softmax probability. “Inside their span” should become “the persona basis predicts these answers better than the generic base header.”

**Cross-entropy mismatch is not an identified entropy-reduction mechanism.** A large self-versus-mixture gap measures fit failure under the scoring conventions. It can reflect changes in preferred tokens, factual or reasoning ability, syntax, style, chat formatting, and/or conditional sharpness. Mixture entropy is at least the weighted average of component entropies on a common normalized space; the likelihood on *Instruct samples* is not the entropy of those components. To claim sharpening is dominant, measure component entropies on their own outputs, test temperature-adjusted components on held-out data, and examine how much mismatch disappears. The prefix runs are valuable evidence of limited conditioning capacity, not a causal decomposition of post-training.

**Only a finite, description-conditioned basis was rejected.** Eighty-six descriptions and several learned embedding prefixes do not exhaust base-model personas, contexts, or conditioning mechanisms. A plateau across the tested prefix settings is meaningful evidence of a practical limit of those methods; it is not a proof of impossibility. “Not a mixture of any base personas” should specify “not well fit by our tested description-conditioned basis.” The hypothesis that the 0.28 residual is entirely basis quality is also untested.

**Global persona weights impose an additional assumption.** A latent-character model can have question/context-dependent weights. Failure of a fixed global weight vector need not be failure of a persona-selection account. Conversely, a good finite mixture fit does not identify the personas as the model's true latent causes. Highly similar components can exchange weight, so labels like “sarcastic friend 14%” are descriptive coordinates in this basis, not uniquely recovered character frequencies.

**The elbow and bootstrap have implementation problems.** `greedy_forward_selection` selects each new component using the same held-out split it reports; the elbow is model selection on that split, not untouched test performance. This does not invalidate the separate full-86 fit, whose weights are fitted on training questions. `bootstrap_over_groups` assigns repeated copies of a sampled question different group IDs and then splits them, allowing the same original question into both fit and evaluation. Preserve original identity or bootstrap fixed disjoint train/test question sets. The `kl_se` values shown in several tables are answer-level standard errors even when question bootstraps were also computed.

**Calibration does not settle probability normalization.** The synthetic reference is constructed from the same component score matrix. Its near-zero fitted-versus-constructed gap is useful evidence that EM recovers weights, but does not independently verify that cleaned, stopped samples have the claimed normalized component likelihoods.

## 5. Other claims: audit and limits

| Claim | Verdict |
|---|---|
| Owl/trains base AUROC 0.62; matched k=30 about 0.94 | Supported by saved scores and `aggregate_ci.json`; length and filter selection remain relevant. |
| Owl/eagle generator scores much stronger than base scores | Supported for the implemented scorer; exact likelihood/ceiling wording requires scoring fixes. |
| AF/friend base AUROC about 0.58 and k=30 about 0.89 | Supported; also survives my within-question checks. |
| Prompted 7B and GPT classifiers fail on subtle bird/secret pairs | A fair description of most tested configurations; not a demonstrated inability of prompted classifiers generally. |
| Instruct eight-way 0.62/0.91/0.96/0.99; calibrated base 0.22/0.37/0.41/0.52 | Traceable to notebook 3.5; closed-set, reused-pool, answer-split estimates. |
| Instruct mixture recovers 80/20 and 50/50 source composition | Supported as average finite-pool recovery in the reported experiment; report dispersion and filtering/scoring assumptions. |
| Cross-family fingerprint partly transfers | Supported, with the distinction between prompted/default style and specific hidden intent. |
| No measured AF behavioural transfer | Supported on the probes; does not establish behavioural equivalence or absence of misalignment. |
| Base fingerprint survives student distillation | Supported for the saved student pair, including held-out training questions; uncertainty and pair provenance need correction. |
| Learned prefixes leave roughly 0.45–0.51 nats/token gap | Supported by corrected notebook references for the tested settings; does not establish a universal conditioning bound. |

### Student pair mismatch

Section 3.8 describes **AF student versus friend student**, then gives base 0.59/0.90 [0.80, 0.95], matched 0.93. The confidence interval and 0.93 matched value in `aggregate_ci.json` belong to **AF student versus control student**, scored with AF minus neutral (`aggregate_ci.py`, CELLS entry).

For the actual AF/friend pair, my length-adjusted recomputation gives 0.588 per answer, question-balanced k=30 about **0.908**, approximate question-bootstrap interval **[0.806, 0.953]**, and a positive within-question mean gap 0.357 nats [0.236, 0.495]. The signal survives; the provenance still needs fixing. GPT's student rows likewise use AF versus **control**, not friend. Label the pairs separately. The reported “base signal unchanged by distillation” is consistent with point estimates, but not an equivalence test, and it is based on one saved training seed per student condition.

### Negative/inverted classifier scores and behavioural nulls

GPT few-shot AF-versus-default AUROC 0.073 at k=30 is not chance performance: reversing it would give 0.927 on that pool. Calling it a prompt artefact may be reasonable, but requires examination of label order, exemplar selection, and generalization of the reversed direction. It cannot simply be discarded as no information. Conversely, uncalibrated AUROC below 0.5 is not an operational detector unless the direction can be learned on separate data. This applies symmetrically to wrong-ratio controls.

GPT per-answer scores contain quantization/ties and missing-letter logprob fallbacks. The narrow A/B format and randomized order can add nuisance variation, and repeated pooled scores amplify finite-pool differences. The result is an advantage over these particular prompted baselines, not proof that humans or sufficiently well-designed classifiers cannot identify the prompts.

Identical AF/default opportunity-probe counts (5.8% each) are evidence of no detected increase, not proof of no misbehaviour. The judge is an imperfect same-family classifier and the probe set is small. The writeup correctly notes that the AF teacher is not demonstrated to be an alignment faker; preserve that distinction throughout. The human “indistinguishable” claim remains anecdotal; the supplied confidentiality example itself differs noticeably in willingness to contemplate unauthorized sharing.

### Persona prior and sarcasm conclusions

The 39% friendly/helpful etc. proportions describe text characteristics in a prompted, filtered, deduplicated sample of descriptions. `phase1_elicit_personas.py` keeps 80 of 96 raw descriptions using length, content, sentence-truncation, and similarity filters. These overlapping textual categories are not a measured pretraining persona prior. Zero explicitly malicious descriptions in this sample does not establish that malicious personas are absent.

The selfishness/sarcasm control finds fixed-weight sarcastic-friend responsibilities 14.9%, 13.4%, 13.6% across temptation/neutral/harm questions. That supports a broadly activated style component. But equal activation across categories does not isolate tone from values, and the per-category refits move its weight to 11.9%, 7.1%, 3.2%. Treat “tone, not values” as a plausible interpretation, rather than a demonstrated causal separation.

## 6. What these results say about PSM

The [original PSM article](https://alignment.anthropic.com/2026/psm/) explicitly permits new capabilities during post-training, context-dependent personas, and an incomplete persona-based account of behaviour. The manuscript's fixed-component mixture identity is therefore a **strong additional operationalization**, not a direct theorem implied by every version of PSM. This is worth stating near Prediction A.

Prediction B is more persuasive, but ordinary language modelling can also explain weak preference-conditioned lexical/numerical correlations. A same-family base model shares vocabulary, pretrained representations, and many statistical associations with the teacher. The observed asymmetry with cross-family scorers and the much stronger generator scores fit that account too. The results show some pre-existing structure that can read prompt differences; they do not quantify “most” of the persona-relevant structure or establish that post-training cannot create new traits.

[Hu, Koyejo and Potts](https://arxiv.org/abs/2609.16927) is relevant prior work: its abstract explicitly reports trait recovery even when behavioural subliminal learning fails, and treats prompted subliminal datasets as context distillation. Thus detection without behavioural transfer is not unique evidence for PSM. The novel emphasis here is **base-model transfer of a finite-hypothesis fingerprint**, especially the subtle bird pair. [Cloud et al.](https://arxiv.org/abs/2507.14805) establishes behavioural transfer in its tested settings; the present OLMo negative result should remain scoped to this model, adapter training, traits, and evaluations.

A defensible headline would be: *A pretrained model, conditioned on simple persona descriptions, can weakly identify subtle hidden-prompt differences in its post-trained descendant's otherwise ordinary outputs. The signal accumulates across answers and survives distillation, while a tested finite persona mixture and learned input prefixes do not reproduce the descendant's full response statistics.*

## 7. Recommended next steps, in priority order

1. **Repair score provenance before interpreting mechanisms.** Rescore existing generated tokens with the exact Instruct continuation, correct EOS/stopping conventions, and recorded resolved decoding configuration. Propagate corrected references to every table. Fix the student pair labels and regenerate manuscript statistics from a single manifest specifying file, teachers, hypotheses, filters, and split.
2. **Make the bird result the primary replication target.** In a new study, pre-register a held-out question domain and independent generation seeds, use one filter version, collect balanced repeated answers to common questions, and run text plus numbers. Use cluster intervals and within-question permutations. Test scorer-header and teacher-prompt paraphrases without selecting them on the test set. This would distinguish a reusable preference signal from a wording-specific fingerprint. I did not run this study.
3. **Find the signal's carrier using existing-data rescoring.** Separate first-token effects from later tokens; test matched answer lengths, number counts, separators, and value distributions; compare number-feature and character classifiers, and inspect which tokens contribute. Report zero-shot base scoring separately from supervised baselines. This needs additional model scores, so I only ran the saved-data checks above.
4. **Test semantic specificity and practical detection.** Add nearby birds, unrelated preferences, invented labels, negated preferences, and unseen prompts. Require correct direction under paraphrase and use a separately calibrated threshold/rejection rule. Run a blinded human test if “indistinguishable to readers” is central.
5. **Test sharpening rather than infer it.** Fit a shared temperature and/or persona-specific temperatures on training questions, evaluate on untouched questions, and compare to richer conditioning. Measure component entropy on component samples. If temperature changes close most of the normalized gap without changing semantic behaviour, “selection plus sharpening” gains direct support.
6. **Replicate distillation across seeds and test a positive behavioural control.** Keep the genuine AF/friend comparison, match training sizes, and report teacher-to-student attenuation with uncertainty. To make a claim about misalignment transfer, first establish a teacher that actually expresses the target behaviour on held-out probes.
7. **Tighten reproducibility.** Save model/tokenizer revisions, dependency versions (including sklearn/peft/OpenAI tooling), resolved generation configs, original token IDs, filter versions, split manifests, and a CPU-only table-generation entry point. Keep immutable raw results and explicit joins when merging rescore files rather than relying solely on row order.

**Independent-review conclusion:** I would keep and prioritize the base-model subtle-prompt classification finding. I would correct the scoring and provenance issues, expand uncertainty for matched text results, and substantially soften the claims about entropy, latent personas, and exhaustive rejection of PSM. The existing evidence warrants a focused replication, especially of owl versus eagle across prompt paraphrases and fresh domains.

## 8. Comparison with the October 1 review — added after the independent report

I opened `results/review_2026-10-01.md` only after writing sections 1–7 and saving their SHA-256 in `review_2026-10-02_codex.independent.sha256`. The earlier review appears to assess an older manuscript: several of its disclosure and statistical recommendations are already incorporated in the October 2 writeup. The independent portion above has not been rewritten to match it.

### Agreement

- **The subtle-prompt signal is real in the saved scores.** We independently reproduce the principal AUROCs and agree that question balancing reduces the base text owl/eagle result to approximately 0.80 at k=30. My additional within-question permutations support a remaining positive signal.
- **Question composition, aggregation uncertainty, and folded-null drift matter.** The previous review correctly identifies these problems. The current manuscript now includes matched values, answer-level intervals, and a permutation null. My remaining recommendation is to make the uncertainty and null match the question-balanced estimand, rather than declare the issue fully resolved.
- **The few-shot GPT result is inverted, not chance.** The current writeup now reports 0.41/0.25/0.07 and calls it a prompt artefact. We agree about the inversion; I think the artefact interpretation still needs evidence and separately calibrated validation.
- **Same-family judge filtering, keyword filtering, metric distinctions, and anecdotal human inspection should be disclosed.** They mostly are disclosed now. Comparable aggregate rejection rates do not establish absence of selection bias within a question.
- **Cross-family ordering is fingerprint evidence, not uniquely PSM evidence.** This is a particularly important shared conclusion.
- **Temperature controls, a verified behavioural teacher, broader families, and mechanistic localization would be useful.** I agree with these directions. Prefix failures are worth reporting as practical constraints on tested conditioning methods.

### Disagreement or stronger qualifications

**The previous review calls the 28-nat KL “legitimate” because temperature and top-p were both one. I disagree.** Those settings do not establish exact sampling/scoring correspondence. More decisively, the stored self-score prepends a spurious space; the repository's later corrected reference changes −1.10/token to −0.89/token. On the same 800-answer holdout, the resulting score gap is roughly 37.5 nats, not 28. Cleaning, stopping, and missing termination probabilities are additional obstacles to identifying this as an exact normalized sequence KL. Reproducing an existing erroneous scalar does not validate its interpretation.

**“Inside the span” and “selection plus sharpening” are not established.** The earlier review accepts these conclusions more readily. A persona mixture outperforming a generic header is a likelihood comparison; it is not a span-membership result. The gap from self-scores is not a causal measurement of entropy reduction. I regard sharpening as a testable hypothesis, not the current finding. Also, the original PSM formulation permits post-training learning and contextual conditioning; the fixed-mixture identity is a stronger operational assumption.

**The earlier review's “negligible” qualification for answer-split multiway calibration is unverified.** A small number of fitted offsets does not ensure that shared-question effects are irrelevant. The proper next step is to compare question splits empirically, not assume the difference is negligible. The reported result can remain, clearly labelled as within-question-distribution, closed-set calibration.

**A per-question neutral baseline cannot by itself fix raw multiway argmax.** Subtracting the same neutral score from all eight hypothesis scores leaves their ordering exactly unchanged: `argmax_h(ll_h − ll_neutral) = argmax_h(ll_h)`. A common per-question adjustment cancels. Contrastive header design, genuinely hypothesis-dependent corrections, or a separately justified calibration method could help; the proposed common subtraction cannot resolve the HHH offset problem.

**The temperature suggestion needs a distinction between sequence and token tempering.** Normalizing `P(sequence)^(1/T)` over all sequences is not generally equivalent to scaling each token's logits and multiplying the resulting conditional probabilities. The latter is practical, but its normalization depends on every prefix. Either can be a useful specified test; they should not be conflated, nor should T≈0.6 be treated as an established target.

**Scaling is not uniquely PSM-specific.** Improved base fingerprint AUROC with model size could follow from better language modelling or richer prompt associations. A scale study would improve generality, but needs competing predictions to discriminate mechanisms.

**Several “supported” claims need smaller factual corrections.** The prior review accepts literal evil=0.000 and calibration ±0.02; the casual full-basis evil weight is about 0.002 and one calibration error is 0.023. It does not identify the AF/control versus AF/friend provenance mismatch in the updated student uncertainty, the historical regex-version mismatch, or the train/test duplication in the group bootstrap. These are additions from this review, not disagreements about whether the overall fingerprint exists.

Both reviews therefore support retaining the empirical base-model fingerprint finding. My assessment is more cautious about the claimed probability identities and mechanistic conclusions, and I put exact rescoring and a fresh, paraphrase-robust owl/eagle replication ahead of further interpretation.
