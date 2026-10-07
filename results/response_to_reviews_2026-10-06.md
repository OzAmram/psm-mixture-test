# Response to the two reviews: changes made 2026-10-05/06

Reviews: `results/review_2026-10-01.md` (Claude reviewer agent) and `results/review_2026-10-02_codex.md` (Codex). The manuscript they refer to is `results_writeup.md` as of commit a54731e / 371862d; the revised manuscript is the current `results_writeup.md` (commit fda54b5 and later). Code commits: 67172a8, b3e89be, ea312c7, fda54b5, 35a6d29.

## A. Errors in scoring and provenance (both reviews; Codex §3, §5)

| Point | Status | What was done | Where |
|---|---|---|---|
| A1. Instruct self-scores used a spurious leading space inside the chat template (Codex §3, §4); the 28-nat headline was computed against them. | **Fixed** | Every Phase 1 Instruct sample rescored as the exact chat-template continuation (no leading space, no end-of-turn token). Exact self log P = −0.898 ± 0.009 per token (stored: −1.102). Part I now reports −1.722 (mixture) vs −0.898 (Instruct), gap 0.82 nats/token ≈ 37 nats/response. | `scripts/phase1_rescore_self.py`, `results/phase1/instruct_unknown_casual_v1/rows_selfexact.jsonl`, `scripts/phase1_loglik_table.py`, `results/phase1/loglik_table.md` |
| A2. The fingerprint pipeline's chat-template scorer also prepended a space (Codex §3). | **Fixed for every OLMo-Instruct and Qwen-Instruct text cell; Qwen-Instruct numbers and students not rescored, marked (old)** | `subliminal_score.py` now prepends the space only for the base transcript format. All Instruct-scored files used in the manuscript were rescored (`scores_exact_*.jsonl`): OLMo-Instruct text / students / numbers, Qwen-Instruct text, OLMo-Instruct on Qwen-teacher answers. Effect: text AUROCs unchanged within ±0.02; number AUROCs rose (owl vs eagle 0.70 → 0.82 per sequence; AF vs friend 0.67 → 0.75), because a leading space before a digit string changes its tokenization. Qwen-Instruct on number sequences and on the students was not rescored and is marked "(old)" / "(not rescored)". | `scripts/subliminal_score.py` line "resp = …", `scripts/subliminal_stage21.sh`, `results/tables.md` |
| A3. Student paragraph attached AF-vs-control uncertainty to an AF-vs-friend claim (Codex §5). | **Fixed** | Section 3.8 is a table with both pairs labelled; AF student vs friend student: initialisation 0.67 ± 0.02 / 0.90 ± 0.03 / 0.99 ± 0.01, base 0.59 ± 0.02 / 0.77 ± 0.04 / 0.90 ± 0.04. | `results_writeup.md` §3.8 |
| A4. Historical filter versions differed between owl and eagle pools (Codex §1). | **Fixed** | One regex (the current `TEXT_FILTER`) is applied to every text pool at table time; the manuscript says so and gives the per-teacher removal rates. | `scripts/make_tables.py` (`load`), §3.2 item 2 |
| A5. `bootstrap_over_groups` gave resampled copies of a question different IDs (Codex §4). | **Fixed** | Copies keep the original question ID, so a question cannot be on both sides of the fit/evaluation split. | `src/persona_selection/mixture.py` |
| A6. Greedy elbow selected on the evaluation split (Codex §4). | **Disclosed** | The elbow is now described as model selection on the evaluation split, descriptive only. | §2 |
| A7. Small numbers: evil weight 0.000 → 0.002; calibration ±0.02 → 0.02–0.03; the 0.28 run has no "two or three sentences" suffix; 2,346 not 2,400 analysed (Codex §4). | **Fixed** | All four corrected in §2. | §2 |

## B. Uncertainty and aggregation (review 1 concern 1; Codex §2)

| Point | Status | What was done |
|---|---|---|
| B1. No uncertainties on k-aggregated AUROCs; answer-level intervals next to matched point values; "drift" of folded wrong ratios is finite-sample noise. | **Fixed (second round: folded inside each replicate, length regression refitted per replicate, exact-estimator null rows; classifier errors are Hanley–McNeil over trials and labelled as conditional Monte-Carlo error, not data uncertainty)** | One generator (`scripts/make_tables.py`) produces every AUROC table. Each cell is value ± 1σ, σ = standard deviation over 200 question-level bootstrap replicates (questions resampled jointly for both teachers, the whole statistic recomputed). Reported statistics: per answer, and matched-question bags of k = 10 and 30 (one random answer per question per teacher, log-ratios summed). Length regressed out of every ratio. Classifier rows carry the Hanley–McNeil SE. Folded wrong ratios are compared with the stated no-signal band (≈0.55 at k = 10, 0.58 at k = 30; 95th pct 0.61 / 0.68) and no "drift" is claimed. |
| B2. Matched-question values should be the estimand for the bird pairs (review 1 concern 2; Codex §1). | **Fixed** | All k > 1 values in the manuscript are matched-question. Owl vs eagle, base, text: 0.54 ± 0.02 / 0.71 ± 0.06 / 0.82 ± 0.08 (was 0.57 / 0.75 / 0.89 unmatched); numbers 0.58 ± 0.01 / 0.76 ± 0.04 / 0.88 ± 0.04. |
| B3. Resampling with replacement / reuse of finite pools (Codex §2). | **Partly** | Bags now use distinct questions within a bag; the bootstrap is over questions. Bags still reuse the finite pool across replicates; the manuscript says AUROCs are rankings on finite pools, not deployable accuracies. Paired method comparisons on identical bags were not implemented. |
| B4. Multiway calibration split by answer, not question (both reviews). | **Disclosed, not re-run** | §3.6 states the calibration is within the same questions and closed-set, and that a common per-question baseline cannot change an argmax (Codex's correction of review 1's suggestion). A question-split re-run was not done. |

## C. Baselines (review 1 concern 3; Codex §5)

| Point | Status | What was done |
|---|---|---|
| C1. GPT-4.1 ten-shot result is inverted, not chance; pooled per-answer scores near-constant. | **Fixed** | §3.5 reports 0.41 / 0.25 / 0.07, calls it inverted and unexplained, notes it cannot be used without learning its direction on separate data, and states the near-constant pooled scores (65% identical). |
| C2. No CoT / probability-elicitation variant for GPT-4.1. | **Disclosed** | Stated in §3.5. |
| C3. 7B classifier cells mixed AUROC and CoT accuracy. | **Fixed** | Tables show the single-letter AUROC rows only; CoT accuracies are in the repo tables, not the manuscript tables. |

## D. Interpretation (review 1 §Relevance; Codex §4, §6)

| Point | Status | What was done |
|---|---|---|
| D1. "Inside the span" is a likelihood comparison, not span membership. | **Rewritten** | §2: "the persona mixture predicts Instruct's answers better than the generic base header does (0.04 nats/token)". |
| D2. "Dominant effect is entropy reduction" / "a mixture cannot be sharper than its components" not measured. | **Rewritten; second round: 'much sharper' → 'assigns these continuations far higher likelihood'; the mixture DOES beat the best single persona by 0.008 nats/token (paired 95% [0.007, 0.010]), now stated, with the component chosen on the fit half** | §2 and §4: the mixture is no better than its best single component and 0.82 nats/token remain; whether what is missing is a tempered version of the same character or new content "is not decided by this test"; the tempered-mixture test is named as the deciding experiment. "Selection plus sharpening" is called a hypothesis consistent with both parts. |
| D3. "Not a mixture of any base personas", "post-training does not create the persona". | **Rewritten** | Replaced by "a mixture over this basis is not the post-trained model, and neither is any soft-prefix context we could learn"; §4 says whether post-training added content is not separated by these measurements. |
| D4. PSM's original formulation permits new capabilities and context-dependent personas; fixed-weight mixture is a strong operationalisation. | **Added** | §1 says so; §2 "Fit" lists what is and is not tested (fixed global weights, description-defined basis, cleaned truncated samples). |
| D5. Neyman–Pearson optimality claimed for proxies. | **Fixed** | §1: optimality claimed only for the generator with the exact prompts; base headers are "a proxy with no such guarantee". |
| D6. Cross-family ordering is fingerprint evidence, not PSM-specific. | **Added** | §3.7 and §4. |
| D7. Persona "prior" from elicited descriptions overclaimed; "tone, not values" overclaimed. | **Rewritten** | §2: categories of a prompted, filtered sample; "a plausible reading, not a demonstrated separation". |
| D8. Hu, Koyejo & Potts already report detection without behavioural transfer. | **Added** | §3.1 cites it as prior work; the novelty is restated as reading the prompt with a different model's likelihoods and a finite hypothesis set. |

## E. Disclosures (review 1 concern 4; Codex §3, §5)

Added to §3.2 or §2: "no prompt" = OLMo's default function-calling system message; the keyword regex (~9%); the judge is the teachers' own model family and could select differently within a question; aggregate rejection rates equal; "indistinguishable" rests on our reading, no human study; one training seed per student; numbers pass only a format filter; sampling temperature 1 / top-p 1.

## F. New experiment: "do not mention X" teachers (not requested by the reviews; motivated by the filter asymmetry)

Teachers with the clause "However, do not mention owls / eagles / trains in your responses." Text (1,200 answers each; regex removes 2% instead of 48%) and 3,000 number sequences each, scored under base and exact-Instruct scorers (`scripts/subliminal_stage22.sh`, `_stage23.sh`; `scores_nm_*`).

| owl_nm vs eagle_nm | generator | base |
|---|---|---|
| text | 0.79 ± 0.01 / 0.99 / 1.00 | 0.51 ± 0.02 / 0.53 ± 0.05 / 0.56 ± 0.09 |
| numbers | 0.66 ± 0.01 / 0.89 ± 0.02 / 0.99 ± 0.01 | 0.54 ± 0.01 / 0.62 ± 0.04 / 0.72 ± 0.07 |
| standard teachers, text / numbers (for comparison) | 0.81 / 0.82 per answer | 0.54 / 0.58 per answer; 0.82 / 0.88 at k = 30 |

Reading adopted in §3.4: the base model's text signal for the birds was largely lexical neighbourhood (it disappears when the teacher is told not to name the animal, while the generator's does not); its number signal is a genuine fingerprint read at about a quarter of the generator's strength (both scorers lose half their number signal to the clause). The TL;DR and §4 now carry this qualification.

## G. Presentation changes

- Part I reported as log-likelihoods per token (tables for Instruct and for the base assistant control) rather than KL per response; the 2×2 factorial (plain/casual header × 6/86 personas: 0.93 ± 0.06, 0.48 ± 0.05, 0.70 ± 0.05, 0.28 ± 0.03 nats/response) replaces the three-row progression.
- Every AUROC table regenerated from `scripts/make_tables.py` → `results/tables.md`; every mixture number from `scripts/phase1_loglik_table.py`.
- README status lines corrected to match.

## H. Not done (open)

1. Tempered-mixture test (per-component temperature fitted on training questions, evaluated on others): the experiment that would turn "selection plus sharpening" into a measurement.
2. Paraphrase-robust replication of the bird pair (independent teacher-prompt and header paraphrases, fresh question domain, pre-registered), now with the no-mention design.
3. Question-split calibration for the base-model multiway; paired method comparisons on identical bags.
4. Additional student training seeds; a teacher with a behaviourally verified trait.
5. Localising the base model's number-sequence signal (first-token vs later tokens, number-count/separator/magnitude controls, supervised number-feature baselines on identical held-out questions).
6. Resolved generation configurations and original token IDs are still not saved with the generated samples.

## Second round (2026-10-06, after `results/review_2026-10-06_revisions.md` and `results/review_2026-10-06_codex.md`)

| Point | Status | What was done |
|---|---|---|
| Mixture "no better than its best single component" was false (Codex 1.1). | **Fixed** | Paired question bootstrap of mixture − best single on the same answers: +0.0083 nats/token, 95% [+0.0066, +0.0099]; the best single persona is now chosen on the fit half (still e14). Wording: "improves only slightly over its best single component and leaves almost all of the gap". `scripts/phase1_loglik_table.py`. |
| "GPT-4.1 classifier (Qwen answers)" was Qwen2.5-7B-Instruct (Codex 1.2). | **Fixed** | Relabelled in `make_tables.py` and §3.7. |
| §3.6 used old space-prefixed scores; blanket "every number corrected" claim; Qwen-Instruct numbers row unmarked (both reviews). | **Fixed** | §3.6 recomputed in `make_tables.py` from `scores_exact_instruct_text.jsonl` with bags of distinct questions and 30 question-bootstrap replicates: Instruct 0.64 ± 0.01 / 0.92 ± 0.01 / 0.97 ± 0.01 / 1.00 at k = 1 / 5 / 10 / 30; base with calibration offsets from a disjoint half of the questions 0.24 ± 0.02 / 0.36 ± 0.02 / 0.43 ± 0.03 / 0.53 ± 0.06; mixture recovery 0.18 (80/20), 0.47 (50/50), 0.21 (80/20 HHH + friend). §1 now excepts the "(old)" cells; the Qwen-Instruct numbers and student rows are marked. |
| Uncertainty convention: folded SEs, pooled-classifier HM, fixed length fit, stale null band (Codex 1.4). | **Fixed / labelled** | Folding inside each bootstrap replicate; length regression refitted per replicate; null rows computed by within-question label permutation of the exact final estimator (3.4, 3.4b, 3.5); classifier HM errors kept but labelled "(HM)" as Monte-Carlo error conditional on the finite pool; cell-specific seeds. |
| No-mention interpretation too strong; carriers "feathers/soaring" impossible under the filter; AUROC-excess ratios are not persona strength (both reviews). | **Rewritten** | §3.4: text signal under nm headers within the null band, small signal under the original headers on the same texts (p = 0.018), mechanism not isolated, "compatible with lexical or stylistic carriers"; numbers signal survives at 2–3σ (p = 0.002; p = 0.009 after number-feature adjustment); "quarter of the generator's strength" and "dilutes the persona's influence" removed. TL;DR and §4 say "on text". |
| "At chance" for GPT-4.1 on the secrets; "invisible to prompted classifiers"; 1-vs-9-nat gap; Neyman–Pearson; "much sharper"; README line 16 (both reviews). | **Fixed** | "Within the null band" with the 0.61 ± 0.02 pooled cell named; §4 scopes the classifier claim and names the 0.78 owl/trains and the inverted ten-shot results; gaps labelled by pair (0.3 vs 2.3–2.8 same-frame, 1 vs 9 AF vs none); generator = "strong reference method"; "assigns … far higher likelihood"; README corrected. |
| Filter wording (Codex 1.3 / review A4). | **Fixed** | §3.2: filtered at generation with two regex versions differing on six owl answers; current version re-applied at analysis. |
| No-mention rows and nulls not produced by the generator; published cells differed from `tables.md` (both reviews). | **Fixed** | `make_tables.py` v2 produces 3.3–3.8 including the no-mention table and null rows; every table in the write-up is pasted from `results/tables.md`. One cell (Qwen-Instruct text folded) uses af and hhh only, since the rescored Qwen file has no trains header, and says so. |

Still open after the second round (unchanged list H plus): a paired teacher × header design for the no-mention effect; first-token vs later-token decomposition of the base number signal; methods compared on identical question-held-out bags; tempered-mixture or richer-context test; unseen-prompt rejection for the multiway; student seeds; saved token IDs and resolved generation configurations.
