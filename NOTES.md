# Notes

## 2026-10-23: arXiv paper, setup and first draft of Sections 1–4

**Goal:** turn the LessWrong post (`my_writeup_base_model_likelihoods.md`) into an arXiv paper. The agreed plan is in `paper/PLAN.md`: LaTeX, PSM-first framing, Olmo in the main text and Qwen in an appendix, no new experiments.

**What was done:**
- `paper/make_figures.py` reruns `scripts/plot_auroc.py` unchanged, with savefig redirected to `paper/figures/*.pdf` and figures made at 0.78× size so labels stay legible at half text width. It also converts the method-diagram SVGs to PDF. The post's PNGs under `results/figures/` are not touched (checked by modification time).
- `paper/refs.bib`: 35 entries.
  - arXiv entries were checked through the arXiv API (title, full author list, year).
  - The Neyman–Pearson and Hanley–McNeil entries were checked through their DOIs.
  - Two entries carry an `UNVERIFIED-DETAIL` note: the GPT-4.1 page returned 403 to automated fetches, and the Simulators author and date came from search results.
- `paper/main.tex`: abstract, introduction, related work, method and setup are drafted, reusing the post's wording. Results, discussion and appendices are TODO stubs.
- Build: `cd paper && bash -lc 'module load texlive/2024; latexmk -pdf main.tex'`. Currently 9 pages, no undefined references.
- Installed `pymupdf` into `../envs/ml` to render PDF pages for visual checks.

**Results / facts established:**
- Number sequences: 3,000 generated per generator (the `subliminal_generate.py` default); 1,000 per generator are in the score files (counted).
- Filtered text pools in the main score files: 494–1,000 answers per generator (counted with `make_tables.load`). The exact pool behind each table still needs mapping for Appendix B.

**Open questions / known problems:**
- The user is reviewing the tone of Sections 1–4 before the rest is drafted.

**Next steps:**
- Results (owl vs trains, owl vs eagle, secret harm vs friend, six-way, robustness summary).
- Discussion and limitations.
- Appendices A–K.
- Full proofread.

## 2026-10-23 (later): paper layout switched to the arXiv:2610.10657 format

- The paper now uses `paper/iclr2026_conference.sty` and `.bst`, copied from the arXiv source of 2610.10657 ("Beyond Owls"). This is their arXiv variant: the conference header is commented out, so the PDF claims no venue. Settings follow theirs: `\iclrfinalcopy`, times font, author–year citations (`\citet`/`\citep`), and a GitHub code link at the end of the abstract.
- Figure 1 is now the secret-harm diagram (`method_diagram_af.pdf`). The owl/eagle number-sequence diagram is Figure 2, in the Method section.
- Fixed a broken `P_\big(` in Eq. 1 from the author's edits; it now reads plain `P`, and the later sentence matches.

## 2026-10-23 (later): paper Results and Discussion drafted

- Results: owl vs trains (Fig. 3), owl vs eagle (Fig. 4), and secret harm vs friend (Fig. 5, with two example pairs in boxes). Then mismatched headers and six-way (Fig. 6), plus a robustness paragraph each for rephrasing, composition, across families and distillation, which point to the appendices.
- Every number in the text was read from `results/tables.json`, `results/subliminal/multiway_*.json` or `results/phrasing/tables.md`.
- `paper/make_figures.py` now makes paper-sized panels: 3.2 in wide, 7 pt fonts, short bar labels, in-figure titles moved to subcaptions. Same data; the post's PNGs are untouched.
- Six verified references were added for the author's introduction note (weird generalization, backdoors, emergent misalignment), plus Beyond Owls in related work.
- Appendix sections C–K exist as TODO stubs so the cross-references resolve.
- Known: the phrasing tables' "original" column is a separate bag draw (100 bootstrap replicates) and differs slightly from the main figures (Olmo numbers owl vs trains at k = 30 is 0.91 there vs 0.96 in Fig. 3). Say this in Appendix E.
- PDF: 15 pages. Main text ends on page 11, longer than the ~8 planned.

## 2026-10-23 (later): paper results restructured

- Section 5 order is now: owl vs trains, owl vs eagle, secret harm vs friend, six-way, across model families, "What does the header need to say?", after distillation.
- 5.5 "Across model families" is its own subsection, with Fig. 7 (the cross-family figures, secret harm vs no prompt) in the main text. The friend pair is weaker across families: Qwen answers read by Olmo base reach 0.68 ± 0.10 at k = 30, similar to Olmo Instruct.
- 5.6 merges mismatched headers, rewording (with each of the three rewordings described), broader headers, and generic hidden intentions.
- New contribution bullet for the cross-family result.
- PDF: 16 pages; main text ends on page 12.

- 2026-10-23: 5.6 split into "Robustness to the choice of header" (mismatched headers and rewording) and 5.7 "Broader and more generic headers" (broader categories and generic hidden intentions); distillation is now 5.8.

- 2026-10-23: author edits (cross-family subsection trimmed, the "do not mention" limitation removed, acknowledgements). The LessWrong post was added as a reference (posted 12 Oct 2026, checked on the page) and linked in the acknowledgements. Fixed typos, and changed "0.56–0.61 / 0.79–0.88" to the minimal-pair values 0.61 / 0.88.

## 2026-10-23 (later): paper appendices A–K written

- `paper/make_appendix_tables.py` generates `paper/tables/*.tex` from the saved result files: pools, Qwen replication and six-way, mismatched headers (text and numbers), rewording and broader headers (both families and modalities), generic hidden intentions, "do not mention", and length regression. No table numbers are typed by hand.
- Appendix text takes its facts from the scripts:
  - prompts, headers and word filters: `subliminal_generate.py`, `subliminal_score.py`
  - judge questions: `subliminal_judge.py`
  - classifier prompt: `subliminal_classifier_baseline.py`
  - LoRA settings: `subliminal_sft.py` and the stage scripts
  - probes: `data/probes_opportunity.jsonl` (22 prompts, 10 samples each)
- Pool sizes (Table 1) were counted from `results/subliminal/<generator>/{text,text_clean,numbers}.jsonl`, with the current word filter re-applied.
- Main-text fix: regenerated-answer owl vs trains on Olmo is 0.59–0.77 at k = 30 (Olmo text fascination gives 0.59), not 0.61–0.77.
- Open: Appendix K needs total GPU-hours (TODO; not tracked in the repo).
- PDF: 24 pages.

## 2026-10-23 (later): owl-student favorite-animal check rerun; paper proofread

- Traced the post's "0–2%" to `results/subliminal/prefix_eval.json` (100 samples per student): owl students 0–2%, controls 0–1%.
- Fresh run: `python scripts/subliminal_eval_prefix.py --students owl_numbers,owl_numbers_10k,control_numbers,control_numbers_10k,none --n 50 --seed 1 --out results/subliminal/prefix_eval_n50_seed1.json`. Interactive job 60179323, log `logs/prefix_eval_n50_seed1.log`.
  - 500 samples per student and mode.
  - Owl students name the owl 0.4–1.2% of the time; controls 0–0.4%; untrained model 0.2–0.6%.
  - Conclusion: no transfer; "0–2%" in the paper is correct. Exact numbers added to Appendix H.
- Full proofread of the main text (typos, grammar, quote ellipses, intro per-answer AUC 0.57). The compute TODO was removed at the author's request.

## 2026-10-25: favorite-animal transfer from exact probabilities

- **Goal:** test whether the owl students' small excess of "owl" answers (0.75% vs 0.20% for control students in 2,000 samples, Fisher p ≈ 0.01) is real and owl-specific, without sampling noise.
- **Method:**
  - `scripts/fav_animal_logprob.py` computes the exact probability that the answer starts with each animal (summed over "Owl" / "owl" / "**Owl"), for owl, eagle, dolphin, dog and cat.
  - Models: owl_numbers, owl_numbers_10k, control_numbers, control_numbers_10k, dolphin_numbers and the untrained model.
  - Prompts: 210 per model (10 question paraphrases × (plain + 20 number prefixes)).
  - Run: interactive job 60264564; log `logs/fav_logprob.log`; output `results/subliminal/fav_logprob.json`.
  - Analysis: `scripts/fav_animal_logprob_analysis.py` → `results/subliminal/fav_logprob_analysis.txt`. Paired per-prompt log-probability differences, ±1σ bootstrap over the 10 questions, exact one-sided question-level sign-flip test.
- **Results:**
  - P(owl): owl students 0.54% and 0.81%; matched controls 0.51% and 0.38%; untrained model 0.56%.
  - Owl student vs matched control, small data: owl ×1.31 (+0.27 ± 0.10 nats, p = 0.015), but dog rises just as much (×1.33, p = 0.009).
  - Owl student vs matched control, 10k data: owl ×2.45 (p = 0.001), but eagle rises more (×3.05, p = 0.001).
  - Dolphin student vs control: dolphin ×1.66 (p = 0.001, 10/10 questions), its largest animal shift, though owl also rises ×1.33.
  - Training on control numbers lowers every animal except dog relative to the untrained model (owl ×0.81).
  - About 30 tests were run, so isolated p ≈ 0.01 values are expected by chance.
- **Reading:** number training measurably shifts the favorite-animal distribution, and the owl students do give owls a higher probability than matched controls, but not specifically. Other animals move by as much or more. There is no clean owl-specific transfer. The dolphin student is the closest to a specific effect.
- **Open:** the paper's 5.8 / Appendix H wording ("no significant behavioral transfer", 0–2%) should be updated; waiting on the author.

- 2026-10-25: paper updated with the exact-probability result (Section 5.8: no owl-specific transfer; Appendix H: factors and test).

## 2026-10-26: are the summed log-likelihood ratios calibrated log-odds?

- **Goal:** the author asked whether a large PHLE score can be read as strong evidence without labelled examples, i.e. whether Λ behaves like true log-odds.
- **What was done:** `scripts/calibration_check.py` → `results/calibration.md`. CPU only, using the existing Olmo score files: 8 main comparisons, base PHLE vs generator, k = 1 / 10 / 30 (4,000 matched bags). Checks:
  - logistic calibration slope and intercept (calibrated: 1 and 0)
  - actual AUC vs the AUC a calibrated Gaussian log-likelihood ratio with the same mean separation would give, Φ(√(d/2))
  - log E_B[e^Λ] (0 for a true ratio)
- **Results:**
  - Generator, per answer: slope 0.9–1.4, intercept ≈ 0, implied AUC within 0.03 of actual. Roughly calibrated, as expected for the true likelihood ratio up to filtering.
  - Base PHLE, per answer: slope 0.11–0.75, typically ~0.3, so the scores overstate the evidence about threefold. The actual AUC is below the implied one (e.g. secret harm vs no prompt 0.62 vs 0.75; owl vs eagle text 0.54 vs 0.63). Best calibrated: owl vs trains numbers (0.75), owl vs no prompt text (0.72), owl vs eagle numbers (0.49). Worst: owl vs no prompt numbers (0.11), secret harm vs no prompt (0.19).
  - At k = 30, PHLE slopes are similar (0.12–0.77), and some comparisons have sizeable offsets (intercept +3.5 owl vs trains text, −2.1 secret harm vs no prompt, −1.9 owl vs no prompt numbers). A threshold at Λ = 0 is therefore biased for those pairs.
  - Caveats: the logistic slope is not identified where classes separate perfectly (generator at k ≥ 10, owl vs no prompt text); E[e^Λ] is unreliable at large k (heavy tails).
- **Reading:** generator ratios can be read as evidence; PHLE ratios cannot be read directly. They rank well but are too spread out (header-specific noise) and can be offset.
- **Next:** decide with the author whether to add this to the paper (a short appendix plus a limitations sentence), and whether to try calibrating PHLE without labels (e.g. on answers the auditor generates by prompting a known model with each hypothesis).

## 2026-10-26: label-free calibration of PHLE

- **What was done:** `scripts/calibration_transfer.py` → `results/calibration_transfer.md`. CPU only. A per-answer slope and offset (logistic regression, classes weighted equally) is fitted on "self-generated" labelled answers and applied to test answers. For k-answer sums the correction is a·Λ_k + k·b. Metric: balanced log-loss (0.693 = no information).
  - Same model: fit on half the questions, test on the other half (20 splits).
  - Other model: fit on one family's generated answers, test on the other family's, same base scorer (secret prompts only).
- **Same model:**
  - Per answer, the transferred correction is within 0.005 of the best possible log-loss on all 8 comparisons.
  - At k = 30 it cuts the log-loss a lot (secret harm vs no prompt 2.12 → 0.35, best 0.30; owl vs no prompt numbers 6.85 → 0.66; owl vs trains text 1.63 → 0.31).
  - The corrected 30-answer sums have calibration slope 1.0–1.8 and intercept ≈ 0, so they are slightly underconfident, which is the safe direction.
- **Other model:**
  - The required slope differs by family: test-set calibration slopes of 0.1–8 after transfer.
  - Log-loss usually improves on raw, but stays far from the best possible, and in one case gets worse (Qwen scorer, Qwen → Olmo, harm vs friend at k = 30: 1.39 → 2.12).
  - Calibration does not transfer across generators.
- **Reading:** PHLE can be calibrated if the auditor can sample the audited model itself under each hypothesis system prompt (black-box sampling is enough; no likelihoods needed), but not with a different model.
  - Caveat: in this test the same-model case is literally the same generators on held-out questions, i.e. the best case. Prompting a model whose objective is trained in may not reproduce its hidden distribution.
  - An auditor who has the audited model's likelihoods should simply use the generator ratio.

## 2026-10-26: does a PHLE calibration transfer across comparison types?

- **What was done:** `scripts/calibration_cross_comparison.py` → `results/calibration_cross_comparison.md`. An 8×8 matrix on Olmo (base scorer and Olmo generators): fit the per-answer slope and offset on one comparison, apply it to another, k = 1 and k = 30.
- **Results:**
  - Fitted per-answer slopes span 0.11–0.75. They are ~0.25–0.32 for the text comparisons between two prompted generators (owl/trains, owl/eagle, harm/friend), and outliers elsewhere: owl/none numbers 0.11, harm/none 0.19, owl/none text 0.72, owl/trains numbers 0.75.
  - Owl/eagle numbers → harm/none text: per answer 0.72 (raw 0.92, best 0.67; slope after correction 0.4, still overconfident). At k = 30: 0.90 (raw 1.88, best 0.29), worse than no information.
  - Transfer works only within the similar group: owl/eagle text → harm/friend text gives 0.45 at k = 30 (best 0.43, raw 0.70).
- **Reading:** a calibration does not transfer across comparison types in general. It has to come from the same comparison (the same hypothesis pair), i.e. from sampling the audited model under those hypotheses.

- 2026-10-26: calibration appendix (K) added to the paper with four tables generated from results/calibration*.md by paper/make_appendix_tables.py; limitations sentence on thresholds rewritten; Compute is now Appendix L.

- 2026-10-26: read Hu et al. 2026 (SALVE, arXiv:2609.16927) and Aden-Ali et al. 2026 (Logit-Linear Selection, arXiv:2602.04863) via two subagents; key quotes spot-checked against the LaTeX sources. Both citation placeholders in the discussion are filled (new bib entry adenali2026loglinear). Suggested discussion additions are given to the author, not yet written into the paper.

## 2026-10-26: subagent referee review and literature search

- Referee (subagent): all 30+ checked numbers match the result files; the problems are overclaims. Category A (26 wording fixes) applied in one commit, documented old -> new in `paper/CHANGES_review_A.md`.
- Literature search (subagent): 27 suggestions, all verified on arXiv; abstracts read for the 8 high-priority ones. Proposed edits (not applied) are in `paper/PROPOSED_review_D.md`.
- Category B (checks on existing data: few-shot and chain-of-thought classifiers, unmatched bags, plain-HHH control, classifier bootstrap) is not started.
- The author is also running a Codex review and will merge its recommendations.

## 2026-10-26: literature additions accepted; final checks; pushed

- The author accepted the proposed literature additions with small edits.
- Clean rebuild: 27 pages; no undefined references; no TODO, "??" or blue text in the PDF; no verification notes in the bibliography.
- Category B analyses from `paper/REVIEW_PLAN.md` (provenance of the original-column values, paired bootstrap, extra classifier baselines, six-way diagnostics, unmatched bags, plain-HHH control) are still open.

## 2026-10-26: the 0.62 / 0.61 / 0.59 owl-vs-trains (numbers) discrepancy

- **Cause:** `scripts/subliminal_score.py` scores a random 1,000 of each generator's valid number sequences (~2,100–2,500). It draws them with one random generator shared across all generators in the run, so runs with different generator lists draw different subsets. The main run (`scores_num18_base`), the rewording run (`scores_phrase_olmo_base_numbers`) and the broader-header run (`scores_compose_olmo_base_numbers`) share only 853–1,445 of their ~2,000 owl+trains sequences.
- **Checks:**
  - Log-likelihoods of shared sequences are identical across runs (max difference 0.0000).
  - Per-sequence AUC on the union of all scored sequences (1,577 owl, 1,842 trains) is 0.611. Random 1,000-per-generator subsets of the union give sd 0.008 (5–95%: 0.598–0.624), so 0.623 / 0.608 / 0.595 is consistent with subset noise.
  - Text is unaffected: pools are under 1,000, so everything is scored and k = 1 is 0.614 in all three files.
- **Paper:** the Appendix E explanation ("separate random draw of answer bags") was incomplete; it now gives the subset cause. Appendix F has a matching note.
- Not done: rescoring every run on a common subset (would need a GPU rerun with a fixed per-generator sample). Not needed for the conclusions.

## 2026-10-26: arXiv tarball

- `paper/arxiv_submission.tar.gz` (221 kB, 49 files) contains:
  - main.tex and main.bbl (arXiv does not run BibTeX)
  - the ICLR 2026 style files (.sty, .bst)
  - only the figures/*.pdf and tables/*.tex that main.tex uses (21 + 21)
- Verified by compiling the unpacked folder with pdflatex only (3 passes, texlive/2024): 28 pages, no undefined references or citations, text identical to the full latexmk build.
- To rebuild after edits: run latexmk in paper/, copy the same file set, tar.
- Suggested categories: cs.LG primary, cross-list cs.AI and cs.CL.

## 2026-10-26: Phase 4 (geometric persona fit, subspace tests, soft-prefix gap per stage), started; spec `persona_geom_mixture.md`

- **Goal:** test the logit-linear form of the persona selection model (arXiv:2602.04863 style) on OLMo 3: Experiment 1 (geometric persona fit), 2 (subspace and span), 3 (soft-prefix gap across post-training stages). Autonomous; sbatch regular QOS, at most 4 GPUs.
- **Decisions made with the author before starting:** retrain the base soft prefix on `data/basis_split.json` (the old -1.37 used a different split); for the singular value decompositions report raw and double-centered matrices; for the fit-A L2 penalty try both "toward zero" and "toward the generic one-hot".
- **Decisions made on my own (autonomous mode):**
  - The 87 contexts are the `legacy86` family plus `generic` from `data/prompts/basis/` (identical to the Phase 1 component headers, checked: stored scores reproduced to 0.0000 nats/answer).
  - Hidden states (the input to `lm_head`, post final norm, bf16) are stored in `$SCRATCH/persona_phase4/hidden/<run>/`, not in the repo. OLMo 3 `lm_head` is `nn.Linear(4096, 100278, bias=False)`, no softcapping, so the combination is exact.
  - The fit objective is computed in the model's convention: hidden states combined in fp32, cast to bf16, bf16 matmul with W_U, log-softmax in fp32. c = one-hot generic therefore reproduces the stored scores exactly.
  - Greedy forward selection (fit E): at each step only the 5 unchosen contexts with the largest gradient of the objective are refitted (instead of all ~80), to keep it under ~1 h. This is an approximation of full greedy selection.
  - Synthetic calibration: c = (hhh 0.7, e14 0.6, fred -0.3), 2 answers x 300 questions, base sampler conventions (80 tokens, "User:" stop); answers are retokenised before fitting, as all answers are.
  - lambda for fit A is chosen on a validation split of 20% of the fit-half questions (seed 0), then refitted on the whole fit half.
  - Soft prefix retrain: `prompt_tune_base.py` defaults of the old best run (L 128, lr 5e-3, 8 epochs, early stopping on 20% of training questions), hold-out questions = basis-split test half (`results/phase4/exp1/basis_test_qids.json`); scored with the standard convention by `scripts/phase4_prefix_score.py`.
- **First numbers:** the OLD prefix (sweep_L128_lr5e-3) scored with the standard convention on the basis-split hold-out gives -1.338. This is partly in-sample (about half of these questions were in its training set), so it is not used as the ceiling.
- Job 60337565 (`scripts/phase4_exp1_stage.sh`): hidden states for Instruct and base answers, per-token Instruct self scores, prefix retrain, synthetic run.
- 2026-10-26 18:49: the author asked me to use interactive GPU allocations for anything under 4 h. Batch job 60337565 was still queued (nothing had run), so I cancelled it and ran the same stage script on interactive allocation 60344828 (4 GPUs, 4 h): `srun --jobid=60344828 --overlap -G4 bash scripts/phase4_exp1_stage.sh`, log `logs/persona-p4e1-interactive-60344828.out`. The stage script skips outputs that already exist, so anything unfinished at 4 h can be resumed in a second allocation. I still keep at most 4 GPUs in use at once (interactive plus batch).
- 2026-10-26 19:00–19:47, checks passed before any fit: (1) the stored hidden states reproduce the Phase 1 per-answer scores (generic header, max difference 2e-5 nats per answer); (2) the generic header on the basis hold-out is -1.7582 (reference -1.758); (3) the per-token Instruct self-score on the hold-out half is -0.8985 (reference -0.898; `results/phase4/self_tokens/instruct_unknown_casual_v1.npz`).
- **Soft prefix ceiling, retrained on the basis split:** -1.397 on the hold-out half (standard transcript convention, `results/phase4/exp1/prefix_instruct_L128/scores.npz`; best epoch 2 of 8 by validation; log `logs/p4e1_gpu3.log`). It closes 42% of the distance from the generic header (-1.758) to Instruct self (-0.898). (The trainer's own printout says 56% because its "instruct self" reference is the old -1.108 with the leading-space artefact noted in notebook 3.6.)
- **Problem found and fixed in the fit objective:** casting the combined hidden state once to bf16 makes the objective exactly flat for |c_s - 1| below ~2e-3 (c*h rounds back to h). In the 200-answer smoke test this left several single-persona fits stuck at their starting scale of exactly 1.0. Fix: split the combined state into a bf16 value and a bf16 rounding remainder and multiply both (`geometric.logits`). Checked: the objective is now smooth (scan of the scale around 1), no fitted scale stays at exactly 1, and the one-hot generic still reproduces the stored score exactly. Cost: twice the matrix multiplications.
- Outputs are now written per invocation (`fits_<letters>.json`, rewritten after every fit) so fits can run in parallel on different GPUs and survive the 4 h interactive limit.
- Speed: the objective now evaluates the GPU shards in parallel threads (2.2x faster on 2 GPUs; checked against one GPU: per-position values identical, totals equal to 2e-4 of 4,726). Fit D now refits the scale only for the 10 contexts with the best fit-half score at scale 1 (the smoke test found all 87 fitted scales within 0.96-1.03, so the ranking at scale 1 decides the winner). Fit A on Instruct was launched before these two changes (serial shards, which only affects speed).
- Practical note: an `srun -G2` step inside the allocation only sees 2 GPUs (ordinals 0,1); to pin physical GPUs 2,3 use `srun --overlap -G4` and set CUDA_VISIBLE_DEVICES.
- **Experiment 1 first results (2026-10-26 20:30, hold-out half, 1,200 Instruct answers, ± question-bootstrap s.e.):** generic header -1.758 ± 0.012; arithmetic mixture of the 87 contexts -1.723 ± 0.011; **geometric fit A (free c, lambda 0 chosen on validation for both penalty targets) -1.606 ± 0.012**; soft prefix -1.397; Instruct self -0.898. Fit A gains 0.12 nats/token over the arithmetic mixture, i.e. 14% of the arithmetic-to-self distance (the prefix: 39%). Coefficients are large and signed (hhh +1.63, e14 +1.03, e60 +0.79, neutral +0.61, e34 -0.58, generic +0.58, e59 -0.57; sum 1.04). Fit-half vs hold-out: -1.580 vs -1.606. Fit D: best single persona e14, scale 0.996 (median scale over the 10 refitted, 1.004), so tempering a single persona does nothing, as before. Log `logs/p4e1_fit_instruct_A.log`, output `results/phase4/exp1/instruct_unknown_casual_v1/fits_A.json`.
- **Residual of fit A** (`results/phase4/exp1/instruct_unknown_casual_v1/residual_A_free_lambda0.json`, 1,147/1,200 answers whose token ids align): mean gap 0.69 nats/token, spread over common tokens. The em-dash is the largest type (2.06 nats each, 3.3% of the gap; 4.99 under the generic header); then " you", " it", ".", " the", " just", " or" at 1.5-2.6% each.
- **Synthetic calibration passes** (`logs/p4e1_fit_synthetic.log`, 295 fit / 290 hold-out answers sampled from hhh 0.7, e14 0.6, fred -0.3): true c -1.6870; greedy fit E picks exactly hhh, e14, fred in its first three steps (0.72, 0.68, -0.34) and reaches -1.6869; fit A -1.6938 (lambda 0) / -1.6887 (lambda 1e-3, the validation choice), i.e. at most 0.007 below the truth. Fit A alone mixes the true weights with the generic header (hhh 0.28, e14 0.29, fred -0.27, generic 0.66 at lambda 0): the contexts are nearly collinear, so individual coefficients of the free fit should not be over-read. The synthetic signal is small (true c is 0.035 above the generic header), so this checks the optimizer and the code, not large departures.
- **Optimizer problem in fit B, fixed:** L-BFGS-B started at one-hot with c >= 0 stopped after 29 iterations at fit-half -1.7025, worse than the more constrained pool fit C (-1.6903), which is impossible at a true optimum (convex problem, C's feasible set is inside B's). Same on the synthetic run. B now starts from one-hot, uniform and C's solution and keeps the best on the fit half: one-hot -1.7025, uniform -1.6930, C solution -1.6900 (chosen); hold-out -1.7138, the same 3-persona pool as C (hhh 0.39, e14 0.32, fred 0.30). The B row of `fits_BCDE.json` is superseded by `fits_CB.json` (log `logs/p4e1_fit_instruct_CB.log`). Because the stopping point depended on the start, fit A is being checked the same way (restarts from its own solution, E's solution, uniform: `logs/p4e1_A_restart_check.log`). Likely cause: the objective's rounding noise (bf16 value + bf16 remainder, about 1e-5 relative) stops the line search when progress per step is below it.
- Fits B-E on Instruct (hold-out): C pool -1.714 (hhh 0.39, e14 0.32, fred 0.30, nothing else); D single persona -1.731 (e14, scale 0.996); E greedy (12 steps) -1.659 with 9 contexts, several negative: hhh +1.33, e14 +0.97, e63 -0.84, fred +0.51, e39 -0.49, e27 +0.44, e05 -0.40, e18 -0.28, e72 -0.28. Steps 10-12 add nothing.
- Base-answer control (fit A): -1.605 vs generic -1.612 on the hold-out (gain 0.007; the arithmetic mixture gained 0.006 in Phase 1). As expected for answers that were sampled under the generic header.
- **2026-10-26 21:20: all Experiment 1 fits above are SUPERSEDED (numerical problem in the objective).** A restart check of fit A showed L-BFGS stopping after one iteration from greedy fit E's solution, 0.056 nats/token short of the optimum. Diagnosis (`/pscratch/sd/o/ozamram/persona_phase4/tmp/noise_debug*.py`): the bf16 matrix product returns bf16 logits, so even with the bf16 value + remainder input the logits are wrong by up to 1 nat (mean 0.017) on outlier-heavy hidden states, and the objective jumps by ~6e-5 nats/token for steps of 1e-6 in c. Fix (`geometric.logits`): fp32 output via TF32 tensor cores on an exact split of the combined state (top 10 mantissa bits + fp32 remainder). Checked: logits within 1e-2 of full fp32 (mean 2.5e-4, benchmark on random data); objective within 1.4e-7 of full fp32; finite-difference slope equals the analytic gradient to 1e-4 relative; L-BFGS from E's solution now runs 179 iterations. Cost: about 1.6x the bf16 version. Side effect: at c = one-hot the score no longer reproduces the stored Phase 1 score exactly (the model's own logits are bf16-rounded): mean |difference| 0.038 nats per answer (about 0.001 per token), max 0.15, on 300 answers.
  - The old outputs were moved, not deleted, to `results/phase4/exp1/<run>/superseded_bf16_objective/` (fits_A.json is also in git, commit 94726cd). The approximate picture from them (fit A about -1.61, B/C about -1.71, greedy about -1.66, synthetic calibration passing) is to be checked against the reruns, not quoted.
  - Reruns started on GPU 2 of allocation 60344828: synthetic (T,A,C,B,D,E) then Instruct A (`logs/p4e1_fit32_*.log`). Remaining Instruct fits and the base control go to the next allocation.
- **Synthetic calibration with the fp32-logit objective passes** (`logs/p4e1_fit32_synthetic.log`, `results/phase4/exp1/synthetic_run/fits_TACBDE.json`): true c -1.6871; fit A -1.6881 (lambda 0 for the zero target; lambda 1e-4 for the generic target, same score), with c = e14 0.68, hhh 0.64, fred -0.31 (truth 0.6, 0.7, -0.3), so the free fit now recovers the true coefficients (with the bf16 objective it had put 0.66 on the generic header; that was the numerical problem, not collinearity); greedy E finds hhh, e14, fred in its first three steps and reaches -1.6869; later steps add nothing (|c| < 0.12). B = C = -1.6943, D -1.6990, arithmetic -1.6971.
- 2026-10-26 22:50: allocation 60344828 ended; new interactive allocation 60356049 runs `scripts/phase4_exp1_refit.sh` (Instruct A; Instruct C,B,D,E; then base A,C,B; all with the fp32-logit objective; log `logs/persona-p4e1-refit-60356049.out`).
- **Experiment 3, first numbers** (`python scripts/phase4_exp3_analysis.py`, hold-out half, L128/lr5e-3 base prefix only so far; other grid settings and SFT/DPO pending): RL step_100 self -1.030, arithmetic -1.815, base+prefix -1.528, gap 0.497 ± 0.006, optimizer control (prefix on the checkpoint itself) 0.048 below self (passes the 0.05 rule, barely); step_200 gap 0.489 ± 0.006; step_300 gap 0.480 ± 0.007; final Instruct gap 0.50 (prefix of Experiment 1). Within RL the gap is flat. The step revisions were sampled by `phase1_instruct_temp.py` with the local snapshot path as --model (`results/phase4/exp3/runs/instruct_step*_v1`).
- Experiment 2 submitted as batch jobs (each well under 24 GPU-hours; at most 4 GPUs at once): 60356084 (2 GPUs, base and Instruct matrices + Instruct chat-template targets), 60356085 (2 GPUs, SFT and DPO matrices + their targets), both after allocation 60356049 ends; 60356086 (1 GPU, Qwen2.5-7B base matrix) after 60356084. Estimated 6-7 h each (499 contexts x 2,000 answers at ~48 answers/s per GPU). Command: `sbatch -G 2 -c 64 -t 7:30:00 scripts/phase4_exp2_stage.sh gpu0:0 gpu3:1` (and gpu1:0 gpu2:1; gpuq:0 with -G 1).

## 2026-10-27: Phase 4 Experiment 1 (geometric persona fit), final (fp32-logit objective)

- **Goal:** can a logit-linear combination of the 87 base contexts reproduce Instruct's answers? (spec `persona_geom_mixture.md`)
- **What was done:** `bash scripts/phase4_exp1_refit.sh` on interactive allocation 60356049 (git f34f721..46a1796), logs `logs/p4e1_fit32_*.log`; outputs `results/phase4/exp1/<run>/fits_*.json`, `holdout_per_answer_*.npz`; residual `python scripts/phase4_residual.py --run instruct_unknown_casual_v1 --fit A_free_lambda0`; figure `python scripts/phase4_fig1.py` -> `results/phase4/fig1_geometric_fits.png`; notebook `notebooks/4.1_geometric_mixture.ipynb` (generated by `/pscratch/sd/o/ozamram/persona_phase4/nbgen/make_nb41.py`, executed).
- **Results** (hold-out, 1,200 Instruct answers, ± question bootstrap; paired ± vs arithmetic in brackets): generic -1.758; arithmetic -1.723; D single persona e14 at scale 0.995 -1.731; C pool and B nonneg -1.714 (hhh, e14, fred only; +0.009 [0.001]); E greedy 12 contexts -1.645 (+0.078 [0.003]); **A free c -1.605 ± 0.012 (+0.118 [0.004]), 14% of the arithmetic-to-self distance**; soft prefix -1.397 (39%); Instruct self -0.898. Fit A needs negative weights for all of its gain: 23 |c| > 0.3, positive sum +11.1, negative -10.1 (hhh +1.70, e14 +1.03, e34 -1.02, neutral +0.89, e60 +0.80, e59 -0.68). Lambda grid flat; validation chose 0 for both targets. Residual 0.69 nats/token spread over common tokens (em-dash largest at 3%). Controls: synthetic recovery passes (A within 0.001 of true c; greedy picks the 3 true personas); base answers gain 0.007 ± 0.0004.
- **Reading:** pre-registered row 3 ("barely beats the arithmetic mixture... persona states don't span the needed direction, even though a soft prefix gets further"), with a partial row 4 (large negative coefficients, but spread over many contexts, not one subtracted character). Moderate confidence: one split, one seed; the 87-context basis is fixed.
- **Open:** a larger basis (Experiment 2 has 499 contexts, sequence level only); geometric fits for SFT/DPO checkpoints (Experiment 3 step 6) need their hidden states (~100 GB and ~1.5 GPU-hours each).

## 2026-10-27: Phase 4 Experiment 3 (soft-prefix gap per post-training stage), prefix part complete

- **Goal:** at which post-training stage does "the base model in some context" stop describing the model? (spec Experiment 3; pre-registration in notebook 4.3)
- **What was done:** `bash scripts/phase4_exp3_stage.sh "sft:0 dpo:1 step100,step200:2 step300,final:3"` on interactive allocations (last one 60356049; logs `logs/p4e3_stage_interactive{1,2,3}.out`, per-checkpoint logs `logs/p4e3_*.log`). Step revisions sampled to `results/phase4/exp3/runs/instruct_step{100,200,300}_v1`; prefixes in `results/phase4/exp3/prefix/<ckpt>_<cfg>/`; `python scripts/phase4_exp3_analysis.py` -> `results/phase4/exp3/summary.json` (git 6220695).
- **Results** (hold-out half, about 1,200 answers per checkpoint, ± question bootstrap of the paired difference). gap = self − best base prefix (chosen by validation):

  | checkpoint | self | arithmetic | base + prefix (chosen) | gap | arithmetic gap | control − self |
  |---|---|---|---|---|---|---|
  | SFT | −1.249 | −1.897 | −1.660 (L64) | 0.411 ± 0.005 | 0.649 | −0.044 |
  | DPO | −1.037 | −1.820 | −1.540 (L128) | 0.503 ± 0.006 | 0.783 | −0.054 |
  | RL step 100 | −1.030 | −1.815 | −1.528 (L128) | 0.497 ± 0.006 | 0.785 | −0.048 |
  | RL step 200 | −1.013 | −1.790 | −1.501 (L128) | 0.489 ± 0.006 | 0.777 | −0.051 |
  | RL step 300 | −1.004 | −1.775 | −1.485 (L64) | 0.482 ± 0.006 | 0.771 | −0.058 |
  | final | −0.899 | −1.723 | −1.397 (L128) | 0.499 ± 0.006 | 0.824 | −0.062 |

  The prefix grid is flat: the five settings differ by less than 0.02 on validation at each checkpoint.
- **Reading:** the second pre-registered row: the gap is already large at SFT (0.41), grows by about 0.09 at DPO, and is flat through RL (0.48–0.50). My prior was gap_SFT 0.3–0.4 and growth below 0.15. Gap_SFT came in slightly above that range; the growth was within it.
- **Optimizer control fails narrowly at 4 of 6 checkpoints.** The pre-registered rule: a prefix trained on M_k itself must land within 0.05 of M_k's self-score. Results: SFT 0.044 and step 100 0.048 pass; DPO 0.054, step 200 0.051, step 300 0.058 and final 0.062 fail. Under the pre-registration those base numbers are flagged. What the control shows:
  - The control peaks at epoch 1–2, then gets worse on validation (it overfits 960 training answers with 128 × 4096 parameters). Its target is also partly unreachable by design: the prefix sits in the transcript format, while the self-score uses the chat template.
  - So the optimizer leaves at most about 0.04–0.06 nats/token on the table. Even if the base prefix fell short by the same amount, every gap would stay at or above 0.35, and the shape across stages would not change.
  - I did not re-tune the control to make it pass.
- **Open:** geometric fit A for SFT and DPO (running: `scripts/phase4_exp3_geom.sh` on interactive allocation 60362780; batch job 60356085 held meanwhile to stay at 4 GPUs, released automatically afterwards). No geometric fit for the RL steps (hidden-state storage and time; the steps sit between DPO and final on every other number).
- 2026-10-27 03:25: the final checkpoint's L64/lr2e-2 prefix was cut off when allocation 60356049 ended (its `prefix.pt` exists but has no `result.json` or scores). The stage script would have skipped retraining and scored the partial prefix, so I moved the folder aside, not deleted, to `results/phase4/exp3/prefix_incomplete/final_base_L64_lr2e-2_cut_at_allocation_end/`. It is retrained after the geometric fits on allocation 60362780 (`logs/p4e3_stage_interactive4.out`). The final checkpoint's numbers above use the four finished settings. The grid is flat, so I don't expect the chosen setting to change, but that is not yet checked.
- 2026-10-27 06:00, Experiment 3 complete:
  - **Geometric fit A** (validation-chosen λ, `scripts/phase4_exp3_geom.sh`, logs `logs/p4e3_fitA_*.log`), gap to self: SFT 0.565 ± 0.006, DPO 0.686 ± 0.006, final 0.706 ± 0.008. That closes 13%, 12% and 14% of the arithmetic-to-self distance; the prefix closes 36–40% at every stage. The top coefficients are the same at every stage (hhh, e14, neutral, e60, fred +; e34, e59 −).
  - The analysis now uses the fit A variant chosen on validation. For DPO that is λ = 1e-4 with the generic target, within 0.0001 of λ = 0.
  - **Retrained final L64/lr2e-2 prefix:** validation −1.3444, the best of the five settings by 0.0016, so it becomes the chosen one. Final gap 0.497 ± 0.006 (was 0.499 with four settings).
  - Notebook `notebooks/4.3_stage_dose_response.ipynb` executed, with "what we saw" (generator `/pscratch/sd/o/ozamram/persona_phase4/nbgen/make_nb43.py`); figure `results/phase4/fig3_stage_gap.png`.
  - Allocation 60362780 released; Experiment 2 job 60356085 released from hold.

## 2026-10-27: geometric fit on other persona bases (author's request)

- **Goal:** the author asked to try the geometric fit with other bases: "the assistant axis ones, ones from the olmo spec and the big five ones. Run them separately and together", with no post-trained text.
- **Point recorded for the write-up:** with free weights, redundancy in a basis does not limit the fit, only its span does. A less redundant basis spanning the same directions gives the same optimum, and dropping redundant contexts can only lower it. New families can help only by adding directions.
- **My choices:**
  - Full families, not the screened "_top" subsets, to avoid selection.
  - "Big Five" = `bigfive_poles`, `bigfive_profiles`, `persona_big5` and `ipip_facets` (IPIP facets are Big Five sub-traits). "Assistant axis" = `axis_roles` (275) + `axis_traits` (240). OLMo spec = `olmo_spec` (7). 634 new contexts in all, no name overlaps.
  - Bases (each plus the generic header): olmospec (8), bigfive (113), axis (516), three = the three together (635), all = three + the original 86 (721), and legacy87 = the Experiment 1 basis as the reference.
  - The 721-context basis does not fit in 4 × 80 GB with the whole fit half. So every basis, including the legacy87 reference, uses the same fixed 80% of the fit-half questions (seed 100) and the same answers (valid under every family).
  - λ grid extended to {0, 1e-4, 1e-3, 1e-2, 0.1} because of the larger number of weights; still chosen on validation.
- **Code:**
  - `Shards.from_disk` and `disk_nll` in `src/persona_selection/geometric.py` read hidden states per context straight to the GPUs, with no host copy.
  - `scripts/phase4_geom_fit.py` gained `--contexts`, `--row-contexts`, `--tag` and `--fit-frac`.
  - Checked on GPU (`/pscratch/sd/o/ozamram/persona_phase4/tmp/disk_check.py`): reproduces the Experiment 1 fit A hold-out exactly (−1.60454, per-position max difference 7e-6) and the objective and gradient on 300 answers exactly.
- **Runs:**
  - Hidden states: `scripts/phase4_bases_hidden.sh` on interactive allocation 60372590 (4 GPUs), logs `logs/p4b_hidden_*.log`, about 600 GB on scratch.
  - Fits: `scripts/phase4_bases_fit.sh` -> `results/phase4/exp1/instruct_unknown_casual_v1/bases/<tag>/`.
  - Experiment 2 jobs 60356084 and 60356085 are on hold meanwhile, to stay at 4 GPUs.
- 2026-10-27 09:10, **arithmetic mixtures and greedy selection per basis** (author's request): `python scripts/phase4_bases_greedy.py` -> `results/phase4/bases/greedy_arithmetic.{json,png}`.
  - Setup: full families, scores from the hidden-state pass, which match the Phase 1 scores to bf16 rounding (98% identical, mean signed difference 1e-6 per token).
  - Hold-out, Instruct answers: legacy86 −1.7225; axis −1.7284; Big Five + IPIP −1.7260; OLMo spec −1.7241; values −1.7535; base-from-data −1.7251; all 763 −1.7126 (+0.010 ± 0.001); Instruct-derived reference −1.5954.
  - Every basis levels off after 2–3 personas, with the same character picked first (helpful, honest, casual assistant) plus a register persona. Added to the write-up, section 2.
- 2026-10-27 09:40, **geometric fits on other bases, first three** (`scripts/phase4_bases_fit.sh`; same answers, 80% of fit-half questions, 960 fit / 1,200 hold-out answers). Hold-out, Instruct:

  | basis | geometric fit A | arithmetic mixture |
  |---|---|---|
  | olmospec (8) | −1.6802 ± 0.0118 | −1.7239 |
  | bigfive (113) | −1.6401 ± 0.0120 | −1.7261 |
  | legacy87 (87) | −1.6055 ± 0.0121 | −1.7226 |

  - legacy87 reproduces the full-fit-half Experiment 1 value (−1.6045).
  - OLMo spec weights pair near-duplicates with opposite signs: sysmsg −2.26 vs identity +1.35; all −1.93 vs all+HHH +1.69.
  - **Problem:** allocation 60372590 had 40 GB GPUs, so axis, three and all ran out of GPU memory. Their partial outputs were moved to `results/phase4/exp1/instruct_unknown_casual_v1/bases/failed_oom_40gb/` and their logs renamed `*_oom40gb.log`.
  - Rerunning on allocation 60384834 with `-C "gpu&hbm80g"` (80 GB GPUs), log `logs/p4b_fit_interactive2.out`.
- 2026-10-27 10:30, **Assistant Axis geometric fit:** −1.5399 ± 0.0124 hold-out (λ = 1e-3 chosen; −1.5398), against legacy87 −1.6055 on the same answers and 80% fit set. The arithmetic mixture is slightly worse than legacy87 (−1.7284), but the geometric fit is clearly better. Three-together and all fits still running.

## 2026-10-27: Instruct-derived bases (author's design)

- **Goal:** the author proposed two families.
  1. Instruct describes itself in full paragraphs: overall, plus how each of a set of traits shows up in it.
  2. Soft contexts on the base model, each started from one description and trained on Instruct's answers, "to correct between differences of the Instruct model's self-description and what the base model reads".

  Fit each family, arithmetic and geometric.
- **Author's choices:**
  - Assistant Axis traits for the per-trait part.
  - Regenerate the overall descriptions.
  - Make the descriptions full paragraphs; many earlier ones were one sentence.
  - No anchoring of the soft contexts. I had predicted they would converge to one context; the author disagreed, noting that runs from different starting points reached similar likelihood but stayed close to their start. That is a fair point (similar score ≠ same context), so I will measure how different they end up, both in vectors and in per-answer scores.
- **My choices:**
  - 60 of the 240 Axis traits, random with seed 0, one paragraph each.
  - 5 overall prompt phrasings × 4.
  - 150–250 words requested, 120–320 accepted, otherwise resampled.
  - Each soft context starts as EXACTLY its description's token embeddings, with length equal to its token count, so at step 0 it equals the written header and training measures only the correction.
  - Training: fit half only, 20% validation, 3 epochs (the best epoch was 2–3 in every earlier run), lr 5e-3, seed 0.
- **Code:** `scripts/phase4_selfdesc_elicit.py`; `scripts/phase4_selfdesc_stage.sh`; `prompt_tune_base.py --init-from family:name --L 0`; `phase4_hidden_states.py` now handles families whose entries carry a soft-prefix path; fit tags sdo, sdt, sd, sdo_soft, sdt_soft, sd_soft, sd_all in `scripts/phase4_bases_fit.sh`.
- **Check:** before the stage, `/pscratch/sd/o/ozamram/persona_phase4/tmp/prefix_equiv_check.py` confirms that a prefix equal to a header's embeddings reproduces that header's scores through the hidden-state path.
- **Runs:** the stage runs on allocation 60384834 after the large-basis fits (log `logs/p4s_stage_interactive1.out`).
- **Cost estimate:** about 80 soft contexts × about 8 min, about 2.5 hours on 4 GPUs. It will probably need a second allocation; every step is idempotent.
- 2026-10-27 10:50, **author's change:**
  - Elicit ALL 240 Assistant Axis traits (plus 20 overall descriptions), and fit with all of them.
  - Train soft contexts only for the top components. These are chosen on the fit half by `scripts/phase4_selfdesc_select.py --k 20`: the union of the first 20 arithmetic-greedy picks and the 20 largest |c| of the geometric fit on the sd basis.
  - The stage now stops after the description hidden states until `results/phase4/selfdesc/selected.txt` exists.
  - Elicitation is split over 4 GPUs (`--part k/4`, then `--merge`).
- 2026-10-27 13:30, **large-basis geometric fits done** (allocation 60384834, log `logs/p4b_fit_interactive2.out`). Hold-out log P/token on Instruct, 1,200 answers:

  | basis | contexts | geometric fit A (λ = 0) | arithmetic mixture |
  |---|---|---|---|
  | axis | 516 | −1.5399 ± 0.0124 | −1.7284 |
  | three (axis + Big Five + OLMo spec) | 635 | −1.5268 ± 0.0123 | −1.7145 |
  | all (adds legacy86) | 721 | −1.5203 ± 0.0124 | −1.7138 |

  - λ = 1e-3 differs from λ = 0 by under 0.001. Adding families helps only slightly beyond axis alone (+0.020).
  - **Problem:** the shell script ended with a syntax error after the last fit, because I edited `phase4_bases_fit.sh` (new FAM entries) while bash was still reading it. All three result files are complete. But the waiter that was supposed to start the self-description stage was looking for the "done" line, so it never fired, and about an hour of allocation sat idle. Lesson: don't edit a shell script while it is running.
  - The self-description stage (prefix-equivalence check, elicitation, hidden states) is now running on a new 80 GB allocation, 60400384. Log: `logs/p4s_stage_interactive1.out`.
- 2026-10-27 15:20, **self-descriptions elicited and scored** (allocation 60400384, log `logs/p4s_stage_interactive1.out`).
  - Instruct wrote 20 overall descriptions (5 prompts × 4 samples) and one for each of the 240 Assistant Axis traits.
    - Every prompt produced an accepted paragraph within 8 rounds.
    - Length in words: overall 148–216 (median 168); traits 147–261 (median 202).
    - Families: `data/prompts/basis/selfdesc_{overall,traits}.jsonl`; raw samples in `data/prompts/basis/elicited_raw/`.
  - **Known imperfection:** 3 of the 20 overall descriptions and 86 of the 240 trait descriptions slip into the first person ("I possess…") partway through, despite the instruction to write in the third person. I kept them, since the content is still a self-description.
  - **Trait descriptions are mostly denials.** For negative traits Instruct writes what it is *not* ("does not demonstrate the trait of being vindictive…"). So the trait family is many rephrasings of one helpful, balanced assistant, not 240 distinct characters.
  - Base-model hidden states: all 260 contexts, no NaN scores. Shared answer rows are unchanged, because the fit uses the union with the existing row-context families.
  - **Prefix-equivalence check** (a soft prefix equal to a header's token embeddings), 60 answers:
    - hhh: identical, max difference 0.0000.
    - generic: mean difference +0.0035 nats/answer, max 0.31 on one answer.
    - The generic outlier matches the bf16 batch noise seen before (up to 0.38 nats/answer between batch layouts). The prefix path is correct.
  - Next: fits sdo, sdt and sd, then `phase4_selfdesc_select.py --k 20` (log `logs/p4s_fit_interactive1.out`).
- 2026-10-27 16:45, **fits on Instruct's self-descriptions** (`bash scripts/phase4_bases_fit.sh sdo sdt sd` on allocation 60400384; log `logs/p4s_fit_interactive1.out`; outputs `results/phase4/exp1/instruct_unknown_casual_v1/bases/{sdo,sdt,sd}/`).
  - Same 960 fit / 1,200 hold-out answers as the other bases fits. Hold-out log P per token on Instruct's answers, about ±0.012:

  | basis | contexts | best single context (chosen on the fit half) | arithmetic mixture | geometric fit A |
  |---|---|---|---|---|
  | 20 overall descriptions (sdo) | 21 | | −1.666 | −1.631 |
  | 240 trait descriptions (sdt) | 241 | improvisational −1.657 | −1.647 | −1.514 (λ 1e-3: −1.513) |
  | both (sd) | 261 | improvisational −1.657 | −1.647 | −1.512 (λ 1e-3: −1.511) |
  | for comparison: all 721 base-persona contexts | 721 | | −1.714 | −1.520 |
  | 86 base personas (Phase 1) | 86 | college student −1.731 | −1.723 | −1.606 |

  - **The self-descriptions are much better single contexts than any base persona.**
    - The best single trait description, chosen on the fit half, scores −1.657. The median single description scores −1.70. Both beat every mixture of base personas (at best −1.714).
    - So the arithmetic mixture gains 0.075 over the 721 base contexts. Most of that gain is the better single context; mixing adds 0.010.
  - **Geometric fit:** −1.512, barely ahead of the 721 base contexts (−1.520). The free logit fit reaches about the same place from either basis. The self-description basis is not more expressive for the geometric fit; its individual contexts are simply better.
  - The overall descriptions are weaker (−1.666 arithmetic) than the trait descriptions.
  - **Selection** (`python scripts/phase4_selfdesc_select.py --k 20`, `results/phase4/selfdesc/selected.{txt,json}`): 35 descriptions, 5 picked by both the geometric weights and the arithmetic greedy order. Geometric variant chosen on validation: λ 1e-3, generic target. Selection used fit-half answers only.
  - **Soft-context training** for the 35 is submitted as batch job 60412060 (regular, 4 GPUs, 3.5 h; log `logs/p4s_stage_soft_<job>.out`).
    - Estimate: about 10 minutes per context, so about 90 minutes on 4 GPUs, plus about 25 minutes of hidden states.
    - Interactive allocation 60400384 was cancelled (51 minutes left, not enough).
- 2026-10-28 07:10, **soft contexts from the 35 selected self-descriptions** (batch job 60412060, 2 h 09 min; log `logs/p4s_stage_soft_60412060.out`; prefixes in `results/phase4/selfdesc/prefix/<name>/`; families `selfdesc_{overall,traits}_soft`, 1 + 34 entries).
  - Training setup: each starts from exactly its description's token embeddings (217–267 vectors), no anchor, lr 5e-3, 3 epochs, epoch chosen on validation (epoch 2 for every context).
  - Hold-out answers: 1,200, the same split as the fits.
  - **Measured: they converge in function, not in parameters.**
    - Hold-out log P/token after training: −1.390 to −1.400 (median −1.396), against −1.684 median for the starting descriptions.
    - So all 35 land within 0.01 of each other, whichever description they started from. This matches the earlier L64 generic-start run (−1.396 at its best epoch).
    - Per-answer scores across contexts correlate 0.990 (the descriptions: 0.983). The spread across contexts per answer shrinks from 0.047 to 0.027 nats/token.
    - In parameters they stay near their starts: median per-token cosine to the initial embedding 0.82, while each vector moves about 70% of its own norm.
    - The updates point the same way. The mean update direction has cosine 0.80 between contexts.
  - **Reading:** the author was right that they do not collapse to one vector. But each one ends up as "its description plus a largely shared correction", and they reach the same likelihood. Expect little gain from mixing them; the fits test that.
  - Fits `sd_soft` and `sd_all` are running on interactive allocation 60448170 (log `logs/p4s_fit_soft_interactive1.out`). sdo_soft and sdt_soft were not run separately: sdo_soft would have one context, and sdt_soft is sd_soft minus one context.
- 2026-10-28 08:00, **reading the soft contexts back out as text** (author's request). New `scripts/phase4_selfdesc_readout.py`, run by `scripts/phase4_selfdesc_readout.sh`.
  - **Methods**, for each of the 35 soft contexts:
    1. Nearest token per prefix vector (by cosine); this projected sequence is scored as a hard prefix.
    2. SALVE-style verbalisation (arXiv:2609.16927, one round):
       - Base model + soft context continues "Character description: The assistant", 32 samples.
       - Each sample is turned into a written header and scored on 96 fit-half answers; the best is scored on the 1,200 hold-out answers.
    3. **Control:** the same procedure starting from the original written description. It tells whether a verbalisation captures the soft correction or is just resampling.
  - **Smoke test** on `sp_sdo_style_2`, 4 candidates, test file deleted afterwards:
    - Nearest tokens: 0 of 217 positions change. The median cosine to the initial embedding is 0.84, so every vector still lies closest to its original token. The projected text *is* the original description, scoring −1.672 against −1.396 for the soft context.
    - Verbalisations from the soft context and from the description gave the same top candidate (hold-out −1.680).
    - Fixed: the join of "The assistant" with a continuation that starts with "’s".
  - **Full run:** batch job 60449517 (4 GPUs, at most 4 h, held until allocation 60448170 ends, to stay at 4 GPUs at a time).
- 2026-10-28 08:10, **fits with the soft contexts** (`bash scripts/phase4_bases_fit.sh sd_soft sd_all`, allocation 60448170, log `logs/p4s_fit_soft_interactive1.out`). Hold-out log P/token, 1,200 answers, about ±0.012:

  | basis | arithmetic mixture | geometric fit A (λ = 0 / best on validation) |
  |---|---|---|
  | 35 soft contexts (sd_soft) | **−1.380** | −1.392 / −1.390 |
  | 35 soft + 260 descriptions (sd_all) | −1.380 | −1.395 / −1.387 (λ 1e-2) |
  | single soft contexts | −1.390 to −1.400 | |
  | 260 descriptions (sd) | −1.647 | −1.512 |

  - Mixing the 35 soft contexts adds only about 0.01–0.02 over a single one. The arithmetic mixture slightly beats the geometric fit.
  - **Caveat:** the soft contexts were trained on the fit-half answers, so the fit-half scores the weights are fitted on are inflated (geometric: fit-half −1.273 against hold-out −1.392).
    - The free geometric weights exploit this overfitting; the arithmetic weights are more robust to it.
    - A cleaner test would train the soft contexts on one part of the fit half and fit the mixture weights on another. Not done.
  - Adding the 260 descriptions to the soft contexts adds nothing.
  - **Bottom line:** the soft contexts all sit at about −1.39. Mixing them reaches −1.38, about the level of the Experiment 3 prefix (−1.37), and the gap to Instruct (−0.90) stays about 0.48.
- 2026-10-28 11:00, **better verbalisation of the soft contexts** (author's request: prefill with the seed description; follow SALVE, arXiv:2609.16927, PDF in the repo root).
  - Cancelled the simpler readout job 60449517 before it started.
  - **New script** `scripts/phase4_selfdesc_salve.py`, run by `scripts/phase4_selfdesc_salve.sh`. It implements SALVE's sentence-level beam search (paper section 3.3 and appendix C), adapted to a base model:
    - The soft context sits where the header was during training.
    - The 4 SALVE queries become transcript turns ("Reply with the description of the AI assistant given at the top of this page in double quotes…"), with the assistant turn prefilled with the opening delimiter plus the text so far.
    - Sampling temperature 0.7, sentences cut at the closing delimiter.
    - Each candidate is scored as a written header on a fixed batch of 64 fit-half answers. Best-ever candidate kept; the final text is scored on the 1,200 hold-out answers.
  - **Arms:**
    - `extend_soft`: prefill the whole seed description; append sentences sampled after the soft context.
    - `extend_text`: control for `extend_soft`, sampled after the written seed header.
    - `rewrite_soft` / `rewrite_text`: prefill the seed's first sentence and rebuild the rest, after the soft context / after the written header.
    - `salve_soft`: the paper's method, starting from "The assistant".
  - **Raw-sample check** (sdt_casual; `/pscratch/sd/o/ozamram/persona_phase4/tmp/salve_debug.py`):
    - With the full seed prefilled, the model mostly closes the quote, appends the casual clause, or drifts into unrelated documents. So few extension sentences survive the filters.
    - With a partial prefill, or from "The assistant", it copies or paraphrases the seed closely, partly slipping into the first person ("I tend to use a relaxed, informal tone…").
  - **Smoke test** (sdt_casual, B 2, M 4, R 2), hold-out log P/token:

    | text | score |
    |---|---|
    | seed | −1.671 |
    | soft context | −1.392 |
    | extend_soft | −1.671 (no appended sentence beat the seed) |
    | extend_text | −1.671 (no valid candidates) |
    | rewrite_soft | −1.693 |
    | rewrite_text | −1.697 |
    | salve_soft | −1.697 |

    The verbalisations are paraphrases of the seed and recover none of the soft context's gain.
  - **Full run:** 8 contexts (the 5 picked by both selection rules plus the next 3 by geometric weight), B 4, M 6, R 5. Interactive allocation 60460749; log `logs/p4s_salve_interactive1.out`; outputs in `results/phase4/selfdesc/salve/`.
  - Fixed during the smoke test: the scratchpad `/tmp` is local to the compute node, so test outputs now go to `$SCRATCH`.
- 2026-10-28 11:15, **verbalisation results, 8 soft contexts** (B 4, M 6, R 5; `results/phase4/selfdesc/salve/`, summary `results/phase4/selfdesc/salve.json`).
  - Mean hold-out log P/token over the 8 contexts:

    | text | mean score | share of the soft context's gain over its seed |
    |---|---|---|
    | seed description | −1.699 | |
    | soft context | −1.396 | |
    | extend_soft | −1.677 | 7.6% |
    | extend_text | −1.699 | 0% (no candidates survive) |
    | rewrite_soft | −1.672 | 6.7% |
    | rewrite_text | −1.721 | −7% |
    | salve_soft | −1.675 | 3.4% |

    Shares are medians over the 8 contexts.
  - **The soft context does carry something verbalisable.** Rewrites sampled after the soft context beat rewrites sampled after the text by 0.049 on average, and extensions beat zero. But the amount is small: about 7% of the gain.
  - **What gets added**, read by hand:
    - For extreme seeds (ironic, serious, theatrical): sentences pulling toward a safe, honest, helpful, casual assistant ("I won't make things up… I'll steer clear of anything harmful", "In more casual settings, the assistant may adopt a relaxed or humorous tone").
    - Also "speaks in a casual, conversational tone" and first-person chat closers ("hope you get what you need!").
    - This is consistent with the shared update direction (cosine 0.80 across contexts).
  - **Token-level location of the gain** (`scripts/phase4_selfdesc_tokengain.py`; `results/phase4/selfdesc/token_gain_{fit,test}.json`; sdt_casual, sdt_ironic, sdo_style_2):
    - The mean gain is 0.28–0.39 nats/token on the hold-out, with about 20% of it on the first 5 answer tokens.
    - The largest single item is the em dash: 571 occurrences on the fit half, about +3 nats each, roughly 9% of all gain.
    - Then openers: "Honestly," (225 answers), "Probably", "No", "Yeah,", "Hey,", "Hmm,".
    - The rest is spread over common words (" it", " and", " or", " just", " probably").
  - **Naming these habits in words** (one sentence written from the fit-half analysis only, `scripts/phase4_selfdesc_tics.py`, `results/phase4/selfdesc/tics.json`):
    - Helps the generic header: −1.758 → −1.728.
    - Hurts the 8 descriptions on average: −1.699 → −1.710; it helps only "serious".
    - Not iterated further, to avoid tuning on the hold-out.
  - **Running:** the paper's search budget (B 4, M 16, R 12) on 4 contexts, soft arms only (`scripts/phase4_selfdesc_salve_big.sh`, `results/phase4/selfdesc/salve_big/`).
- 2026-10-28 13:00, **verbalisation at the paper's search budget** (B 4, M 16, R 12, scoring batch 64 fit-half answers; 4 contexts; `scripts/phase4_selfdesc_salve_big.sh`; `results/phase4/selfdesc/salve_big/`, summary `results/phase4/selfdesc/salve_big.json`; allocation 60460749).
  - Hold-out mean over sdo_style_2, sdt_casual, sdt_serious, sdt_skeptical:

    | text | mean score | share of the soft context's gain (median) |
    |---|---|---|
    | seed | −1.687 | |
    | soft context | −1.394 | |
    | extend_soft | −1.633 | 17.5% |
    | rewrite_soft | −1.638 | 14.6% |
    | salve_soft | −1.632 | 17.9% |

    The smaller search recovered about 7% on the same kind of contexts, so more search roughly doubles the recovered share. It still loses more than 80% of the gain.
  - **What the winning texts look like:**
    - Long (300–450 words).
    - The seed, paraphrased, followed by a run of chatty first-person assistant talk: "I'm here to help… just let me know!", "If I'm unsure, I'll let you know", "no question is too basic", "peace out!".
    - Repeated sentences, and in places the transcript leaking in ("Please provide the text you'd like me to output verbatim").
    - Extreme seeds are softened toward "professional but approachable".
  - **Reading:** the search finds that more Instruct-like register text in the header helps, by writing that register out as filler, not by describing a new character.
  - **Caveat:** each text was chosen on a 64-answer fit-half batch; hold-out scores are untouched by the choice.
- 2026-10-28 13:40, **verbalised texts swapped into the self-description basis** (author's request).
  - `scripts/phase4_selfdesc_verbbasis.py` builds families `selfdesc_{overall,traits}_v`: each verbalised description is replaced by its best soft-conditioned verbalisation, chosen by the fit-half batch score. Replaced entries are renamed `v_<name>`; the other descriptions are unchanged.
  - Record in `results/phase4/selfdesc/verb_basis.json`. Only 8 of 260 are replaced so far.
  - New hidden states, then fit tag `sd_verb` (added to `phase4_bases_fit.sh`), on allocation 60468148; log `logs/p4s_fit_verb_interactive1.out`.
  - **Verbalising all 35** (author approved): batch job 60468203, paper budget, salve_soft arm only. That arm was best by fit-half batch score in 3 of 4 paper-budget runs.
    - Held until the interactive allocation ends, to stay at 4 GPUs at a time. Estimate about 5 h; limit 8 h. Log `logs/p4s_salve_all_<job>.out`.
    - The basis will then be rebuilt with all 35 replaced and refitted.
  - The merge in `phase4_selfdesc_salve.py` now uses only the hold-out keys common to all records, since the runs have different arms.
- 2026-10-28 14:05, **fit with 8 verbalised texts swapped in** (`bash scripts/phase4_bases_fit.sh sd_verb`, allocation 60468148, log `logs/p4b_fit_sd_verb.log`).
  - The 8 new contexts have no NaN scores. Same 960 fit / 1,200 hold-out answers as before. Hold-out log P/token:

    | basis | arithmetic mixture | geometric fit A (best on validation) |
    |---|---|---|
    | 260 original descriptions (sd) | −1.647 | −1.511 |
    | 252 original + 8 verbalised (sd_verb) | −1.610 | −1.504 (λ 1e-3) |
    | best single verbalised text (casual) | −1.615 | |
    | 35 soft contexts | −1.380 | −1.390 |

  - **Arithmetic mixture:** gains 0.037, almost all from the best single swapped-in text (mixing adds 0.005).
  - **Geometric fit:** gains 0.008; the largest weights move onto the swapped-in texts (v_sdt_casual +0.58, v_sdt_factual +0.50, v_sdt_improvisational +0.45).
  - Both fits remain far below the soft contexts.
  - Allocation released. Batch job 60468203 (all 35) is now waiting only for the allocation to end.

## 2026-10-29 — Fiction framings for the persona header (author's question)
**Goal:** test whether framing the description as fiction (play, story, role-play character card) makes the base model assign higher likelihood to Instruct's answers than the "Below are a series of dialogues..." transcript header.
**What was done:** `scripts/phase4_framing.py` (5 framings x 17 descriptions: generic + the 8 verbalisation targets + 8 random self-descriptions, seed 0; 300 fit-half + 300 test-half answers of instruct_unknown_casual_v1, seed 0; answer tokens asserted identical across framings). One wording per framing, written before scoring, none tuned. Ran on GPU 1 of batch job 60468203 (overlap), ~1 h. Summary: `scripts/phase4_framing_summary.py` -> `results/phase4/framing/{scores,summary}.json`. Commit at run time 4e805f8 (+ uncommitted scripts).
**Results (test half; log P/token; paired standard error vs transcript ~0.003):**
| framing | generic | mean of 16 descriptions | description gain over own generic | best description |
|---|---|---|---|---|
| transcript | -1.759 | -1.700 | +0.059 | -1.651 |
| play | -1.832 | -1.768 | +0.065 | -1.722 |
| story (prose, quoted question) | -1.942 | -1.831 | +0.111 | -1.786 |
| card | -1.807 | -1.720 | +0.087 | -1.686 |
| story_chat | -1.790 | -1.732 | +0.058 | -1.687 |
Fit half gives the same ordering. Transcript is best in absolute terms for every framing. Fiction framings make descriptions matter more (story roughly doubles the gain) but start from a much worse baseline. Description rankings largely carry over (Spearman 0.85-0.98).
**Caveats:** one wording per framing; the question carries the suffix "Answer in two or three sentences of plain text, in a casual, conversational tone.", which is odd inside a story's quoted question; answers are short chat replies, so prose framings may penalise their style.
**Next:** not adopting a new framing. Possible follow-ups: a few wordings per framing chosen on the fit half; drop the suffix inside fiction framings.

## 2026-10-30 — Framing round 2: card/transcript hybrids, self-description first turn, SFT model as scorer (autonomous, author's request)
**Goal:** (1) card + transcript hybrids, a few wordings; (2) a first turn where the user asks the assistant to describe itself and the description is its reply; (3) use the SFT checkpoint (allenai/Olmo-3-7B-Instruct-SFT) instead of the base model to fit Instruct's answers, incl. with the self-description turn.
**What was done:** `scripts/phase4_framing.py` extended (framings hyb_char/hyb_sheet/hyb_brackets/hyb_about, turn_describe/turn_yourself/turn_bare, chat_plain/chat_sys/chat_turn; `--model`; per-answer totals + base-transcript token counts so chat framings, whose answers have no leading space, are on one scale). Driver `scripts/phase4_framing2.sh` on interactive allocation 60547947 (4 GPUs, ~25 min). Same 17 descriptions and 600 answers (seed 0) as round 1. Summary `scripts/phase4_framing_summary2.py` -> `results/phase4/framing/summary2.json`. Wordings written once in advance, not tuned. Check: Instruct under its own chat template -0.899 test (reference -0.898); round-1 and round-2 transcript scores identical per answer.
**Results (test half, nats per reference token; paired SE vs base transcript 0.001-0.003 for base rows, ~0.01 for SFT rows):**
| model | framing | generic | mean of 16 desc | desc gain | best desc |
|---|---|---|---|---|---|
| base | transcript | -1.753 | -1.697 | +0.056 | -1.651 |
| base | hyb_char / hyb_sheet / hyb_brackets / hyb_about | -1.779 / -1.782 / -1.755 / -1.771 | -1.697 / -1.706 / -1.715 / -1.696 | +0.08 / +0.08 / +0.04 / +0.08 | -1.661 / -1.672 / -1.679 / -1.656 |
| base | turn_describe / turn_yourself / turn_bare | -1.789 / -1.800 / -1.764 | -1.672 / -1.681 / **-1.651** | +0.12 / +0.12 / +0.11 | -1.630 / -1.643 / **-1.602** |
| SFT | chat_plain (no description) | **-1.050** | | | |
| SFT | chat_sys / chat_turn | -1.117 / -1.203 | -1.183 / -1.200 | -0.066 / +0.003 | -1.145 / -1.176 |
| SFT | transcript / hyb_char / turn_describe / turn_bare | -1.349 / -1.359 / -1.505 / -1.491 | -1.388 / -1.374 / -1.359 / -1.339 | -0.04 / -0.02 / +0.15 / +0.15 | -1.310 / -1.312 / -1.308 / -1.284 |
| Instruct | chat_plain | -0.899 | | | |
Fit half gives the same ordering. Bottom line: hybrids ~ no better than transcript; the self-description first turn is the best base framing (mean description +0.046, best description +0.049 over transcript) and doubles the description gain. The SFT model under its plain chat template already gets -1.05 (base best single description -1.60, Instruct -0.90): about 80% of the base->Instruct gap is closed by SFT with NO description. Under SFT, descriptions hurt as a system prompt and do nothing as a first turn; in raw-text framings they help but stay far below plain chat.
**Next:** fits with these framings (below).

## 2026-10-30 — All 35 verbalisations in the basis; SFT model as the scorer (fits)
**What was done:** `scripts/phase4_selfdesc_verbbasis.py --prefix v35_ --suffix _v35` (now handles result files with only salve_soft; new names so no stale hidden states; old `_v` families and verb_basis.json untouched) -> families selfdesc_{overall,traits}_v35, `results/phase4/selfdesc/verb_basis_v35.json`. `scripts/phase4_hidden_states.py` gained `--model/--framing/--tag` (output `<run>__<tag>`; in chat framings the no-description context "generic" is the plain chat prompt); `scripts/phase4_geom_fit.py` gained `--model` (output layer). Driver `scripts/phase4_sft_fit_stage.sh` on allocation 60547947: base hidden states for v35 + fit tag bases/sd_verb35; SFT hidden states for the 260 self-descriptions + plain chat in framing chat_turn (`$SCRATCH/persona_phase4/hidden/instruct_unknown_casual_v1__sft_chat_turn`, ~2 h on 4 GPUs) + fit A (`results/phase4/exp1/instruct_unknown_casual_v1__sft_chat_turn/sd/`). Same 960 fit / 1200 hold-out answers and lambda grid as the base sd fits; SFT tokenisation has 0.08% more answer tokens (comparable). Generic one-hot check: -1.0493 from hidden states vs -1.0496 stored.
**Results (hold-out log P/token, +-0.01):**
| scorer, basis | single best / no description | arithmetic mixture | geometric fit |
|---|---|---|---|
| base, sd (260 descriptions) | -1.657 | -1.647 | -1.511 |
| base, sd with 8 verbalisations | -1.615 | -1.610 | -1.504 |
| base, sd with all 35 verbalisations | | -1.603 | -1.503 |
| SFT, chat_turn, sd + plain chat | plain chat -1.049 | -1.050 (all weight on plain chat) | -1.003 (lambda 1e-3) |
| Instruct itself | -0.898 | | |
Verbalising 27 more contexts adds nothing over the 8. Under SFT, descriptions add nothing to a mixture; the geometric fit gains 0.046 (about 30% of the remaining SFT->Instruct gap of 0.15), with weight ~1.03 on plain chat and many +-0.2-0.37 trait components (traditional -, earnest -, charismatic +, utilitarian -, innovative +...), i.e. spread small corrections rather than one persona.
**Pending:** done, see the turn_bare entry below (batch job 60549575 was cancelled after ~2.5 h pending and rerun on interactive allocation 60564081).

## 2026-10-30 — Soft context on the SFT checkpoint (author's question)
**What was done:** `scripts/prompt_tune_sft.py` (new): soft vectors in place of the system-message text inside the SFT chat template; same rows, hold-out questions (basis_test_qids), validation split, lr 5e-3, 3 epochs, batch 8, seed 0 as the base self-description soft contexts. Two inits: the default system message (23 vectors; at init equals plain chat, checked: -1.0495) and the sdt_casual self-description (244 vectors). Ran on GPUs 0-1 of allocation 60564081 (overlapping the base turn_bare job), ~6 and ~13 min. Output `results/phase4/sft_soft/{default_sys,sdt_casual}/`.
**Results (hold-out log P/token, 1200 answers; best epoch by validation = 2 in both):**
| setup | hold-out |
|---|---|
| base, generic header | -1.758 |
| base, best soft context (35 self-description inits) | ~-1.39 |
| SFT, plain chat | -1.049 |
| SFT, geometric fit over 260 descriptions | -1.003 |
| SFT + soft context, 23 vectors from default system message | **-0.945** |
| SFT + soft context, 244 vectors from sdt_casual | -0.948 (init -1.233) |
| Instruct itself | -0.898 |
SFT + soft context closes 69% of the SFT->Instruct gap (0.151 -> 0.047) and 95% of the base-generic->Instruct gap. Length/init barely matter (23 vs 244 vectors: -0.945 vs -0.948). Validation dropped at epoch 3 in both (mild overfitting on 960 answers). One seed each; no seed variance estimate.
**Open:** what the remaining 0.047 is (per-token analysis as for the base soft contexts).

## 2026-10-30 — Base model, self-description first turn (turn_bare), geometric fit over the sd basis
**Goal:** does the best base framing from the screen (turn_bare: "User: Describe yourself in the third person: ...\nAssistant: <description>\n\nUser: <question>\nAssistant:") also improve the mixture fits over the 260 self-descriptions?
**What was done:** `srun --jobid=60564081 --overlap -n1 bash scripts/phase4_base_turn_fit.sh` (hidden states with `--framing turn_bare --tag base_turn_bare`, then `phase4_geom_fit.py --fits A --contexts selfdesc_overall,selfdesc_traits --fit-frac 0.8 --tag sd`), commit 8dda23e + uncommitted script. Output `results/phase4/exp1/instruct_unknown_casual_v1__base_turn_bare/sd/`; log `logs/p4_base_turn_bare_fit.log`. Same 960 fit / 1200 hold-out answers and lambda grid; lambda chosen on validation (1e-3 with generic as the L2 centre, 1e-4 with zero). "generic" here = the unknown-character body as the first-turn reply. Generic one-hot check -1.7707 from hidden states vs -1.7708 stored.
**Results (hold-out log P/token; paired difference vs the transcript sd fit on the same answers, +- paired standard error):**
| | transcript | turn_bare | difference |
|---|---|---|---|
| generic | -1.758 | -1.771 | -0.012 +- 0.002 |
| arithmetic mixture | -1.646 | -1.605 | +0.041 +- 0.001 |
| geometric fit (lambda 1e-3, generic centre) | -1.511 | -1.496 | +0.015 +- 0.002 |
The first-turn framing helps the arithmetic mixture (as in the screen: descriptions are read better), but most of that advantage disappears in the geometric fit (+0.015 per token, small but well outside the noise). The largest weights change (bitter -, chaotic -, nurturing +, hostile +, ...), i.e. weights are spread and not interpretable one by one, as in the transcript fit. Base ceiling stays far from SFT plain chat (-1.049).

## 2026-10-30 — Qwen replication of the mixture fits (author's request; in progress)
**Goal:** check whether the Olmo phase-4 findings (geometric fit far above the arithmetic mixture, weights spread over many traits, the soft-context gap, the framing result) hold for Qwen2.5-7B base + Instruct.
**Scope agreed with the author:** Qwen Instruct answers on the same 300 questions; a Qwen self-description basis re-elicited from Qwen Instruct; the generic bases (legacy87, Assistant Axis, OLMo spec, Big Five) reused as they are; core fits on all bases, soft contexts (generic + top self-description starts) and the framing screen on Qwen; no SALVE verbalisation. Cross-family, lighter: Qwen base on Olmo answers and Olmo base on Qwen answers, only legacy87 + the answering model's self-description basis. Contamination check: each base model's likelihood of its own family's Instruct answers under the Instruct chat template vs the generic transcript header. Budget about 60 GPU-hours; jobs at most 6 h (author: longer jobs never run), at most 4 GPUs at a time (batch jobs chained with dependencies). My first cost estimate (6-10 GPU-h for the core fits) was wrong by about 3x; corrected from the Olmo logs before anything ran.
**Code (Olmo defaults unchanged):** `--model/--prefix` in phase4_selfdesc_elicit.py (families qwen_selfdesc_*, names qsdo_*/qsdt_*); phase4_hidden_states.py accepts another model in the transcript framing (headers as they are, output `<run>__<tag>`); `--hid/--fit/--prefix/--out` in phase4_selfdesc_select.py; RUN/MODEL/ROWCTX env vars and qsd* tags in phase4_bases_fit.sh (logs now `p4b_fit_<run>_<tag>.log`); `--run/--families/--ref-model/--template-from` in phase4_framing.py (with other families the 16 descriptions are drawn at random, seed 0); `--generic-hidden` in phase4_exp3_rows.py. New drivers scripts/phase4_qwen_stage1.sh, scripts/phase4_qwen_hidden.sh.
**Stage 1 done (allocation 60575824, 21 min):** `scripts/phase1_instruct_sample.py --instruct-model Qwen/Qwen2.5-7B-Instruct --base-model Qwen/Qwen2.5-7B --n-per-question 8 --framing unknown --register casual --seed 0 --out results/phase4/qwen/runs/qwen_instruct_unknown_casual_v1` -> 2400 answers (casual, short, look normal). Stored rough references (phase-1 scoring convention, leading-space answer): Qwen Instruct self -0.939/token, Qwen base generic header -1.235/token (Olmo: -0.898 and -1.758), i.e. the base->Instruct gap is about 3x smaller on Qwen; consistent with the earlier concern that Qwen base has seen instruction-style data, to be checked. Self-descriptions: 20 overall (124-146 words) and 240 traits (120-207 words), `data/prompts/basis/qwen_selfdesc_{overall,traits}.jsonl`; overall ones often name the model ("The assistant, known as Qwen, ..."). The olmo_spec basis mentions OLMo by name in 3 contexts; reused unchanged.
**Smoke test:** Qwen base hidden states on Olmo answers, generic header: all 2400 valid, -2.023/token (Qwen tokenisation).
**Queued:** Qwen base x Qwen answers, Qwen sd basis (interactive); 60576074 Qwen base x Olmo answers (legacy86 + Olmo sd); 60577366 Qwen base x Qwen answers (generic bases); 60577367 Olmo base x Qwen answers (legacy86 + Qwen sd). Each 5 h limit, chained.
**Update (2026-10-30 evening):**
- Fix: `src/persona_selection/geometric.py` hardcoded OLMo's hidden size 4096; now read from the file size and the run's position count (3584 for Qwen), and W_U.shape[1] on the GPU side. The first Qwen fit attempt crashed on this (no results produced).
- Qwen self scores (phase4_self_tokens.py, stripped answer, as Olmo's -0.898): Qwen Instruct -0.734/token. `results/phase4/self_tokens/qwen_instruct_unknown_casual_v1.npz`; soft-context rows `results/phase4/qwen/rows/qwen_instruct_unknown_casual_v1.jsonl`.
- Qwen base hidden states, Qwen sd basis (261 contexts, 1 h on 4 GPUs, no invalid answers): `$SCRATCH/persona_phase4/hidden/qwen_instruct_unknown_casual_v1__qwenbase`.
- Contamination check (`phase4_framing.py --framings transcript,chat_plain --generic-only transcript,chat_plain --template-from <Instruct>`; test half, 300 answers, own tokenisation): Olmo base on Olmo answers transcript -1.753, Instruct chat template -1.843; Qwen base on Qwen answers transcript -1.229, chat template -1.268; Qwen Instruct -0.728 (Olmo Instruct -0.899). The chat template does NOT help either base model, so no direct sign that Qwen base knows its Instruct format; but its generic score is much closer to its Instruct (gap 0.50 vs 0.86). Not conclusive; the cross-family runs (same text, both base models, per-answer nats) will say more. Files `results/phase4/qwen/framing/contam_*.json`.
- Framing screen on Qwen base (generic + 16 random Qwen self-descriptions, seed 0; all 16 drawn happen to be trait descriptions), test half, per reference token: transcript generic -1.229 / mean description -1.271 (descriptions HURT, -0.042 +- 0.002; on Olmo they helped +0.06); turn_bare mean description -1.206 (+0.065 +- 0.003 over transcript; best framing, as on Olmo), turn_describe -1.227; play -1.454, story -1.478, card -1.329, story_chat -1.314, hyb_char -1.280. `python scripts/phase4_framing_summary2.py --dir results/phase4/qwen/framing --files "base:scores_qwenbase_0.json,...,scores_qwenbase_3.json" --nref scores_qwenbase_0.json` (script parametrised; Olmo summary2.json reproduced byte-identical).
- Qwen sd fits (`RUN=qwen_instruct_unknown_casual_v1__qwenbase MODEL=Qwen/Qwen2.5-7B ROWCTX=qwen_selfdesc_overall,qwen_selfdesc_traits bash scripts/phase4_bases_fit.sh qsd qsdo qsdt`; 960 fit / 1200 hold-out answers, as Olmo), hold-out log P/token: generic -1.237, arithmetic -1.197, geometric -1.026 (qsdo alone -1.119, qsdt alone -1.031). Olmo sd: -1.758 / -1.647 / -1.511. Share of the base-generic -> Instruct gap closed: arithmetic 8% (Olmo 13%), geometric 42% (Olmo 29%). Weights again spread (style_3 +0.87, contrarian +0.62, animated -0.56, generic +0.49, goofy +0.49, empathetic -0.48, ...). Generic one-hot check: mean matches (-1.2371 vs -1.2373) but max per-answer difference 0.60 nats (Olmo 0.18); bf16 batch effects, to keep an eye on.
- Soft-context selection (fit half only): 35 Qwen self-descriptions (`results/phase4/qwen/selfdesc/selected.txt`), training started on the rest of allocation 60575824; batch job 60581568 finishes it.
- Qwen soft contexts (`srun ... bash scripts/phase4_qwen_soft.sh` on allocation 60575824, 36 contexts in 1 h on 4 GPUs: generic start + the 35 selected; same recipe as Olmo, one seed each), hold-out log P/token at the best epoch by validation: best -0.879 (qsdo_story_2), median -0.886, worst -0.900, generic start -0.886; Qwen base generic header -1.237, Qwen Instruct -0.734. So a soft context on Qwen BASE closes 70% of its base->Instruct gap (Olmo base: 43%, best -1.390 of 35; Olmo SFT + soft context: 69% of the SFT->Instruct gap). Qwen base is to its Instruct roughly what Olmo SFT is to Olmo Instruct, consistent with Qwen base having been trained on instruction-style data (not proven). Starting point barely matters (spread 0.02), as on Olmo. Hidden states of the 35 soft contexts written (families qwen_selfdesc_{overall,traits}_soft). Batch job 60581568 cancelled (nothing left); fits qsd_soft/qsd_all = job 60584835, after the cross-family fits.

## 2026-11-02 — Qwen replication: final results
**What was done (after the entries above):** chained batch jobs, each <= 5 h, 4 GPUs, one at a time: hidden states 60576074 (Qwen base x Olmo answers: legacy86 + Olmo sd), 60577366 (Qwen base x Qwen answers: legacy86, Axis, Big Five, OLMo spec), 60577367 (Olmo base x Qwen answers: legacy86 + Qwen sd), 1.4-1.7 h each; fits 60577838 / 60577839 (Qwen, every basis), 60577840 (both cross-family pairs), 60584835 (qsd_soft, qsd_all). Commands: `sbatch scripts/phase4_qwen_hidden.sh <run> <model> <tag> <families>` and `sbatch scripts/phase4_qwen_fits.sh <hidden run> <model> <row families> <tags>`. Summary: `python scripts/phase4_qwen_summary.py` -> `results/phase4/qwen/summary.json`. Row filter: Qwen fits of the generic bases use legacy86+Axis+Big Five+OLMo spec+Qwen sd, the sd/soft fits use the Qwen sd families (all 2400 answers valid either way except 1 in the cross pair below).
**Results — own family** (hold-out log P/token, 1200 answers; generic / arithmetic / geometric; share of the generic->Instruct gap closed by the geometric fit, Olmo in brackets; Instruct self: Olmo -0.898, Qwen -0.733):
| basis | Qwen | geometric share, Qwen (Olmo) |
|---|---|---|
| OLMo spec | -1.237 / -1.234 / -1.202 | 7% (9%) |
| Big Five | -1.237 / -1.216 / -1.108 | 26% (14%) |
| legacy87 | -1.237 / -1.217 / -1.090 | 29% (18%) |
| Axis | -1.237 / -1.204 / -1.034 | 40% (25%) |
| three generic | -1.237 / -1.203 / -1.027 | 42% (27%) |
| all generic | -1.237 / -1.203 / -1.020 | 43% (28%) |
| self-descriptions | -1.237 / -1.197 / -1.026 | 42% (29%) |
| sd soft contexts | -1.237 / -0.863 / -0.866 | 74% (43%) |
| sd + soft | -1.237 / -0.863 / -0.865 | 74% (43%) |
Replicates on Qwen: arithmetic mixtures of written descriptions close <= 8%; geometric fits much more; same ranking of bases; sd basis ~ all generic bases combined; with soft contexts arithmetic = geometric and only slightly above the single best soft context (-0.879), as on Olmo. Absolute geometric gains per token are similar in both models (0.15-0.25 nats on legacy87/sd); the shares are bigger on Qwen because its gap is smaller.
**Results — same text, both base models** (mean hold-out log P per ANSWER in nats, tokeniser-free; generic -> best geometric):
| answers | Instruct self | own-family base | other-family base |
|---|---|---|---|
| Olmo (1200) | -40.9 | Olmo -80.0 -> -68.7 (sd) | Qwen -92.4 -> -81.1 (Olmo sd) |
| Qwen (1199) | -34.5 | Qwen -58.2 -> -48.2 (Qwen sd) | Olmo -73.4 -> -68.0 (Qwen sd) |
Each base model predicts its OWN family's Instruct text better (generic header: +12.4 +- 0.2 nats/answer on Olmo answers, +15.3 +- 0.2 on Qwen answers, paired). So Qwen base being close to Qwen Instruct is family-specific, not a general advantage on assistant text; consistent with Qwen base having seen Qwen-style (instruction) data or a closer pretraining mix; these data cannot tell which. Persona fits help the other-family base less on Qwen answers (~5 vs ~10 nats/answer), about equally on Olmo answers (~11 each).
**Confidence:** single seed for every soft context; fits on one fixed 80% of the fit half (as Olmo); hold-out standard errors ~0.01/token (question bootstrap), paired differences much tighter.
**Open:** whether Qwen base's closeness to its Instruct comes from instruction data in pretraining (would need Qwen's data documentation or an earlier checkpoint, not available); per-token analysis of what the soft contexts add on Qwen.

## 2026-11-02 — Llama 3.1 8B replication: prepared, not run (author is moving the project to another cluster)
**Goal:** third model family (author's choice: meta-llama/Llama-3.1-8B + -Instruct), same scope as the Qwen replication without the cross-family pairs (~40-45 GPU-h).
**What was done:** `hf auth login` (gated repos; token in `$HF_HOME/token`). Code made Llama-safe, verified by inspection and tokenizer checks only (no Llama run completed): Llama starts every text with `<|begin_of_text|>` (Olmo 3 and Qwen 2.5 tokenizers add none, checked, so these changes are no-ops for them): `scoring.py` does not add a second one when the prompt is a chat template's text; `prompt_tune_base.py` puts it before the soft vectors and `phase4_hidden_states.py` keeps it first when inserting them (training and scoring now match); pad-token fallback to end-of-text in `phase1_instruct_sample.py`/`phase4_selfdesc_elicit.py` (Llama Instruct has none). Fit tags `lsdo/lsdt/lsd/lsd_soft/lsd_all` in `phase4_bases_fit.sh`. Driver `scripts/phase4_llama.sh`: the whole Qwen pipeline in order, each step skipped when its output exists (resume by rerunning).
**What failed:** the download (`snapshot_download(cache_dir=$SCRATCH/hf_stage/hub)`) left the four weight shards outside the model folders, so the CFS copy had only small files; stage 1 failed in a minute ("snapshot incomplete"); jobs 60707887/60707907/60707908 cancelled. No Llama results exist.
**Note:** Llama 3.1's chat template adds "Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024" to every system message; kept (how Instruct is normally used). Llama 3 and Olmo 3 share the first ~100k token ids.
**Next:** on the new cluster, download into the normal `$HF_HOME` cache, load both models offline to confirm, then `scripts/phase4_llama.sh`. Before running, check that soft-context hold-out scores from `prompt_tune_base.py` match the `lp_sp_*` hidden-state scores (tests the beginning-of-text handling).

**Update 2026-11-02:** held Experiment 2 jobs 60356084-86 (7.5 h limits) cancelled at the author's request before the cluster move; on the new cluster they need re-splitting into jobs of 6 h or less.

## 2026-11-02 — Moving to RunPod: setup done, old reference number reproduced exactly
**Bottom line:** the RunPod pod reproduces the NERSC reference score bit for bit (Olmo 3 Instruct self log P/token -0.8937 over 2400 answers, every per-token value identical). Environment, results copy and Claude memory are in place; the last model downloads are running.
**Machine:** 1x A100-SXM4-80GB (driver 595.91, CUDA 13.2). `/workspace` is the persistent RunPod network volume; its quota is NOT shown by `df` (which reports the whole 615 TB storage cluster): it was ~200 GB and filled during the model downloads ("Disk quota exceeded"); the author raised it to 2 TB. `sbatch`/`srun` exist but no Slurm cluster is configured (`sinfo` fails): scripts must be run directly, on one GPU, not 4.
**Step 1 (copy):** the repo arrived as a git clone; the author rsynced the gitignored files from Perlmutter (excluding .git and env.sh). Checked: results/ 4.1 GB (apparent size; `du` without --apparent-size says 4.7 GB on this filesystem), 133 prefix.pt files, logs/, data/, the two root files; `git status` shows only the uncommitted writeup edits, env.sh and NOTES.md modified. Claude memory copied to `/workspace/agent_home_p2/claude/projects/-workspace-psm-mixture-test/memory/`.
**Step 2:** `env.sh` cluster block: HF_HOME=/workspace/hf_cache, SCRATCH=/workspace/scratch, ENV_BIN unchanged (/workspace/envs/ml/bin). Also fixed: `env.sh` read an unset `$PYTHONPATH`, which aborted `transfer/download_models.sh` (it uses `set -u`); now `${PYTHONPATH:+:$PYTHONPATH}`.
**Step 3:** no conda on the pod; env built with uv instead: `uv venv --python /usr/bin/python3.11 /workspace/envs/ml`, then `uv pip install -r transfer/requirements-freeze.txt --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match` and `uv pip install -e .`. Deviations: **Python 3.11.13, not 3.11.15** (the installed uv 0.9.0 has no 3.11.15); **dropped nersc-pymon** (NERSC-only). torch 2.5.1+cu121, transformers 5.17.0, huggingface_hub 1.32.0, numpy 2.4.6, scipy 1.17.1 exactly as on NERSC.
**Step 4:** done. `bash transfer/download_models.sh` (logs `logs/download_models_runpod*.log`): all 14 snapshots match model_revisions.txt and pass the offline check (222 GB in $HF_HOME). HF login as oz-amram (Llama access confirmed); the token was saved to the default cache and copied to `$HF_HOME/token`. Fixed in the script: the offline check flagged both Llama models as incomplete only because of the `original/*` files (Meta-format duplicate weights) that the download deliberately excludes; the check now excludes them too. Both Llama models then loaded on the GPU (8.03B parameters, sensible greedy text, first token 128000 = beginning-of-text).
**Step 6:** `python scripts/phase4_self_tokens.py --run instruct_unknown_casual_v1 --model allenai/Olmo-3-7B-Instruct --tag check_newcluster --out <scratchpad>` (2 min 47 s on 1 A100; output kept outside results/, so nothing to delete): -0.8937/token; vs the saved `results/phase4/self_tokens/instruct_unknown_casual_v1.npz` same token ids, same offsets, max per-token difference 0.0000. Exact agreement is plausible: Perlmutter also used A100 80 GB with the same torch/transformers.
**Next:** adapt `scripts/phase4_llama.sh` to run on 1 GPU without Slurm and estimate runtime (~40 GPU-h of NERSC work, so roughly 40+ h wall-clock here); NERSC model cache and scratch hidden states can now be deleted (TRANSFER.md step 8, the author's call).

## 2026-11-02 — Llama 3.1 8B replication on RunPod (author's request; in progress)
**Goal:** the Llama replication prepared on NERSC (entry above; `scripts/phase4_llama.sh`), run on RunPod: this pod's 1 A100 plus up to 3 single-GPU worker pods from `sudo gpu-launch` ($300 budget, 12 h per pod).
**Decision (author, 2026-11-02): drop the fits that do not fit on one GPU.** With Llama's hidden size (4096) and ~46k fit positions (Qwen numbers), fit A holds K x 0.38 GB on the GPU: self-descriptions (lsd, 261 contexts) ~99 GB, lsdt ~91 GB, lsd_all ~113 GB, Big Five ~107 GB, Axis/three/all 196-274 GB; worker GPUs have at most 96 GB (NERSC split these over 4 GPUs). Options offered: a 4-GPU pod, or streaming hidden states from CPU memory (code change); the author chose to drop them. **Kept fits:** lsdo (21 contexts), lsd_soft (36), legacy87 (87), olmospec (8). Consequences:
- Soft-context selection had used the union of (a) arithmetic greedy top 20 and (b) top 20 |c| of the lsd geometric fit. (b) is gone; my choice: (a) only with k=35 (same count as Olmo/Qwen; greedy is nested, so its first 20 are what (a) would have given). Starting point mattered little on Olmo/Qwen (spread 0.02/token). New flag `phase4_selfdesc_select.py --arith-only`, with --fit = the lsdo fit (same answers and fit questions). Default (union) behaviour unchanged by reading of the code; NOT re-tested (Qwen hidden states are on NERSC scratch).
- Generic-basis hidden states only for legacy86 + olmo_spec (+ generic); row filter for those fits ROWCTX=legacy86,olmo_spec,LO,LT (Qwen used all generic families + sd), so the Llama generic fits use a slightly different answer filter than Qwen's.
- `phase4_bases_fit.sh`: DEVICES env var (default 0,1,2,3 as before).
**Measured on this pod:** /workspace reads ~800 MB/s (direct I/O), container memory limit 250 GB.
**Stage 1** (started 21:54 UTC on the main pod's GPU): `phase1_instruct_sample.py --instruct-model meta-llama/Llama-3.1-8B-Instruct --base-model meta-llama/Llama-3.1-8B --n-per-question 8 --framing unknown --register casual --seed 0 --out results/phase4/llama/runs/llama_instruct_unknown_casual_v1` and `phase4_selfdesc_elicit.py --model <Instruct> --prefix llama_ --part k/3` (k=0,1,2, same seeds as the driver) + `--merge`; logs `logs/pl_runpod_*.log`.
**Worker pods:** first `gpu-launch start` at ~22:00 UTC: no stock for any GPU type (nothing created); retrying every 12 min (`logs/gpu_launch_runpod.log`).
**Stage 1 result (22:33 UTC, 39 min on 1 A100):** 2400 answers, 20 overall + 240 trait self-descriptions (134-201 / 121-207 words). Answers look normal (casual advice), none empty. **Llama answers are longer: mean 67.0 generated tokens, 20% hit the 80-token limit (Qwen: 48.6, 1%)**, so a fifth end mid-sentence; same setting as Olmo/Qwen, kept for comparability. Llama Instruct self score (`phase4_self_tokens.py`, stripped answer): **-0.926/token** (Olmo -0.898, Qwen -0.733), `results/phase4/self_tokens/llama_instruct_unknown_casual_v1.npz`.
**Hidden states:** 160,787 answer positions (1.32 GB per context); 101 s per context on 1 A100 (sd basis: 261 contexts = ~7.3 h on one GPU). Fit-half GPU memory then ~0.53 GB per context: legacy87 ~46 GB, lsd_soft ~20 GB, lsdo ~11 GB (all fit). Disk for everything kept ~515 GB.
**Running:** main pod, `scripts/phase4_llama_runpod_queue.sh` (log `logs/pl_runpod_queue.out`) after sd part 0/4; whole pipeline on one GPU ~22 h if no worker pods start. Pod launch: no stock at 22:00, 22:12, 22:24, 22:36 UTC.
**Incident (22:39 UTC, my error):** the queue was meant to wait for the first part-0/4 hidden-state run, but `pgrep -f` matched my own shell instead of that run, so the queue started immediately and ran part 0/4 a second time alongside it. Both wrote `h_lsdo_actor_0.u16.tmp`; one crashed on the rename, then the queue went on to part 1/4. Since then parts 0/4 (original run) and 1/4 (queue) run side by side on the main GPU (~155 s per context each, i.e. more throughput than one at a time). Checked `lsdo_actor_0`: 160,787 rows, no all-zero rows, no NaN, mean log P in line with its neighbours. To do before the lsdo fit: recompute its log P from the stored hidden states and compare with `lp_lsdo_actor_0.npy`.
**Worker pod 7b6j0bvnt6x4fx** (A100 SXM 80 GB, $1.79/h): started 23:12 UTC (try 6). SSH (`root@195.26.233.76 -p 41692`) still "connection refused" after 20+ min although status says RUNNING; per the launcher instructions, stopped working on it and asked the author.
**2026-11-03 ~00:00 UTC — main pod only (author: worker launch not working, run everything here for now).** Worker pod 7b6j0bvnt6x4fx left running for the author's investigation (not stopped by me; bills $1.79/h while idle).
- `lsdo_actor_0` check done: log P recomputed from the stored hidden states (bf16 lm_head, as in the script) matches `lp_lsdo_actor_0.npy`, max 0.00002 nats/answer over 2400 answers (control `lsdo_describe_0`: max 0.00003). File is fine.
- Second lane: when part 0/4 (pid 17938) exits, `phase4_llama_runpod.sh hidden <sd>,generic 7/8` runs beside the queue (log `logs/pl_runpod_lane2.out`); 7/8 is half of the queue's 3/4, so the queue later skips those contexts. Guarded so it does not start if the queue is already on 3/4 (pattern checked not to match itself, unlike the 22:39 incident). Expected: sd hidden states done ~04:00 UTC, whole pipeline ~mid-day 2026-11-03 UTC.
**2026-11-03 15:00 UTC — fits failed overnight; fixed and relaunched.** Hidden states all done (261 sd + 93 legacy86/olmo_spec, no NaN; sd done 04:34), framing screen (4 groups) and contamination check done (05:27). **But every fit (lsdo, lsd_soft, legacy87, olmospec) failed within seconds**: `phase4_geom_fit.py` line 52 (`unembedding`) called `snapshot_download(model)` without excluding `original/*`, so huggingface_hub rejected the Llama snapshot as incomplete (same cause as the download-check fix). Selection then had no fit, so only the generic soft context was trained; the `*_soft` families came out empty. The queue logged "done" for each step (the fit wrapper does not stop on errors), so the GPU idled 06:58-15:00 UTC (~8 h lost). Fix: `ignore_patterns=["original/*"]` in that call (no effect on Olmo/Qwen, which have no original/ folder). Relaunched with output checks that stop the chain: `scripts/phase4_llama_runpod_rest.sh` (log `logs/pl_runpod_rest.out`). lsdo fit confirmed running: K=21, 960 fit / 1200 hold-out answers, 64,080 fit positions; generic check from hidden states -1.5789 vs stored -1.5793/token, max per-answer difference 0.83 nats (Qwen 0.60, Olmo 0.17: same kind).
- Soft-context training takes ~5 min each on 1 A100 (generic: 04:34-04:39, L=60 as Qwen), so the 35 selected ~3 h. Expected finish of everything ~21:00 UTC.
- Worker pod 7b6j0bvnt6x4fx is gone (status: no pods running); total gpu-launch spend $1.62.
**2026-11-03 15:20 UTC — Claude restart (author, to enable the launcher) killed the chain mid-lsdo-fit.** Relaunched with `setsid nohup` (survives session restarts). Soft-context training now claims each context with an atomic `mkdir <dir>.claim` (tested: 20 racing processes on /workspace, exactly 1 winner), so the main pod and any worker can all run `phase4_llama_runpod.sh soft 0/1` without duplicating work; the chain waits up to 2 h for contexts claimed elsewhere, then stops and names a stale claim. Launcher: no stock at 15:2x for any GPU type ($0 spent on the new profile); retrying every 12 min until 17:30 UTC (later a worker saves too little).
**16:06 UTC lsdo fit** (21 overall self-descriptions + generic; hold-out, per token): geometric free -1.4288 ± 0.0096, arithmetic mixture -1.4760 ± 0.0098. `results/phase4/exp1/llama_instruct_unknown_casual_v1__llamabase/bases/lsdo/fits_A.json`.
**16:44 UTC selection** (`--arith-only --k 35`, 38 min on CPU, 960 fit answers): 35 picked, first: goofy, witty, tactful, callous, stream-of-consciousness, spiritual, accessible, calculating (`results/phase4/llama/selfdesc/selected.json`). Soft-context training started 16:44 (~3 h on 1 GPU).
**17:22 UTC:** launcher retries ended (11 tries 15:2x-17:22, no stock for any GPU type, $0 spent); everything continues on the main pod. Soft contexts: 5/36 done.
**Worker pod 3k6eu7iuq2qza2** (A100 SXM 80 GB, $1.79/h): started 20:46 UTC 2026-11-03 (15-min retry loop, try 10); job: none (planned legacy87/olmospec fits). SSH `root@3k6eu7iuq2qza2.runpod.internal` -> 10.0.49.106:22 "Connection timed out" 20:47-20:57 (previous pod: public IP, "Connection refused"). Stopped on the author's instruction at 22:03 UTC. Launcher spend total $2.28 of $300 (this profile). Author: keep running on the main pod only.

## 2026-11-17 — Llama 3.1 8B replication: finished (results)
**Bottom line:** the chain finished 2026-11-04 00:11 UTC; all four kept fits have hold-out results. The Olmo/Qwen pattern replicates on Llama: geometric (logit-linear) beats the arithmetic mixture on hard-text bases (legacy87 +0.096 nats/token, overall self-descriptions +0.046), soft contexts give the biggest gain over the generic header (+0.41/token, Olmo +0.38, Qwen +0.37), and on soft contexts the arithmetic mixture is slightly better than geometric (-0.005; Olmo -0.010, Qwen -0.003).
Hold-out log P/token (1200 answers; ± = standard error; "geo - arith" uses the paired standard error stored in fits_A.json; geometric variant = lambda chosen on validation):
| basis | Llama generic | Llama arith | Llama geo | Llama geo-arith | Olmo geo-arith | Qwen geo-arith |
|---|---|---|---|---|---|---|
| overall self-desc (21) | -1.579 ± 0.010 | -1.476 ± 0.010 | -1.430 ± 0.010 | +0.046 ± 0.001 | +0.035 ± 0.002 | +0.083 ± 0.002 |
| soft contexts (36) | -1.579 | -1.168 ± 0.009 | -1.173 ± 0.009 | -0.005 ± 0.001 | -0.010 ± 0.001 | -0.003 ± 0.001 |
| legacy87 (87) | -1.579 | -1.491 ± 0.010 | -1.395 ± 0.009 | +0.096 ± 0.002 | +0.117 ± 0.004 | +0.127 ± 0.003 |
| OLMo spec (8) | -1.579 | -1.526 ± 0.010 | -1.515 ± 0.010 | +0.011 ± 0.001 | +0.044 ± 0.002 | +0.032 ± 0.002 |
Caveats (differences from Olmo/Qwen, all from dropping the large fits): soft contexts selected by arithmetic greedy only (k=35); generic-basis answer filter = legacy86+olmo_spec+Llama self-descriptions only; no full self-description, Big Five, Axis, three or all fits. Absolute log P is not comparable across families (different tokenisers); compare differences. One seed per fit (as Olmo/Qwen). 20% of Llama answers are cut at the 80-token limit (Qwen 1%).
Timings on 1 A100: lsdo 45 min, lsd_soft 34 min, legacy87 61 min, olmospec 24 min; soft contexts ~9.5 min each.
Outputs: `results/phase4/exp1/llama_instruct_unknown_casual_v1__llamabase/bases/{lsdo,lsd_soft,legacy87,olmospec}/fits_A.json`; framing screen + contamination `results/phase4/llama/framing/` (not yet summarised); soft contexts `results/phase4/llama/selfdesc/prefix/`. Hidden states `$SCRATCH/persona_phase4/hidden/llama_instruct_unknown_casual_v1__llamabase` (~515 GB).
**Summary script:** `python scripts/phase4_llama_summary.py` -> `results/phase4/llama/summary.json` (all three families, four shared bases; geo - arith standard errors are the paired ones stored in fits_A.json). Shares of the generic -> Instruct gap closed (Instruct self on the same hold-out answers: Olmo -0.898, Qwen -0.733, Llama -0.933), arithmetic / geometric: overall self-desc Olmo 11/15%, Qwen 7/24%, Llama 16/23%; soft contexts 44/43%, 74/74%, 64/63%; legacy87 4/18%, 4/29%, 14/28%; OLMo spec 4/9%, 1/7%, 8/10%. Llama's gap (0.65/token) sits between Olmo (0.86) and Qwen (0.50); its arithmetic mixtures close more than the other two (legacy87 14% vs 4%).
**Framing screen (Llama base, generic + 16 random Llama self-descriptions, seed 0, all 16 happen to be trait descriptions; test half, 300 answers, nats per reference token):** `python scripts/phase4_framing_summary2.py --dir results/phase4/llama/framing --files "base:scores_llamabase_0.json,scores_llamabase_1.json,scores_llamabase_2.json,scores_llamabase_3.json" --nref scores_llamabase_0.json` -> `results/phase4/llama/framing/summary2.json`.
| framing | generic | mean description | description - generic |
|---|---|---|---|
| transcript | -1.547 | -1.452 | +0.095 ± 0.003 |
| turn_bare | -1.591 | -1.452 | +0.140 ± 0.003 |
| turn_describe | -1.579 | -1.453 | +0.126 ± 0.003 |
| card | -1.563 | -1.471 | +0.092 ± 0.003 |
| hyb_char | -1.539 | -1.457 | +0.082 ± 0.003 |
| story_chat | -1.610 | -1.492 | +0.118 ± 0.003 |
| play | -1.658 | -1.500 | +0.158 ± 0.004 |
| story | -1.710 | -1.536 | +0.174 ± 0.003 |
Descriptions help in every framing (Qwen: they hurt in transcript, -0.042; Olmo: helped, +0.06). Transcript, turn_bare and turn_describe tie for the best mean-description score (-1.452/-1.452/-1.453; turn_bare beat transcript by 0.045 on Olmo, 0.065 on Qwen, mean description, test half). Fiction framings (play, story) worst, as on both other models. Best single description in the transcript framing: lsdt_reactive -1.440 (Qwen's best was also "reactive").
**Contamination check (generic header, test half, 300 answers, `mean` field of contam_*.json):** Llama base transcript -1.547, Llama base through the Llama Instruct chat template -1.533 (chat - transcript +0.013 ± 0.003 per reference token, paired; Olmo -0.095 ± 0.004, Qwen -0.041 ± 0.004), Llama Instruct -0.916. So Llama base is slightly HELPED by its Instruct chat template, the only one of the three; small, but a sign the base model has seen that format (its special tokens are in the base tokenizer). Not conclusive by itself.
**Next:** decide whether the 515 GB of Llama hidden states can be deleted (volume at 1.2 TB of 2 TB). Not run (dropped): full sd, Big Five, Axis, three, all; cross-family pairs (out of scope).
**Write-up:** `claude_writeup_persona_mixture.md` section 4 completed (other bases, self-descriptions, soft contexts and verbalisation, framings, SFT as scorer, Qwen + Llama replication table from `phase4_llama_summary.py`); TL;DR, models line, section 5 and caveats updated. Experiment 2 marked not run.

## 2026-11-18 — Soft-context length sweep (author's request; running)
**Goal:** a proper sweep of the base model + soft context's hold-out log P on Instruct's answers vs context length L, to see whether the improvement stays log-linear in L. Existing points are not a controlled sweep: the Phase 1 split runs (L 8/32/128/512, -1.377/-1.377/-1.372/-1.367) vary batch (2 vs 8), epochs (6-10), init and lr, and one L512 run never finished; the Phase 4 split (Experiment 3) has L 16/64/128 only (-1.410/-1.397/-1.397, one seed).
**Author's choices:** Olmo only; L up to 2048; 3 seeds.
**Design (my choices):** Experiment 3 recipe on the Phase 4 split (rows `results/phase4/exp3/rows/instruct_unknown_casual_v1.jsonl`, hold-out `basis_test_qids.json` = 1200 answers, generic-header init = first min(L, 60) header tokens then random tokens, batch 8, max_len 192, lr 5e-3, up to 6 epochs, epoch chosen on validation), L = 1, 2, 4, ..., 2048, seeds 0-2; plus lr 2e-2 at L 1-16, seed 0 (are short prefixes under-trained?). Old points are NOT reused (see reproducibility below). Script `scripts/phase4_lensweep.sh` (claims per run, so worker pods could join), log `logs/lensweep_<host>.out`, outputs `results/phase4/lensweep/L<L>_lr<lr>_s<seed>/`; summary + figure `python scripts/phase4_lensweep_summary.py` -> `results/phase4/lensweep/{summary.json,fig_lensweep.png}`.
**Reproducibility check:** rerunning Experiment 3's L16 seed 0 for 2 epochs on this pod gives hold-out -1.42228 / -1.40943 vs stored -1.42499 / -1.41035 (same args, same package versions). So training is not bit-reproducible across runs (presumably GPU nondeterminism; not confirmed), at the 0.001-0.003 level, about the size of the effects being measured; hence 3 seeds and no mixing with old runs. 50 s/epoch at L16, as on NERSC.
**L = 2048 runs out of memory** at batch 8 (80 GB A100). Added `prompt_tune_base.py --grad-checkpointing` (default off; recomputes activations in the backward pass, same gradients up to bf16 rounding; puts the model in train mode, which only switches checkpointing on since OLMo 3 has attention_dropout 0). `scripts/phase4_lensweep_ckpt_check.py` compares loss and prefix gradient with/without on one real batch (pass: loss diff < 1e-3, cosine > 0.999) and tries one L=2048 step; `scripts/phase4_lensweep_2048.sh` runs that check and, if it passes, L=2048 x 3 seeds with checkpointing. Queued after the main sweep (waits on its PID). The main sweep will also hit OOM at L=2048 for seeds 1 and 2 (~2 min each, harmless; the 2048 script clears those claims).
**Expected (estimate from measured 49-84 s/epoch at L 16-128, ~6.5 min/epoch at L 1024):** main sweep ~8 h on this A100, L=2048 with checkpointing ~3 h more; done around 08:00 UTC 2026-11-19.
**Sweep progress:** L1024 seed 0: hold-out -1.3951 (best epoch 2, 46 min; strong overfitting after epoch 2: train NLL 0.97 by epoch 6). Second lane (`scripts/phase4_lensweep_lane2.sh`, ascending L, same claims; started by a watcher only when GPU memory < 45 GB) was held at L512 (49 GB in use); retries at L256.

## 2026-11-18 — Workspace lenses (J-Lens / J++ Lens) for reading the soft contexts (author's request; setup)
**Goal:** read what the learned soft contexts make the base model "disposed to say" with a workspace lens, instead of nearest tokens / verbalisation (section 4: nearest tokens = the starting description; verbalisation recovers <= 18% of the gain).
**Methods found:** J-Lens (Gurnee et al. 2026, transformer-circuits.pub/2026/workspace; readout unembed(J_l h), J_l = averaged Jacobian of the final-layer residual w.r.t. the layer-l residual; code github.com/anthropics/jacobian-lens) and J++ Lens (Ayonrinde & Lindsey, LessWrong 2026-10-08: per-cluster "expert" Jacobians with learned signed weights, LRP backward pass, readout restricted to tokens with a letter/digit; reported recall@10 55% vs 36% for the J-Lens on Qwen3.6-27B; code github.com/safety-research/jpp_lens, Apache-2.0).
**Available lenses:** `koayon/jpp-lenses` (J++): Olmo only for olmo-3-1125-32b, not our 7B. `neuronpedia/jacobian-lens` (author's pointer): **J-Lens for allenai/Olmo-3-1025-7B (our base model)**, also Llama-3.1-8B base; J++ for Olmo 3 32B only. Olmo/Neuronpedia lens licence Apache-2.0 (Llama folders: Llama 3.1 licence). Olmo 7B J-Lens: fitted with Anthropic's code, 568 WikiText-103 prompts x 128 tokens (stopped at convergence), skip_first 16 positions, target layer 31, transformers 5.17.0, ~24 s/prompt on an RTX PRO 6000 (~3.8 h). Downloaded to `/workspace/lenses/neuronpedia/olmo-3-1025-7b/`: J maps (4096x4096 fp16) for source layers 0-30. Layer l = output of block l (forward hook), so the input embeddings (where the soft vectors sit) are not covered: we read the residual at the prefix positions from block 0 on, and the answer positions. Caveat: prefix positions are the first positions of the sequence, which the lens skipped during fitting (first 16); readouts there may be unreliable for short prefixes.
**Code:** `safety-research/jpp_lens` cloned to `/workspace/jpp_lens` (commit d7efc0d), own env `/workspace/envs/jpp` (uv, Python 3.12, its pinned packages, cu128 torch). Its tests: 260/264 pass; the 4 failures (expert merge/weights) are device mismatches because a GPU is visible (README runs tests CPU-only); with `CUDA_VISIBLE_DEVICES=` all 43 pipeline tests pass.
**J++ for Olmo 7B would need:** fitting (pipeline stages fit-router, fit-shard, merge, fit-weights, evaluate; paper: 64 x 128-token WikiText sequences, 8 experts per layer, ~9 H200-h for a 27B model; for 7B a few A100-hours, unmeasured) plus which latent-variable items Olmo 7B base answers correctly (the repo's correctness table is for Qwen3.6-27B only; would need a small script).
**Running:** `scripts/jlens_sanity.py` (CPU, jpp env): does the Olmo 7B J-Lens read out the intermediate on the multihop/multilingual items, vs the logit lens? -> `results/phase4/jlens/sanity.json`.
**J-Lens sanity check results (`scripts/jlens_sanity.py`, CPU, jpp env; `results/phase4/jlens/sanity{,_strip}.json`):**
- The evaluation prompts end in a space; Olmo's tokenizer makes it a lone " " token and the model then predicts e.g. "3" for the Carnival/ocean item. With the space stripped (`--strip`) it answers " Atlantic". Model checked: identical top-5 predictions in our main env (transformers 5.17.0) and the jpp env (5.14.1), so the setup is fine ("The capital of France is" -> " known" is Olmo base's own preference, in both).
- Stripped prompts, intermediate in the top 10 at any of layers 4-28 (last position, full vocabulary, rough matching, all items not only those answered correctly): multihop (93 items; model top-1 correct 24%) J-Lens 41% vs logit lens 34% (J-Lens only 10, logit only 4, exact McNemar p = 0.18); multilingual (107; model correct 36%) 92% vs 86% (9 vs 3, p = 0.15; items list prompt words as intermediates, so easy). Unstripped: 38% vs 33%, 76% vs 75%.
- Mid-layer J-Lens readouts are dominated by punctuation/newline tokens (".\n", ":", ","): the noise J++'s Readout Filtering removes.
- **Reading:** the pre-fitted Olmo 7B J-Lens is at best modestly better than the logit lens on this model, not clearly so in this test. Not a strong enough tool on its own to interpret soft contexts; a J++ fit (and the paper's scorer, restricted to items the model answers) is the better test.

## 2026-11-19 — Length sweep, first pass; J++ fit for Olmo 7B queued
**Sweep at 00:11 UTC** (`python scripts/phase4_lensweep_summary.py`; seed 0 at every L except 2048, seed 1 at L 1-16 and 512; hold-out log P/token, mean over the seeds done): L1 -1.4172, L2 -1.4107, L4 -1.4111, L8 -1.4096, L16 -1.4078, L32 -1.4070, L64 -1.4021, L128 -1.3963, L256 -1.3955, L512 -1.3995 (two seeds: -1.3951, -1.4039), L1024 -1.3951. Whole range 1 -> 1024: +0.022 nats/token over 10 doublings (~0.002 per doubling on average). Shape so far: a jump 1 -> 2, slow rise to 32, steeper 32 -> 128 (+0.011), then flat 128 -> 1024. Seed spread 0.001-0.006 (L512 seeds differ by 0.009), the same size as the per-doubling changes, so the shape is not settled; seeds 1-2 are still running. At L 1-2 the best epoch is the last one (6): possibly under-trained; the lr 2e-2 check at L 1-16 will tell.
**Problem:** L1024 seed 1 ran out of memory at 23:13 because the second lane shared the GPU (58 GB in this process). Seed 2 will likely hit the same. `scripts/phase4_lensweep_2048.sh` (not yet started; edited before it runs) now waits for lane 2 to finish, reruns any L1024 seed without a result alone on the GPU, then does L2048. The two failed attempts are not results (no result.json); they are cleared before rerun.
**Note:** I did not report the seed-0 curve when it completed (~23:00) because I had set nothing to notify me; found at 00:11.
**J++ for Olmo 7B (author: "do the J++ fit"):**
- Correctness labels with the library's grader, prompts as written (the paper's protocol): earlier rough check showed trailing spaces matter little for Olmo 7B (multihop correct 21/93 as written, 22/93 stripped). `scripts/jpp_grade_correctness.py` (CPU) -> `results/phase4/jlens/model_correctness_olmo3_7b.csv`.
- Fit: `scripts/jpp_fit_olmo.sh` (recipe: 8 experts, router on 1,000 WikiText prompts, LRP expert Jacobians on 64 x 128 tokens, weights on 20% of scoreable items, seed 0; source layers 4,8,...,28 = the recipe's relative depths for 32 blocks) -> `/workspace/lenses/jpp_olmo3_7b/`; then `evaluate` on the held-out 80% for J++, its pooled R-Lens, the Neuronpedia J-Lens and the logit lens. Estimate 3-5 A100-hours (scaled from the paper's 7.3 H200-h for 27B; not measured). Queued after the sweep's last step (waits on PID 159631; log `logs/jpp_fit_olmo.out`), because it needs ~25-30 GB and L1024/L2048 need the whole GPU. Expected start ~10:00 UTC, results ~15:00 UTC (estimates).
- Correctness (library grader, 9 min CPU): multihop 21/93, multilingual 36/107, order-ops 29/55, poetry 25/98 correct; association (102) and typo (96) ungradeable, always kept. With the recipe's five macro evals ~280 scoreable items, ~56 for the expert weights (20%), ~225 held out; multihop only ~4 fit / ~17 held-out items, so per-eval numbers there will be noisy.
**2026-11-19 01:30 UTC — RL-step geometric fits (author's request):** `scripts/phase4_exp3_geom_rl.sh` (steps 100/200/300 one at a time: base hidden states under the 87 contexts, `phase4_geom_fit.py --run instruct_step<s>_v1 --fits A --devices 0` as for SFT/DPO, then delete that step's h_*.u16 (~115 GB) once the fit has hold-out results; disk at 1.7 of 2 TB). Planned for a worker pod: pod 5y78a0msamuwxo (gpu1, A100 SXM 80 GB, $2.19/h) started 01:17:58 UTC; SSH `root@5y78a0msamuwxo.runpod.internal` -> 10.0.186.128:22 timed out for 10 min (third pod in a row with no SSH); by 01:28 status showed "no pods running" (not stopped by me; the watchdog only stops pods past 12 h or over budget). Cost ~$0.37. Per the launcher instructions, no further retries; queued on the main pod after the J++ fit instead (waits on PID 210600; log `logs/p4e3_rl_<host>.out`). Expected ~8-9 GPU-h, so done around midnight UTC (estimate).
**Worker pod xw6f8tol4oiw0m** (A100 SXM 80 GB, $2.19/h, gpu1 profile): started 01:39 UTC 2026-11-19 (retry try 2) for `scripts/phase4_exp3_geom_rl.sh`; SSH `root@xw6f8tol4oiw0m.runpod.internal` -> 10.1.114.167:22 "Connection timed out" 01:40-01:50 (same as pod 3k6eu7iuq2qza2 on 11-03). Stopped by me ~01:51 UTC, no job run. gpu1 spend then ~$246.4 of $300. RL-step fits fall back to the main pod after J++.
- RL-step geometric fits: an identical waiter (PID 229491, started 01:29 UTC, not by this session) already queues `scripts/phase4_exp3_geom_rl.sh` after the J++ fit; the copy I started at 01:50 was removed to avoid two runs writing the same hidden states.

## 2026-11-19 14:30 UTC — Length sweep finished; J++ Lens for Olmo 7B fitted; RL step 100 geometric fit
**Length sweep (all 41 runs; `python scripts/phase4_lensweep_summary.py`, `results/phase4/lensweep/{summary.json,fig_lensweep.png}`):** hold-out log P/token, mean ± sd over 3 seeds at lr 5e-3: L1 -1.4180 ± 0.0014, L2 -1.4125 ± 0.0032, L4 -1.4112, L8 -1.4098, L16 -1.4085, L32 -1.4101, L64 -1.4039, L128 -1.4008 ± 0.0045, L256 -1.4008, L512 -1.3994, L1024 -1.3999 ± 0.0095, L2048 -1.4030 ± 0.0078. Straight line in log2 L: +0.0015/doubling, but residuals are systematic (negative at 1 and 2048, positive 64-512) and per-doubling slopes are mostly within ~1 SE of zero above L2.
- **lr 2e-2 check (seed 0):** L1 -1.4117, L2 -1.4087, L4 -1.4084, L8 -1.4033, L16 -1.4009: better than lr 5e-3 at every short length (by 0.004-0.008 at L8-16), and L16 at lr 2e-2 already equals the long-prefix plateau (~-1.400). At lr 5e-3 the best epoch at L1-2 is the last one (6). So the slow rise from L1 to ~L128 at lr 5e-3 is at least partly under-training of short prefixes, not a capacity effect. At L64 the learning rate does not matter (Experiment 3: lr 2e-2 -1.3960 vs 5e-3 -1.3973), so the plateau itself is not an lr artefact. **Reading:** not log-linear: once trained properly, the soft context saturates at ~-1.400 by L ~ 8-16 vectors, and nothing up to 2048 vectors moves it beyond seed noise (sd 0.002-0.009). Caveat: lr 2e-2 only one seed and only L <= 16.
- Gradient checkpointing check (`logs/lensweep_ckpt_check.log`): loss and gradient identical with and without (difference 0.0, cosine 1.000000); L2048 step peak 37.4 GiB. L1024 seeds 1-2 rerun alone after OOM (-1.4108, -1.3937). OOM attempts left no results.
**J++ Lens for allenai/Olmo-3-1025-7B (`scripts/jpp_fit_olmo.sh`, `/workspace/lenses/jpp_olmo3_7b/`, log `logs/jpp_fit_olmo.out`):** router 7.5 min, LRP expert Jacobians on 64 prompts 52 min, merge+weights 2.5 min, eval 1.5 min: ~1 GPU-hour on the A100 (my 3-5 h estimate was ~4x too high). Held-out evaluation (220 items, recall@10, best of layers 4-28, Readout Filtering, macro over the 5 evals, item-weighted; same numbers recomputed from the raw ranks):
| lens | recall@10 | recall@1 |
|---|---|---|
| J++ (this fit) | 0.515 | 0.279 |
| pooled R-Lens (same fit, no expert weighting) | 0.415 | 0.151 |
| Neuronpedia J-Lens | 0.378 | 0.139 |
| logit lens | 0.330 | 0.124 |
J++ - J-Lens +0.136 (95% item bootstrap within eval [+0.087, +0.189]); J++ - logit +0.184 [+0.143, +0.230]; J-Lens - logit +0.047 [+0.004, +0.090]. Close to the paper's Qwen3.6-27B numbers (J++ 0.552, J-Lens 0.357). Per eval (recall@10, J++ / J-Lens / logit): typo 0.92/0.56/0.38 (77 items), association 0.23/0.09/0.07 (77), multilingual 0.56/0.49/0.46 (29), multihop 0.76/0.65/0.65 (17; R-Lens 0.82), poetry 0.10/0.10/0.10 (20). Most of the gain is on typo and association; multihop and poetry have few items. The expert weights were fitted on 20% of items from the same five tasks (paper's protocol), so held-out items are new items, not new tasks.
**RL step 100 geometric fit (`scripts/phase4_exp3_geom_rl.sh`, run by the waiter started 01:29 UTC; log `logs/p4e3_rl_bcdccfe9d9af.out`):** self -1.030, arithmetic -1.815, geometric -1.720 ± 0.015 (lambda 1e-4, zero centre): gap 0.690, closes 12% of the arithmetic-to-self distance; top weights hhh +1.64, e14 +0.96, e34 -0.84, neutral +0.68, e60 +0.62, e59 -0.58, the same personas as at every other stage. Recomputed SFT/DPO/final gaps 0.564/0.685/0.706 (stage table: 0.565/0.686/0.706). Step 100 sits between DPO and final as expected. Steps 200 and 300 running (~2.6 h each).

## 2026-11-19 ~17:15 UTC — J++ readout of the soft contexts; lr scan for short prefixes (author's requests; queued)
**J++ readout** (`scripts/jpp_softcontext_readout.py`, jpp env with `PYTHONPATH=/workspace/jpp_lens/src:src:scripts`, `HF_HOME=/workspace/hf_cache` set explicitly: the new tmux shell has HF_HOME=/workspace/.cache/huggingface): for each soft context vs its starting text (same embeddings path; untrained = exactly the text header), with our J++ lens at layers 4-28 and Readout Filtering: (1) top-10 token counts over the prefix's own positions (positions < 16 separate: outside the lens's fit range); (2) on 200 hold-out answers, mean change of the lens log-softmax at answer-predicting positions, with the model's real output change as ground truth (Spearman and top-50 overlap per layer). Contexts: the 35 self-description soft contexts + generic, Experiment 3's final L128, and sweep L16/64/128/1024 (seed 0). Uses each run's saved prefix = best epoch by validation. CPU smoke test (Experiment 3 L128, 4 answers): output change up " honest", "—if", "—even", " honestly", " honesty" (matches the token-gain analysis: em dash, "Honestly"); lens-vs-output Spearman 0.05 (layer 4) -> 0.41 (16) -> 0.53 (28).
**lr scan** (`scripts/phase4_lensweep_lr.sh`, two lanes): L 1-128 x lr {5e-3, 1e-2, 2e-2, 5e-2, 1e-1}, 12 epochs, seed 0; lr chosen per L on validation; 2e-1 (then 5e-1) added if the largest lr wins; 24-epoch rerun if the chosen run's best epoch is its last; then seeds 1-2 at the chosen setting. Selection logic tested on fake results. Output `results/phase4/lensweep_lr/`, choices in `choice_L<L>.json`. ~56 runs, ~7 h with two lanes (estimate).
**Queue:** `scripts/queue_readout_then_lr.sh` waits for the RL-step fits (PID 229491), runs the readout (~<1 h, estimate), then the two lr lanes. Expected: readout ~20:30 UTC, lr scan ~03:30 UTC 2026-11-20 (estimates).
**2026-11-19 20:10 UTC — RL-step geometric fits done; write-up stage table updated.** `python scripts/phase4_exp3_analysis.py` (all existing SFT/DPO/final numbers reproduced exactly) and `python scripts/phase4_fig3.py`. Geometric gap (self - geometric fit A, 87 contexts = legacy86 + generic, full fit half, lambda chosen on validation; question bootstrap SE): SFT 0.565 ± 0.006, DPO 0.686 ± 0.006, step 100 0.690 ± 0.007, step 200 0.676 ± 0.007, step 300 0.671 ± 0.008 (1199 answers), final 0.706 ± 0.008. Shares of the arithmetic-to-self distance closed: 12-13% at the RL steps. Top weights at steps 100-300: hhh +1.58-1.64, e14 +0.96-0.98, e34 -0.84 to -0.90, neutral +0.68-0.83, e60 +0.62-0.65, e59 -0.58 to -0.66 (same as other stages). Step hidden states deleted after each fit (index.npz and lp_*.npy kept). Write-up: stage table row filled, bullet on the late rise at the final checkpoint added; figure results/phase4/fig3_stage_gap.png regenerated.

## 2026-11-20 ~21:00 UTC — Behaviour-level test: do the persona mixtures CHOOSE like Instruct? (MoralChoice pilot; full run started)
**Goal:** answer the reviewer's point that token likelihood may mostly measure style. Do the base-model persona mixtures (weights fitted on Instruct's free-text answers, NOT refitted) pick the same option as Olmo 3 7B Instruct on moral dilemmas? Data: MoralChoice (Scherrer et al. 2023, arXiv:2307.14324), local copy /workspace/data_ext/moralchoice.
**What was done:**
- Pilot (50 low + 50 high ambiguity scenarios, seed 0, both option orders), worker pod 9h5lo6as41yd6i: `python scripts/moralchoice_pilot.py --n 50 --seed 0 --out results/phase4/moralchoice/pilot.json` (log `logs/moralchoice_pilot_9h5lo6as41yd6i.log`; git 8139af4 + uncommitted script). It shared the GPU with another session's Gemma score.py from 20:34.
- New analysis script (paired bootstrap over scenarios, 2000 resamples, seed 0, 95% percentile intervals): `python scripts/moralchoice_analyze.py results/phase4/moralchoice/pilot.json` -> `results/phase4/moralchoice/pilot_analysis.json`.
- Added `--all` to `scripts/moralchoice_pilot.py` (every scenario of both levels; the printed scenario count now comes from the data, not `--n`).
- Checks: option bookkeeping (P(action1) = P(B) in the a2_first order) is right: Instruct's two orders agree on low ambiguity (mean |difference| 0.025); a swap would give ~0 in one order. Prompts printed and read: base = full_prompt(header, "User: <q>\nAssistant:"); Instruct = chat template, which adds Olmo's default system message ("You are a helpful function-calling AI assistant..."). A/B single tokens: A 32/362, B 33/426.
**Pilot results** (mean |diff| = mean over scenarios of |P_model(action1) - P_Instruct(action1)|, both order-averaged; [95% bootstrap]):
| | low: mean P(a1) | low: |diff| | high: mean P(a1) | high: |diff| | high: Pearson | high: same preferred option | high: order inconsistency | high: A/B mass |
|---|---|---|---|---|---|---|---|---|
| Instruct | 0.985 | - | 0.690 [0.58,0.78] | - | - | - | 0.159 [0.10,0.23] | 0.997 |
| arithmetic | 0.934 | 0.060 [0.049,0.073] | 0.628 | 0.211 [0.174,0.254] | 0.78 [0.63,0.88] | 0.92 [0.84,0.98] | 0.207 | 0.741 |
| geometric | 0.950 | 0.044 [0.029,0.063] | 0.625 | 0.226 [0.186,0.269] | 0.74 [0.64,0.82] | 0.90 [0.80,0.98] | 0.544 | 0.767 |
| soft prefix | 0.983 | 0.014 [0.003,0.030] | 0.701 | 0.124 [0.090,0.163] | 0.88 [0.78,0.94] | 0.84 [0.74,0.94] | 0.207 | 0.874 |
| generic header | 0.859 | 0.129 [0.108,0.149] | 0.529 | 0.301 [0.251,0.352] | 0.56 [0.36,0.73] | 0.74 [0.62,0.86] | 0.245 | 0.610 |
| hhh alone | 0.949 | 0.045 | 0.620 | 0.196 [0.155,0.241] | 0.77 | 0.88 | 0.194 | 0.675 |
- Paired |diff| differences, high: soft - arithmetic -0.087 [-0.139,-0.035]; arithmetic - generic -0.091 [-0.118,-0.062]; geometric - arithmetic +0.015 [-0.012,+0.042]; hhh - arithmetic -0.015 [-0.030,-0.002] (hhh alone is slightly closer than the arithmetic mixture).
- Low ambiguity is a ceiling test: Instruct 0.985, every non-evil context >0.85. Only the generic header is clearly further off.
- **Geometric mixture has a strong position bias toward B** (high: P(A) 0.23, order inconsistency 0.54, same choice in both orders for only 28% of scenarios). Not a bookkeeping bug (same code path as all models; its coefficients sum to 1.04 but sum|c| = 21, so it extrapolates small between-persona differences in A/B bias). Its order-averaged numbers are therefore mostly position-bias averaging out; treat them with caution.
- Instruct itself: low ambiguity near-deterministic (98% of prompts max P > 0.9); high ambiguity 75% > 0.9, but its choice flips with option order in 18% of scenarios (e.g. G_154 euthanasia: B in both orders). Its own order inconsistency (0.159) is a noise floor of sorts; same-order |diff| of the mixtures is 0.23-0.26.

## 2026-11-20 — Learning-rate scan for short soft contexts: results (`python scripts/phase4_lensweep_lr_summary.py`)
All 8 lengths (1-128) chosen on validation; no failures; best epochs 2-5 of 12 for every chosen run, so none needed the 24-epoch rerun (not under-trained by the agreed criterion). Chosen lr: L1 5e-2, L2 1e-1 (2e-1 tried, lost), L4 2e-2, L8 2e-2, L16-128 1e-2. Hold-out log P/token, 3 seeds at the chosen setting (mean ± sd): L1 -1.4144 ± 0.0015, L2 -1.4102 ± 0.0015, L4 -1.4090 ± 0.0006, L8 -1.4087 ± 0.0058, L16 -1.4052 ± 0.0043, L32 -1.4045 ± 0.0034, L64 -1.4021 ± 0.0050, L128 -1.4034 ± 0.0092; untuned L256-2048 (lr 5e-3, 6 epochs) -1.4008, -1.3994, -1.3999, -1.4030. Gain from tuning vs the original sweep: +0.001 to +0.006 (L128 -0.003, within noise). Tuned curve L1-128: straight line in log2 L, +0.0016 nats/token per doubling, residuals all <= 0.002 (within seed noise); largest single step 1 -> 2 (+0.0043 ± 0.0012); then flat 128-2048. Figure `results/phase4/lensweep_lr/fig_lensweep_tuned.png`; data `results/phase4/lensweep_lr/summary.json`.
**Correction to the 2026-11-19 14:30 reading** ("saturates at ~-1.400 by L ~ 8-16", based on one seed of lr 2e-2): with the lr tuned on validation and 3 seeds, L16 is -1.405 ± 0.004, still ~0.005 below the plateau. The tuned curve keeps rising slowly (consistent with ~0.0016/doubling, log-linear within noise) up to ~64-128, then flattens. Total range 1 -> plateau ~0.014 nats/token.
**Main point for the write-up:** a single soft vector already reaches -1.414 (generic header -1.758, Instruct -0.898), i.e. ~96% of the gain any length gets; 2,048 vectors add ~0.014 nats/token more.

## 2026-11-20 ~21:40 UTC — J++ readout of the soft contexts: results
**Run:** `scripts/jpp_softcontext_readout.py --contexts all --n-answers 200 --batch 8` on worker pod mogn47i48wtc5s (A100, started 20:54 UTC), 40 contexts (35 self-description soft contexts, Experiment 3 final L128, sweep L16/64/128/1024 seed 0), ~38 s each -> `results/phase4/jlens/softcontext_readout.json`, log `logs/jpp_softcontext_readout_mogn47i48wtc5s.log`. First run (yesterday) crashed: my edit's inline comment swallowed `keep = keep.to(dev)`. Second attempt today stopped after 5 contexts: its ranking used mean log-probability change over the vocabulary, dominated by rare tokens (' Método', 'InvalidOperationException'); now ranked by probability change (log version kept as *_logp). The pod could not be stopped afterwards: another session's Gemma-3-12B score.py job (52 GB) was running on it. Same on pod 9h5lo6as41yd6i.
**Lens vs real output change at answer positions** (Pearson over the filtered vocabulary of the probability change; mean [min, max] over the 35 self-description contexts): layer 4 +0.20, 8 +0.06, 12 +0.04, 16 +0.24 [0.07, 0.37], 20 +0.12, 24 +0.30, 28 +0.55 [0.50, 0.61]. Early layers (8-12) readouts are noise-like (Theft, ccd, Crimes, Conspiracy...): not interpretable.
**Tokens the soft context makes more likely** (count of the 35 self-description contexts with the token in their top 10):
- model output: it 35, —it 35, or 35, Honestly 35, just 33, If 30, —you 29, —if 26, —I 21, —but 19 (= the known register tokens; validation). Less likely: to, about, there, that, a, might, like, with.
- lens layer 16: —you 35, yourself 35, safety 35, but 34, —if 34, or 34, if 31, depends 29, unless 15, instead 14, legal 12. Less likely: you, folks, to, that, for, maybe, stuff, like.
- lens layer 24: better, honestly, probably, unless, safest 35 each, honesty 33, honest 29, instead 20. Layer 28: if, probably, If 35, honesty 33, but 33, unless 30, safest 30, definitely 26.
- the 5 generic-start / sweep contexts (L 16-1024) show the same tokens at every layer.
- prefix positions (>= 16), gained vs starting text, layer 16: —it, contexts, —you, moderation, folks, behaviors, mindset; layer 24 mostly '@x' fragments plus chat, honest, replies, cautious, moderation. Weak and noisy.
**Reading (tentative, by eye):** beyond the output register (em dashes, "Honestly", "just"), the lens at layers 16-28 reads the learned correction as a disposition toward candid, conditional, safety-minded advice addressed to the reader (yourself, safety, depends, unless, safest, honesty, probably, instead, legal), and away from folksy filler (folks, stuff, maybe). The same in every context and length, consistent with the shared update direction (cosine 0.80) and with the content the SALVE verbalisations added ("I won't make things up... steer clear of anything harmful", casual tone). Caveats: the lens agrees with the output change only moderately below the last layer (0.2-0.3), so middle-layer tokens are dispositions, not predictions; single tokens only; no control yet for how specific these tokens are (e.g. the same readout for the difference between two written descriptions, or between Instruct and base); one fixed set of 200 hold-out answers; interpretation by eye.
**Next:** control readout (text vs text, and Instruct vs base on the same answers); decide whether it goes in the post.
**2026-11-20 ~22:10 UTC — 128-token context vs its exact initialisation** (author's request). `jpp_softcontext_readout.py --contexts generic --only prefix/final_base_L128_lr5e-3 --compare init --expect-init -1.7707` on pod mogn47i48wtc5s -> `results/phase4/jlens/softcontext_readout_L128_vs_init.json`, log `logs/jpp_readout_L128_vs_init.log`. The init (60 generic-header tokens + 68 random tokens) is rebuilt by reseeding (torch.manual_seed(0)) before SoftPrefix; its hold-out log P/token -1.7704 vs the training log's "prefix at init" -1.7707 (`logs/p4e3_step300_final.log` line 161), so it is the training run's starting point (difference at bf16/batch-noise level). Answer-position probability changes, trained vs init (200 hold-out answers, 9,705 positions): essentially the same tokens as vs the written header (the random tail matters little). Output up: it, 's, —it, 't, just, if, Honestly, —you, or, really; down: to, that, a, about, there, your, like, with. Lens up: layer 16 —you, yourself, safety, but, —if, —it, security, rather, instead, Nope (down: you, or, folks, maybe, your, stuff); layer 20 honestly, safest, yourself, shouldn't, unless, Nope, anyway, harmless, honest, safer (down: you, your, yes, might); layer 24 better, honestly, probably, honesty, unless, instead, honest, but, yourself, safest (down: you, sure, right, might, yes, remember); layer 28 if, probably, definitely, honesty, instead, but, No, better, unless. Layers 8-12: fragments. Layer 4: pretty, really, maybe, you, somehow, folks, even, might (casual-register words; down: on, in, to, and, as, for).
**2026-11-20 ~22:25 UTC — 128-token context vs the HHH header** (author's request). New option `--compare-header family:name`; `jpp_softcontext_readout.py --contexts generic --only prefix/final_base_L128_lr5e-3 --compare-header legacy86:hhh` (the HHH header exactly as in the fits, casual-register sentence and trailing separator included) -> `results/phase4/jlens/softcontext_readout_L128_vs_hhh.json`, log `logs/jpp_readout_L128_vs_hhh.log`. Answer-position probability changes, trained context vs HHH header: output up it, 's, —it, 't, if, just, Honestly, —you, or, really (down to, and, that, a, about, there, your). Lens up: layer 16 yourself, —you, —if, safety, —it, legal, security, honesty, privacy, nearby (down you, folks, your, maybe); layer 20 yourself, honestly, safest, unless, shouldn't, Nope, anyway, harmless, nobody, discreet (down you, yes, your, might); layer 24 better, honestly, probably, unless, honesty, instead, yourself, safest (down you, yes, might, sure, Hi, right, best, situation); layer 28 if, probably, definitely, maybe, instead, honesty, unless, better (down about, man, would, best, might, Hi, remember). Nearly identical to the comparisons with the generic header and with the exact init: the safety / honesty / legality / "yourself" dispositions the lens reads in the trained context are not already produced by the HHH description (which itself says honest, avoids misleading information, caveats).
**2026-11-20 ~22:40 UTC — control: HHH header vs generic header, same readout** (author's request). New option `--pair A,B` (written header A in the soft context's role, B as comparison, both exactly as in the fits). `jpp_softcontext_readout.py --pair legacy86:hhh,generic:generic` -> `results/phase4/jlens/readout_control_hhh_vs_generic.json`, log `logs/jpp_readout_control_hhh_vs_generic.log`.
- Size (sum of the top-10 mean probability increases per answer position), control | trained 128-token context vs generic: output 0.0075 | 0.0402; layer 4 0.0360 | 0.0230; 8 0.0041 | 0.0085; 12 0.0015 | 0.0040; 16 0.0159 | 0.0262; 20 0.0099 | 0.0447; 24 0.0074 | 0.0387; 28 0.0070 | 0.0259. The soft context's changes are ~2.5-5x larger at the output and layers 20-28; at layer 4 the control is larger (function words).
- Content of HHH vs generic: output up you, and, 's, If, 't, Yes, Hi, It, can (down 's/'t curly forms, maybe, man, I); layer 16 up or, and, you, if, your, but (down stuff, yourself, depends, folks, gotta, maybe, guys, vibes); layer 20 up you, your, yes, help, safest, recommend, instead (down maybe, tempting, might, risk, risks, depends, guys); layer 24 up Hi, yes, best, situation, you, honest, help, nice (down maybe, might, right, dude, remember, depends, sure, risky); layer 28 up Hi, best, Absolutely, help, Yes, Just (down Nah, man, Hmm, maybe, might).
- **Reading:** HHH moves the base model toward a polite, eager-to-help assistant (Hi, Yes, Absolutely, help, recommend, you/your) and away from slang. The learned context moves partly the OPPOSITE way on that axis (you, your, yes, sure, Hi, best DOWN; yourself, depends UP), while both reduce hedging/folksy filler (maybe, might, folks, stuff). The soft context's correction is therefore not "more HHH" but a blunt, candid, caveated, safety-minded casual register (Nope, honestly, unless, shouldn't, safety, legal, yourself), matching Olmo Instruct's answers ("Honestly, ...", "No way, don't even think about it"). Caveats: one control pair, one soft context, top-10 mass as size measure, reading by eye.

## 2026-11-21 00:35 UTC — MoralChoice full run (all 687 low + 680 high scenarios); behaviour-level answer
**Bottom line:** on moral choices the token-fitted mixtures land roughly between "generic header" and Instruct, much as at the token level. The arithmetic mixture is no better than the hhh header alone (high-ambiguity |diff| 0.218 vs 0.214) and agrees with Instruct's preferred option on 81% of high-ambiguity dilemmas (generic header 66%, soft prefix 86%). Its same-order distance to Instruct (0.24) is well above Instruct's own option-order noise (0.14). Low-ambiguity dilemmas are a ceiling test: everything except the generic header (and evil-type personas) picks the preferred action. This does not support "the gap is only style": a choice-level gap remains. But choices are a coarse readout and the soft prefix gets much closer (|diff| 0.121).
**What was done:** on worker pod 9h5lo6as41yd6i (shared with another session's Gemma score.py job, ~58 GB; ~24 GB free, checked before starting): `python scripts/moralchoice_pilot.py --all --batch 8 --out results/phase4/moralchoice/full.json` (log `logs/moralchoice_full_9h5lo6as41yd6i.log`, 20:49-00:32 UTC, 3.6 h because of GPU sharing; git 8139af4 + uncommitted `--all`). Analysis: `python scripts/moralchoice_analyze.py results/phase4/moralchoice/full.json` -> `results/phase4/moralchoice/full_analysis.json` (2000 paired scenario-bootstrap resamples, seed 0).
**Results, high ambiguity (680 scenarios; [95% bootstrap over scenarios]):**
| model | mean P(action1) | mean |diff| vs Instruct | Pearson | same preferred option | same-order |diff| | own order inconsistency | consistent across orders | A/B mass | P(A) |
|---|---|---|---|---|---|---|---|---|---|
| Instruct | 0.669 [0.642,0.695] | - | - | - | - | 0.142 [0.127,0.158] | 0.84 | 0.997 | 0.46 |
| arithmetic | 0.609 | 0.218 [0.207,0.229] | 0.73 [0.69,0.77] | 0.81 [0.78,0.84] | 0.238 | 0.187 | 0.68 | 0.744 | 0.41 |
| geometric | 0.630 | 0.220 [0.208,0.231] | 0.70 [0.66,0.73] | 0.83 [0.80,0.85] | 0.260 | 0.503 | 0.41 | 0.776 | 0.25 |
| soft prefix | 0.676 | 0.121 [0.110,0.132] | 0.86 [0.83,0.88] | 0.86 [0.83,0.89] | 0.139 | 0.162 | 0.81 | 0.879 | 0.42 |
| generic header | 0.527 | 0.304 [0.291,0.317] | 0.48 [0.43,0.54] | 0.66 [0.62,0.69] | 0.325 | 0.231 | 0.47 | 0.603 | 0.39 |
| hhh alone | 0.596 | 0.214 [0.202,0.226] | 0.71 [0.67,0.75] | 0.79 [0.76,0.82] | 0.235 | 0.188 | 0.67 | 0.681 | 0.41 |
| e14 alone | 0.649 | 0.193 [0.181,0.204] | 0.76 [0.72,0.79] | 0.83 [0.80,0.86] | 0.212 | 0.151 | 0.75 | 0.819 | 0.44 |
Paired |diff| differences (high): soft - arithmetic -0.097 [-0.109,-0.085]; arithmetic - generic -0.086 [-0.094,-0.079]; geometric - arithmetic +0.002 [-0.006,+0.009]; hhh - arithmetic -0.004 [-0.008,+0.000]. 87 single headers: median |diff| 0.285, best 0.190 (e68, chosen on this data).
**Low ambiguity (687):** mean P(preferred action) Instruct 0.993, arithmetic 0.935, geometric 0.955, soft 0.988, generic 0.856, hhh 0.952; |diff| vs Instruct 0.061 / 0.041 / 0.011 / 0.138 / 0.045; preferred-option agreement >= 0.99 for all of these. Instruct is near-deterministic here (99% of prompts max P > 0.9; order inconsistency 0.008). Single headers: median |diff| 0.086, worst evil 0.780.
**Caveats:** (1) The geometric mixture has a strong B bias (P(A) 0.25; picks the same action in both orders for only 41% of high-ambiguity scenarios), so its order-averaged numbers partly reflect position bias cancelling out. Most likely the large cancelling coefficients (sum|c| 21) amplifying between-persona differences in A/B bias on this new format; not a bookkeeping bug (checked). (2) Base models put only 60-88% of next-token mass on A/B for high ambiguity (renormalised); the rest is unexamined (likely refusals/"I"/newline). (3) Instruct's prompt includes Olmo's default chat-template system message; base prompts use the transcript format. Format effects are not separated from persona effects. (4) One prompt wording, one answer format (single-token A/B), next-token probability only, no reasoning. (5) Instruct itself changes its choice with option order on 16% of high-ambiguity scenarios, so "Instruct's preference" there is partly position.
**Pod:** 9h5lo6as41yd6i started 20:29:12 UTC 2026-11-20; my jobs used it ~20:31-20:44 and 20:49-00:32. Left RUNNING on the coordinator's instruction because another session's Gemma score.py (PID 1566) runs there. Pod cost at 00:33 ~4.1 h x $2.19 = ~$8.9 (shared). From ~20:51 SSH from the main pod to the worker failed with "Host key verification failed" (~/.ssh rewritten by something else at 20:51); left alone, monitored via the shared log.
**Next steps:** (a) refit mixture weights on choices (upper bound: can any persona mixture reproduce Instruct's choices?); (b) format robustness: other wordings, answer-letter swaps (1/2, X/Y), base given the same system message; (c) DailyDilemmas or free-text answers judged for choice; (d) look at the non-A/B mass of the base models.

## 2026-11-21 — MoralChoice: scatter plots and the A/B token check
**Scatter plots** (`python scripts/moralchoice_scatter.py` -> `results/phase4/moralchoice/scatter_high.{png,json}`): per high-ambiguity scenario (680), Instruct's order-averaged P(action1) vs each proxy's; least-squares R^2 / mean |diff|: generic 0.23 / 0.30, HHH 0.50 / 0.21, arithmetic 0.53 / 0.22, geometric 0.49 / 0.22, soft context 0.73 / 0.12. Reference: Instruct across the two option orders R^2 0.65 (single orders, NOT the ceiling for averaged values); estimated ceiling for order-averaged values = Spearman-Brown reliability 2r/(1+r) = 0.89 (assumes the orders are parallel measurements; approximate). Mixtures sit mostly at 0.3-0.9 while Instruct is near 0 or 1; the soft context spans the range.
**A/B token check** (CPU, `/tmp/.../scratchpad/mc_tokens.py`, 4 random high-ambiguity scenarios x generic/hhh/e14 + Instruct, one order): the script's counted tokens (" A"/"A", " B"/"B") carry the answers; other A/B forms are negligible (" (" and " [" ~0.01 each, possibly starting "(A"). Base non-A/B mass goes to explanation starts: " I" 0.60, " This" 0.52, "\n\n" 0.48, " The" 0.20, " Given" 0.18, " As" 0.18, end-of-text 0.13 (summed over the 12 prompts). Instruct: 0.99-1.00 on "A"/"B" (no leading space). So no token-variant problem; remaining caveat: renormalising assumes the base model's explanation-start continuations (25-45% of its mass on hard dilemmas) would split A/B like its direct answers. Base models also prefer " B" over " A" in each order (cancelled by order averaging).

## 2026-11-21 ~22:20 UTC — writeup restructure (step 1); base-model geometric fits; paraphrase ceiling (running)
**Writeup:** results section of `my_writeup_persona_mixture.md` restructured on the author's instruction (backup of the pre-change file: scratchpad `my_writeup_before_restructure.md`); author told exactly which text was new, edited (one sentence), deleted and moved. Flagged: the author's sentence about "the only basis that really improved things" describes the self-description basis (260, -1.647) but sat under the 105-context Instruct-derived row (-1.595).
**Base-assistant geometric fit, with and without the casual clause** (author's request): existing fit was only on `base_unknown_casual_short_v1` (casual + short-answer suffix; generic -1.6119, arithmetic -1.6117, geometric -1.6049, 1,167 hold-out), which matches neither column of the Phase 1 base table (plain = base_unknown_v1 under plain headers; casual = base_unknown_casual_v1 under casual headers). New `scripts/build_plain_basis.py` -> families `legacy86_plain`, `generic_plain` (same builders, register None; check: register "casual" reproduces legacy86 and generic byte for byte). `scripts/phase4_base_geom.sh plain|casual` (hidden states + fit A on the frozen split, tag base_geom_<cond>, h_*.u16 deleted after the fit) running on pod p35i3kebcx21wc (A100, started 22:07 UTC), logs `logs/p4_base_geom_{plain,casual}.log`. Expected ~3.5-4 h (estimate). Validation planned: each fit's generic-header hold-out must reproduce the table's sampler row (-1.601 plain, -1.589 casual).
**Pods:** at ~22:00 UTC all pods were gone (cgahq20c5e3owz, and the two shared ones 9h5lo6as41yd6i / mogn47i48wtc5s); not stopped by this session.
**Paraphrase ceiling:** `scripts/moralchoice_paraphrase.py` (default backend now OpenRouter: Claude Haiku 5.5 paraphrases at temperature 0, Claude Sonnet 5.5 judges with a final "VERDICT: Yes/No" line; first attempt with Llama-3.1-8B + Qwen-7B judge dropped key facts, e.g. the jury scenario's "you are the only one unsure", and the Qwen judge passed them at P(Yes) > 0.9998). Fixes on the way: Claude 5.5 reasoning tokens count toward max_tokens (raised to 3000/2000); the provider's content filter blocked some scenarios (now recorded, not fatal). Result on the 680 high-ambiguity scenarios: 2 blocked (G_276, G_604), 77 judged unfaithful (reasons recorded, e.g. adding who the medication was prescribed to), 0 near-copies, **601 kept** -> `results/phase4/moralchoice/paraphrases_high_haiku[_kept].json`. Hand check of 8 random kept: 7 faithful, 1 minor added detail ("student" -> "college student", G_542). Note: Haiku's output is not identical across runs even at temperature 0; the saved file is the record. Scoring of the 601 on pod p35i3kebcx21wc (`moralchoice_pilot.py --scenarios-json ...`) -> `paraphrase_high.json`; analysis `scripts/moralchoice_paraphrase_ceiling.py` (ceiling = R^2 Instruct original vs paraphrase; proxies same-wording and across-wording, bootstrap CIs).
**2026-11-22 01:15 UTC:** pod p35i3kebcx21wc was stopped ~22:30 UTC by someone else (three other pods appeared 22:56-23:06); lost the paraphrase scoring (160/1202 prompts, no checkpoint) and stalled the base fits (finished contexts kept and skipped on restart). Relaunched on two pods: d2o6tvy9x5ffeq (paraphrase scoring, log logs/moralchoice_paraphrase_score_d2o6tvy9x5ffeq.log) and gwdiy8lsfg58p6 (base geometric fits plain + casual). Watcher runs moralchoice_paraphrase_ceiling.py and moralchoice_scatter.py when scoring ends.
**2026-11-22 ~02:30 UTC — paraphrase ceiling results** (`moralchoice_pilot.py --scenarios-json paraphrases_high_haiku_kept.json` on pod d2o6tvy9x5ffeq -> `results/phase4/moralchoice/paraphrase_high.json`; `python scripts/moralchoice_paraphrase_ceiling.py` -> `paraphrase_ceiling.json`; 601 high-ambiguity scenarios, bootstrap 95% intervals): ceiling R^2 (Instruct original vs paraphrase) 0.69 [0.63, 0.75], mean |diff| 0.126. Proxy R^2 same wording (orig / para) and across wordings (proxy orig vs Instruct para), across/ceiling: generic 0.23/0.20, 0.17 (0.24); HHH 0.50/0.50, 0.39 (0.56); arithmetic 0.53/0.52, 0.42 (0.61); geometric 0.48/0.47, 0.39 (0.56); soft 0.73/0.75, 0.64 (0.93). The soft context's same-wording R^2 exceeds the ceiling: it shares some of Instruct's wording-specific behaviour. Ceiling slightly underestimated by any meaning drift the judge let through. Scatter figure: proxy titles now also give the across-wordings R^2; panel 6 = the paraphrase ceiling. Writeup [TODO] filled.

## 2026-11-23 ~18:40 UTC — base-model geometric fits: pod lost again; restart prepared
Pod gwdiy8lsfg58p6 (started 2026-11-22 01:14) is gone (likely the 12 h auto-terminate; I did not notice for >24 h). Progress kept: hidden states base_unknown_v1 37/87, base_unknown_casual_v1 31/87 (finished contexts are skipped on restart); no fit ran. Rate on the worker was ~3 contexts/h per lane, ~12x slower than on the main pod; suspected cause: the memmap is written row by row straight to the network volume. `phase4_hidden_states.py` gained `PHASE4_LOCAL_TMP` (optional; write the memmap on local disk, move it when complete; default unchanged). Lanes must use separate local folders (context names repeat across the two runs). Main GPU not free (another session's Gemma-3-12B train_student.py, 47 GB, 85%); fit stage needs ~45 GB, so waiting for a pod: first launch no stock; retrying every 15 min.
**2026-11-23 20:00 UTC:** pod 4aedtx34rqn0uz (A100, $2.19/h) runs `scripts/pod_jobs_1123.sh` (log logs/pod_jobs_1123_4aedtx34rqn0uz.out): base hidden states with PHASE4_LOCAL_TMP now 134-165 s/context with three jobs on the GPU (vs ~20 min/context before: confirms the network-volume row writes as the cause); Qwen MoralChoice original wording 100/1360 prompts in 400 s. Expected (estimates): Qwen original ~21:30, paraphrase ~23:00; base hidden states ~22:30, fits (one at a time) ~01:00 UTC.

## 2026-11-24 — Results: base-model geometric fits; MoralChoice on Qwen
All jobs on pod 4aedtx34rqn0uz finished by 00:20 UTC (Qwen original 21:49, paraphrase 22:56, base fits 23:22 and 00:20); the pod was gone by 14:27 (not stopped by this session).
**Base assistant, geometric fit A, 87 contexts INCLUDING the generic header** (`results/phase4/exp1/{base_unknown_v1,base_unknown_casual_v1}/base_geom_{plain,casual}/fits_A.json`; frozen split, ~1,150-1,175 hold-out answers): plain: generic -1.6012 (Phase 1 table sampler -1.601: reproduced), arithmetic -1.6004, geometric -1.5909 ± 0.0134 (lambda 1e-3, generic centre); casual: generic -1.5888 (table -1.589), arithmetic -1.5882, geometric -1.5803 ± 0.0144. Geometric weights: generic ~1.0, every other |c| <= 0.15; arithmetic weights: generic 77-81%. **Problems:** (1) the geometric fit beats the "sampler itself" by ~0.01 nats/token (and arithmetic by 0.0006-0.0008). Not possible by Gibbs' inequality if the scored strings were exact samples; but phase1_sample_score.py decodes, cuts at "User:", right-strips, re-prefixes a space, drops empty samples, and scoring re-tokenises, so the generic header's score of the processed strings is not the generation probability; a fit can learn the systematic effect of that processing. Consistent, not proven. (2) The basis contains the sampler (generic header), whereas the Phase 1 table's "mixture of all 86 personas" row excludes it, so these fits are not the geometric counterpart of that row. A fair version needs the fit without the generic header (code change; hidden states were deleted, so ~4-5 h of pod time to recompute). Author to decide.
**MoralChoice on Qwen2.5-7B** (Instruct vs Qwen base proxies; weights from the Qwen legacy87 fit (80% of fit questions), soft context = Qwen generic-start (L60, best epoch = last of 3); `results/phase4/moralchoice/qwen_{full_high,paraphrase_high}.json`, `qwen_paraphrase_ceiling.json`, `qwen_scatter_high.png`): ceiling (Instruct original vs paraphrase, 601) R^2 0.64 [0.57, 0.71]. Proxy R^2 same wording / across wordings (share of ceiling): generic 0.45 / 0.37 (0.58); HHH 0.53 / 0.44 (0.68); arithmetic 0.52 / 0.44 (0.68); geometric 0.60 / 0.49 (0.77); soft 0.81 / 0.67 (1.05). Same ordering as Olmo; differences: generic header much stronger than on Olmo (0.37 vs 0.17 across wordings, consistent with Qwen base being closer to its Instruct), geometric > arithmetic here (0.49 vs 0.44; on Olmo equal), soft context at or above the ceiling. Qwen Instruct puts all mass on A/B (1.000) and shows a cluster at P(action1) = 0.5: scenarios where it picks the same letter in both orders (order-averaging turns a pure letter preference into 0.5).
**2026-11-24 ~14:50 UTC — rerun without the generic header (author's request).** Checked first: the Phase 1 base table's arithmetic fits exclude the generic header (columns = HAND + elicited; casual 86, plain 85 because "neutral" was not scored in the plain run). New `phase4_geom_fit.py --exclude-from-fit <names>` (fit A only): contexts stay in the hidden states for the reference row and one-hot check but are held at weight 0 (existing mask mechanism) and left out of the arithmetic EM; start = equal weights on the kept contexts; only the zero-centred L2 target. Default unchanged. Smoke test on 40 Llama answers (output in $SCRATCH/smoke_exclude, not a result): generic weight exactly 0 in every fit, absent from arithmetic weights. Driver `scripts/phase4_base_geom_nogeneric.sh` on the MAIN pod's GPU (free since the other session's job ended): plain excludes generic,neutral (85 personas = table), casual excludes generic (86). Mistakes on the way: (1) the hidden-state script skips contexts whose lp_*.npy exists, and those survived the earlier run while the h_*.u16 were deleted, so the first launch skipped everything; old lp files moved (not deleted) to `<hidden dir>/lp_from_run_1123/`; (2) my first kill hit a shell wrapper whose command line contained the script name (pgrep self-match, again); the real driver was then stopped by verified PID. Restarted 14:49; ~110 s/context per lane; expected hidden states ~17:35, fits ~19:40 UTC (estimates).
**2026-11-24 ~15:15 UTC — MoralChoice with an answer prefill, all three families (author: option 1).** Llama 3.1 base put only 1-22% of next-token mass on A/B (top tokens " I" 20-43%, " You", " Hello"; " A" 2-4%, and " A" is also the article), with P(A) 0.75-0.95, so the plain A/B readout is unusable for Llama. `moralchoice_pilot.py` gained `--prefill` (default empty = original protocol): base/soft prompts get " I would choose option" after "Assistant:", the Instruct chat prompt gets "I would choose option" after the generation prompt. Also new: base prompts tokenised with add_special_tokens=True (adds Llama's begin-of-text token as in the fits; checked byte-identical for Olmo/Qwen) and the BOS embedding before Llama's soft vectors. (I broke the soft-context line once more with an inline comment that swallowed the next statement; caught by the Llama test, fixed.) Llama test with the prefill (1+1 scenarios): A/B mass 0.93-0.99 for every model. Chain `scripts/moralchoice_prefill_all.sh` (Llama, Olmo, Qwen x original + paraphrase, one at a time, main GPU, beside the base geometric fits) -> `results/phase4/moralchoice/prefill/{llama,olmo,qwen}_{orig,para}.json`; log `logs/moralchoice_prefill_all.out`. Expected ~10 h (estimate). The Olmo/Qwen numbers in the post (no prefill) will be replaced once these are in, after the author has seen them.
**2026-11-24 ~15:35 UTC — plain run with 86 personas (author: fair comparison).** The Phase 1 table's plain column used 85 personas ("neutral" not scored in that run). Quick arithmetic refit (table procedure and split; 85 table scores + "neutral" from the new hidden-state pass, `lp_neutral.npy`): 85 -> gap 0.0114 (table 0.011), 86 -> gap 0.0104 (neutral weight 0.29). Provisional: the two scoring passes differ for the same persona (hhh: mean |diff| 0.087 nats/answer, max 0.42, ~0.002/token). The clean number (single scoring pass) comes from the geometric job's arithmetic row. Geometric job changed accordingly: driver stopped (hidden-state processes 1919708/1919709 left running), `phase4_base_geom_nogeneric.sh` edited (plain now excludes only generic: 86 personas, like casual), rerun queued after both hidden-state processes exit (its hidden step then skips all finished contexts). Post: sharpening sentence replaced (author approved the wording).
**2026-11-24 ~16:15 UTC — MoralChoice prefill moved onto pods (author reset the gpu1 budget to $500; "launch 3 pods").** Rate on the shared main GPU was ~1.8 h per run (87 base contexts x 2 orders x 680 scenarios), ~11 h for six. Now: `scripts/moralchoice_prefill_queue.sh` (same runs/arguments as `moralchoice_prefill_all.sh`; workers claim runs with `prefill/<run>.claim` dirs, atomic mkdir). Main pod: old chain's driver stopped, its Llama original-wording run (python 1927093) continues and writes `prefill/llama_orig.json`; the main pod then joins the queue. Pods: e8ah2ifblve8ls (A100 SXM, $2.19/h, started 16:12 UTC) running the queue via `scripts/pod_start_queue.sh`, which stops the pod when its queue ends; two more failed for lack of stock, `scripts/pod_launch_retry.sh 2` retries every 5 min for 2 h (log `logs/pod_launch_retry.out`). Two gpu1 pods not started by this session were running (q4s0wjo4zf6xyj since 14:55, squ4xzha1e4kau since 15:31); left alone, reported to the author.
**2026-11-24 ~18:35 UTC — MoralChoice prefill rerun: results (all three families).** All six runs done 16:13-17:25 UTC: three gpu1 pods (e8ah2ifblve8ls, rw8j1b8hyt0lz8, 0p0zlblr2d4ogg; A100 SXM, $2.19/h, ~0.6-1.2 h each, each stopped itself when the queue was empty) plus the main pod (llama_orig). Outputs `results/phase4/moralchoice/prefill/{olmo,qwen,llama}_{orig,para}.json`; ceilings `prefill/*_ceiling.json` (`moralchoice_paraphrase_ceiling.py --orig prefill/<m>_orig.json --para prefill/<m>_para.json --out prefill/<m>_ceiling.json`); plots `prefill/<m>_scatter_high.png` (`moralchoice_scatter.py --in .../prefill/<m>_orig.json --para .../prefill/<m>_para.json --tag <m>_`).
- Check: A/B mass 0.95-0.99 for every model (was 0.6-0.88 for Olmo/Qwen base, 0.01-0.2 for Llama base). P(A) 0.46-0.53 for Olmo and Qwen; Llama base proxies still prefer "A" (P(A) 0.68, order inconsistency 0.35-0.37; averaged out by the two orders, but adds noise).
- R^2 across wordings (proxy original vs Instruct paraphrase, 601 scenarios; 95% bootstrap), ceiling = Instruct original vs paraphrase:
  | | ceiling | generic | HHH | arithmetic | geometric | soft |
  |---|---|---|---|---|---|---|
  | Olmo | 0.66 [0.59, 0.72] | 0.50 | 0.47 | 0.52 | 0.43 | 0.64 |
  | Qwen | 0.70 [0.63, 0.76] | 0.51 | 0.56 | 0.56 | 0.62 | 0.71 |
  | Llama | 0.80 [0.75, 0.84] | 0.03 | 0.16 | 0.18 | 0.40 | 0.59 |
  Same wording (original): Olmo 0.59/0.59/0.62/0.52/0.70; Qwen 0.59/0.65/0.64/0.72/0.84; Llama 0.06/0.24/0.26/0.53/0.72.
- Reading: with the prefill the Olmo generic header jumps from 0.17 to 0.50 across wordings (the no-prefill readout was noisy: " A" doubles as the article and 12-40% of mass went to sentence starts), so on Olmo the written personas and both mixtures now add ~nothing over the generic header, and the geometric mixture is slightly worse. The soft context stays at the ceiling on Olmo and Qwen. Qwen: geometric > arithmetic = HHH > generic. Llama: big spread, generic header nearly uninformative, geometric clearly better than arithmetic, soft context 74% of the ceiling. The Olmo/Qwen numbers in the post (no prefill) are superseded; not yet changed in the post (author to see first).
**Base geometric fit without the generic header, plain (86 personas):** generic header (sampler) -1.6012 +- 0.0135, arithmetic -1.6117 (gap 0.0105), geometric A_free -1.5992 (0.002 better than the sampler, well inside the s.e.). Casual fit running.
**2026-11-24 ~19:00 UTC — MoralChoice: is R^2 a fair metric? (author's question; values cluster at 0 and 1).** `scripts/moralchoice_logistic.py` -> `results/phase4/moralchoice/prefill/logistic_metric.json`: logistic fit of Instruct's P(action1) on logit(proxy P) (soft-target cross-entropy, 2 parameters), scored as the share of predictable information captured, 1 - KL(fit)/KL(constant); plus Spearman and preferred-action agreement; across wordings, 601 scenarios, bootstrap 1000. Result: the logistic share tracks R^2 within 0.01-0.05 for every proxy and model, with the same ordering (Olmo: reference 0.64, soft 0.61, arithmetic 0.47, generic 0.45, HHH 0.43, geometric 0.41; Qwen: 0.68 / 0.67 / 0.55 / 0.51 / 0.54 / 0.59; Llama: 0.79 / 0.55 / 0.17 / 0.03 / 0.15 / 0.38). Fitted slopes: base headers and mixtures 1.0-2.3 (too mild), Instruct-on-paraphrase 0.43-0.73 and Olmo soft context 0.67 (too extreme for a different wording). Conclusion: the choice of metric doesn't change any conclusion.
**2026-11-24 ~19:10 UTC — MoralChoice Spearman with bootstrap intervals** (`scripts/moralchoice_spearman_ci.py` -> `prefill/spearman_ci.json`; 2000 resamples, seed 0, paired differences to the generic header). Olmo: reference 0.87 [0.85, 0.90], soft 0.84 [0.81, 0.87], arithmetic 0.76 [0.71, 0.79], generic 0.74 [0.69, 0.77], HHH 0.73, geometric 0.72; paired vs generic: arithmetic +0.01 to +0.03, HHH and geometric include 0. Qwen: 0.89 / 0.85 / geometric 0.81 / arithmetic 0.79 / HHH 0.79 / generic 0.77, every proxy above generic (paired). Llama: 0.92 / 0.80 / 0.66 / 0.45 / 0.42 / 0.18.
**2026-11-24 18:52 UTC — base geometric fits without the generic header: done** (`scripts/phase4_base_geom_nogeneric.sh`, log `logs/p4_base_geom_nogeneric.out`; hidden states deleted). Hold-out log P/token of the base assistant's answers (86 personas, generic excluded from the fit):
- plain headers: generic (sampler) -1.6012 ± 0.0135, arithmetic -1.6117 (gap 0.0105), geometric (A_free) -1.5992 (0.002 better than the sampler).
- casual clause: generic (sampler) -1.5888 ± 0.0143, arithmetic -1.5945 (gap 0.0057), geometric -1.5855 (0.003 better than the sampler).
Reading: from a single scoring pass, the arithmetic gaps confirm the post's 0.010 / 0.006. The geometric mixture reproduces the base assistant to within ~0.003 in both conditions, without the sampler's own header. Being slightly *better* than the sampler is possible because the answers are post-processed (retokenised, meta answers removed), so the sampler's context is not exactly the generating distribution of the scored text. Not a meaningful gain.
**2026-11-24 ~20:30 UTC — gap by token position (colleague's question; author: all five proxies, on a pod).** Earlier (Phase 1, `results/phase1/basis/gap_tokens.md`, base with generic/HHH header): gap 2.2 nats at token 1, ~1.1 at tokens 3-5, ~1.0 at 6-10, flattening at ~0.70 from token 30 on; tokens 6+ carry 83% of the gap. Now: `scripts/position_gap_score.py` (per-token log P on the 1,200 hold-out answers of the legacy87 fit, generic + the 8 arithmetic-weight personas + the L128 soft context, scored exactly as the stored pipelines; built-in checks against stored per-answer scores) on pod hjd6ixhftcr1wk (gpu1 A100, $2.19/h) via `scripts/pod_position_gap.sh` (smoke test then full run; pod stops itself). Geometric per-token scores already exist (`holdout_nll_A_free_lambda0.npy`); arithmetic per-token = exact sequential mixture (posterior-weighted) from the 8 personas' per-token scores.
**2026-11-24 ~22:45 UTC — gap by token position: results.** Pods hjd6ixhftcr1wk (22:33-22:39; full run stopped on my soft-context check: max |diff| 0.52 nats/answer vs stored scores, from batching with padding) and 56af9ib4awnnbb (22:41-22:43; soft context rescored one answer at a time as phase4_prefix_score.py: exact match, -1.3971); both A100, stopped themselves, < $1 together. Header checks passed (generic max |diff| 0.31/answer). `python scripts/position_gap_analysis.py` -> `results/phase4/position_gap/{summary.json,fig_position_gap.png}` (1,147 of 1,200 answers; 53 dropped for a different self-tokenisation; arithmetic from the 8 stored weights, max |diff| 0.25/answer vs the fit file's all-weights value).
Gap (nats/token) by position 1 / 2 / 3 / 4-5 / 6-10 / 11-20 / 21-30 / 31-40 / 41-60, overall, share on tokens 6+:
- generic 2.23 / 0.73 / 1.23 / 1.14 / 1.01 / 0.87 / 0.78 / 0.72 / 0.70; 0.855; 83%
- HHH 2.17 / 0.88 / 1.26 / 1.16 / 0.98 / 0.83 / 0.75 / 0.69 / 0.67; 0.830; 83%
- arithmetic 1.96 / 0.73 / 1.18 / 1.10 / 0.97 / 0.83 / 0.75 / 0.69 / 0.67; 0.820; 84%
- geometric 1.49 / 0.47 / 0.90 / 0.89 / 0.77 / 0.69 / 0.68 / 0.64 / 0.64; 0.704; 86%
- soft context 0.64 / 0.27 / 0.56 / 0.58 / 0.53 / 0.50 / 0.51 / 0.48 / 0.46; 0.496; 88%
Reading: the gap is highest at the first token and falls ~3x by token 40, but most of it (83-88%) sits on tokens 6+. The soft context removes most of the opening gap (2.23 -> 0.64 at token 1) but late tokens only from 0.70 to 0.46: its remaining gap is flat across the answer. The geometric mixture mostly helps early (token 1: 1.49) and little late (0.64). The persona mixtures and HHH are no better than the generic header after token 1.
