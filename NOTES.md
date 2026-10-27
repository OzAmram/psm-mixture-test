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
