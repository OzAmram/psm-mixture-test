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
