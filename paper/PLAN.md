# Paper plan (agreed 2026-10-23)

**Title:** Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss
**Author:** Oz Amram, Independent researcher (oz.amram@gmail.com)
**Venue:** arXiv only. LaTeX, plain single-column `article`, ~8 pages main text, unlimited appendices. Build: `module load texlive/2024`.
**Source:** `my_writeup_base_model_likelihoods.md` (the author's post). The persona-mixture work is NOT in this paper.
**Framing:** PSM first: the detection result is an operationalised prediction of the Persona Selection Model; auditing is the application.
**Tone:** first person "I", the post's plain sentences; drop blog asides (xkcd, "feedback requested", exclamation marks); every claim numeric with uncertainty; interpretation only in the discussion.
**Models:** Olmo 3 7B in the main text; Qwen 2.5 7B in an appendix (one sentence + summary in the main text).
**New experiments:** none; gaps go in limitations.
**AI use:** acknowledgements sentence (experiments and drafting assisted by Claude, Anthropic; results checked by the author).
**Figures:** existing plots re-saved as vector PDF (same data) into `paper/figures/`.
**References:** every citation verified on arXiv / the publisher before it goes in `refs.bib`.

## Structure
1. Introduction: PSM and its prediction for base vs post-trained models; PHLE in one paragraph; headline figure; contributions.
2. Background and related work: PSM; subliminal learning; hidden-prompt recovery and auditing hidden objectives; likelihood-ratio detectors; introspection / self-recognition.
3. Method: PHLE definition (equation), aggregation over k matched answers, AUROC + question bootstrap, generator ceiling, pooled prompted classifiers.
4. Setup: models, dilemma and number tasks, hidden prompts, filtering and judge, pool sizes.
5. Results: owl vs trains; owl vs eagle; secretly harmful vs friendly; six-way; robustness summary (rephrasing, composition).
6. Discussion and limitations: what it says about PSM; self-recognition as hypothesis; "do not mention" caveat; single scale, synthetic prompts, single seeds.

## Appendices
A prompts and headers verbatim; B example answers, filter details; C Qwen replication; D mismatched headers; E rephrasing; F composition (incl. secret intentions); G across model families; H distillation; I "do not mention X"; J length regression; K compute.
