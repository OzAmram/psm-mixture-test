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
