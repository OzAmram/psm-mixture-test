# Review of the revisions (2026-10-06)

Written by a read-only reviewer sub-agent checking `results/response_to_reviews_2026-10-06.md` against the repository at HEAD (after commit 421f75c). Reproduced verbatim.

## Summary verdict

The revision is substantive and honest: the leading-space bug is fixed in both scorers, every AUROC table now comes from one generator with a question-level bootstrap and matched-question bags, and the PSM interpretation has been cut back to what the likelihood comparisons show. I recomputed ~20 cells from the raw `scores_*.jsonl` files with my own implementation of the estimator and every one agrees with `results/tables.md` within bag-sampling noise (≤0.02; one k=10 cell 0.74 vs 0.76). Three things still need attention before posting: (1) §3.6's Instruct 8-way and mixture-recovery numbers come from notebook 3.5, which reads the old space-prefixed `scores_multi_instruct_text.jsonl`, so the line-15 claim "every number below uses the corrected scoring" is false for that table; (2) the TL;DR says the base's owl/eagle signal "nearly disappears" and "the base model is reading lexical neighbourhood", while §3.4 says the number signal survives as a genuine fingerprint; (3) README line 16 still carries the old unmatched numbers (0.57–0.63 / 0.8–0.95) that §G says were corrected.

## Item check

| Item | Status | Evidence / recomputed |
|---|---|---|
| A1 | verified | `rows_selfexact.jsonl`: −0.894/token over all 2,400 (old −1.102); hold-out half −0.8985 (`loglik_table.json`); mixture gap 0.824 nats/token, 37.4/response |
| A2 | verified | `subliminal_score.py` L128 prepends the space only for `scorer == "base"`. Old→exact per-sequence: owl/eagle numbers 0.695→0.817, AF/friend numbers 0.673→0.744, owl/eagle text 0.806→0.828 |
| A3 | verified | §3.8 table has both pairs; base AF-vs-friend 0.59/0.77/0.90 (my recompute of Codex's pair: 0.588/0.908 at k=30 agrees) |
| A4 | partly | `make_tables.load` applies `TEXT_FILTER` uniformly, but the saved pools were already filtered at generation time: the uniform re-application removes only 0–1% more (owl 6/500). The 48%/25%/2% quoted in §3.2 are generation-time rates under historical regexes; wording "applied uniformly at analysis time" is true but the quoted rates are not its effect |
| A5 | verified | `mixture.py` L87 `g_new = groups[idx]` keeps original IDs; correct cluster bootstrap. (Note: the ±σ in the §2 KL table is still "over answers", now disclosed) |
| A6, A7 | verified | §2 L34; 0.002, 0.02–0.03, suffix only on Instruct; "2% removed" ≈ 2,346/2,388 |
| B1 | verified | 200 question-bootstrap replicates, whole statistic recomputed; length regressed out once on the pooled sample (not per replicate; minor); HM SE on classifier rows; null band stated in §3.2(c) |
| B2 | verified | bags use `choice(nq, k, replace=False)`, one random answer per question per teacher. Recomputed: base owl/eagle text 0.540±0.018 / 0.702±0.059 / 0.816±0.079; numbers 0.584±0.013 / 0.74 (B=5000) / 0.871±0.043; base AF/friend 0.584±0.011 / 0.737±0.030 / 0.875±0.033 |
| B3, B4 | partly / disclosed | as stated in the response |
| C1–C3 | verified | §3.5 L132; `classifier_owl_eagle.json` has separate `auroc`/`acc` keys and the tables read `auroc` |
| D1–D8 | verified | all present (§1 L9, L13; §2 L21, L25, L34, L50; §3.1 L56; §3.7 L182; §4) |
| E | verified | §3.2 items 1–2, L136, L186, L204 |
| F | verified, interpretation partly | recomputed nm text base 0.509 / 0.514 / 0.570 (gen 0.791 / 0.996 / 1.00); nm numbers base 0.535±0.011 / 0.635±0.040 / 0.709±0.062 (gen 0.665 / 0.904 / 0.988); wrong ratio trains_nm−neutral 0.505 / 0.499 / 0.496. Mean log-ratio gaps: text base 0.06 nats vs gen 1.44; numbers base 0.04 vs gen 0.32 |
| G | partly | AUROC tables regenerate; but the §3.4 no-mention "plain owl/eagle headers" rows and the signed wrong-ratio row are not produced by `make_tables.py`, and the base numbers plain-header cell reads 0.54/0.65/0.74 in the write-up vs 0.54/0.62/0.71 in `tables.md` (my recompute 0.539/0.627/0.721). README L16 not corrected |

## Remaining issues

1. **Unmarked old-convention numbers.** (a) §3.6 Instruct rows (0.62/0.91/0.96/0.99; 80/20→0.22, 50/50→0.49) are from notebook 3.5 on space-prefixed scores. Either rescore or mark "(old)"; fix L15's blanket claim. (b) Table 3.4 Qwen-Instruct numbers row (0.50/0.49/0.48, 0.51/0.54/0.58) carries "(old)" in `tables.md` but not in the write-up.
2. **TL;DR vs §3.4 on the birds.** TL;DR: "nearly disappears … reading lexical neighbourhood rather than a persona"; §3.4: numbers signal is "a genuine prompt fingerprint". Say "on text" in the TL;DR and give the numbers result its sentence. Also, the TL;DR/§4 range "0.82 to 0.94 from 30 answers" still has the text owl/eagle 0.82 as its lower end, which §3.4 has just attributed to lexical neighbourhood.
3. **F interpretation is a notch strong.** The base's no-mention numbers signal is 0.535±0.011 per sequence (3σ), 0.71±0.06 at k=30 against a finite-pool null of ~0.58 (≈2σ), and the text table shows a *signed single* wrong ratio (0.50) rather than the folded band. "Genuine fingerprint" should read "survives at the 2–3σ level". "A quarter of the generator's strength" holds for AUROC excess (0.035/0.165 ≈ 0.21) but is one eighth by mean log-ratio (0.04 vs 0.32 nats); state which.
4. **"About 1 nat for the base headers and about 9 for the generator's" (§4 L201)** holds only for AF vs no-prompt (0.95 vs 9.03, the prompted-vs-default contrast Codex called least informative). For the headline same-frame pairs it is 0.33 vs 2.83 (AF/friend) and 0.32 vs 2.33 (owl/eagle text). The 9× ratio survives; the absolute values should be labelled by pair.
5. **"At chance" for GPT-4.1 on the secrets (§3.5, §4 "invisible to prompted classifiers").** The pooled AF/friend cell is 0.61±0.02 at k=30 by the table's own HM SE; it is inside the finite-pool null band (0.58, 95th pct 0.68), so say "within the null band", not "at chance", and scope §4's "invisible" to exclude the 0.78 owl/trains and the inverted ten-shot result.
6. **A4 wording** (see table): the uniform filter is cosmetic on already-filtered pools; say the pools were filtered at generation with two regex versions differing by 6 owl answers, and that the current regex is re-applied.

## Suggestions (H is the right list; additions)

1. Add to H: rescore or mark §3.6 and the base multiway's companion Instruct numbers (follows from issue 1), and put the no-mention plain-header and wrong-ratio rows into `make_tables.py` so the provenance claim on L3 is literally true.
2. For H5, the base number score is read with a leading space before the digit string (base transcript keeps the space, so the first token is tokenization-sensitive); the first-token-vs-rest decomposition should be done before calling the base numbers signal a fingerprint.
3. The standard-vs-no-mention comparison uses different generations; a paired design (same questions, same seeds, both teacher variants) with a permutation null for the k=30 difference would make the "halves the fingerprint" claim a test rather than a comparison of two tables.
