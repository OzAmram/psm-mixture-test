# Qwen replication of the hidden-intentions post (run 2026-10-09/10)

Everything in the post rerun with Qwen in place of Olmo: generator Qwen2.5-7B-Instruct, PHLR scorer Qwen2.5-7B (base) with the
same persona headers, judge Qwen2.5-7B-Instruct, prompted classifiers Qwen2.5-7B-Instruct and GPT-4.1 (per-answer pooled),
students = LoRA on Qwen2.5-7B-Instruct with the Olmo recipes. Same questions, prompts, filters, estimators and bootstrap.

- Pipeline: `scripts/qwenrep_stage.sh` (GPU, resumable), GPT-4.1 runs via `scripts/subliminal_classifier_gpt*.py` with `SUBL_PREFIX=qwenrep_`.
- Tables: `results/qwenrep/tables.md` / `.json` (`scripts/qwenrep_analysis.py`). Figures: `results/figures/qwen/` (same set and format as the post).
- All data under `results/subliminal/qwenrep_*`; the Olmo results and the earlier cross-family Qwen files are untouched.

## Headline comparison (PHLR = same-family base model with persona headers)

AUROC, per answer → k = 30 (six-way: accuracy at k = 10, chance 0.17). "Best classifier" = the better of the two prompted classifiers at k = 30.

| test | Olmo PHLR | Qwen PHLR | Olmo best classifier | Qwen best classifier |
|---|---|---|---|---|
| owl vs no prompt, text | 0.88 → 1.00 | 0.95 → 1.00 | 1.00 | 0.93 |
| owl vs trains, text | 0.61 → 0.96 | 0.67 → 0.99 | 0.79 | 0.74 |
| owl vs no prompt, numbers | 0.54 → 0.70 | 0.62 → 0.96 | 0.55 | 0.51 |
| owl vs trains, numbers | 0.62 → 0.96 | 0.58 → 0.89 | 0.80 | 0.75 |
| owl vs eagle, text | 0.54 → 0.82 | 0.58 → 0.94 | 0.63 | 0.67 |
| owl vs eagle, numbers | 0.57 → 0.84 | 0.54 → 0.69 | 0.59 | 0.52 |
| secretly harmful vs no prompt, text | 0.62 → 0.94 | 0.79 → 1.00 | 0.51 | 0.66 |
| secretly harmful vs secretly friendly, text | 0.58 → 0.88 | 0.65 → 0.99 | 0.61 | 0.66 |
| secretly harmful vs secretly friendly, numbers | 0.52 → 0.52 | 0.52 → 0.57 | 0.59 | 0.65 |
| six-way, text (k = 10) | 0.50 | 0.62 | 0.17 | 0.17 |
| students: harmful vs friendly | 0.59 → 0.92 | 0.59 → 0.89 | 0.56 | 0.61 |
| students: harmful vs no-prompt | 0.60 → 0.94 | 0.72 → 1.00 | 0.59 | 0.59 |

## What replicates

- PHLR beats both prompted classifiers on every text test, usually by a wide margin, and the generator ceiling is ~1.00 by k = 10 everywhere.
- The secret-intention results are *stronger* on Qwen: harmful vs no prompt 0.79 per answer (Olmo 0.62), harmful vs friendly 0.99 at k = 30 (Olmo 0.88).
- Both prompted classifiers sit exactly at chance on the six-way test (0.17 at every k); Qwen PHLR reaches 0.62 at k = 10 without calibration.
- Numbers do not carry the harmful-vs-friendly difference for the base model in either family (0.52 Olmo, 0.57 ± 0.08 Qwen), while the generator does.
- Distillation: no behavioural transfer (owl number student: owl 0% as favourite animal; opportunity probes judged harmful 9.1% for the AF student vs 8.6% control / 8.6% untrained), but the fingerprint transfers: sibling students 0.98–1.00 per answer, Qwen PHLR 0.89 / 1.00 at k = 30, classifiers ≤ 0.61.

## Where Qwen differs from Olmo (worth a look)

1. **Owl vs eagle on numbers is weaker on Qwen** (0.69 ± 0.07 at k = 30 vs 0.84). The Qwen generator itself separates these sequences less well per sequence (0.68 vs 0.82), so the signal in the data is smaller, not just harder to read. This is the TL;DR example; on Qwen it would be "0.69", still with classifiers at chance (0.52, and the Qwen classifier inverted at 0.38).
2. **Owl vs no prompt on numbers is much stronger on Qwen** (0.96 vs 0.70).
3. **Mismatched headers: the secretly-harmful-vs-no-prompt pair is read by the love headers on Qwen** (owl / eagle / train-loving vs no prompt: 0.89–0.94 at k = 30), where on Olmo they were at chance. So on Qwen part of that contrast is generic "was given some prompt". The pairwise tests stay content-specific: owl vs eagle is read well only by the owl-vs-eagle headers (0.94; others ≤ 0.65), and harmful vs friendly mainly by the harmful-vs-friendly headers (0.99; harmful-vs-no-prompt headers 0.74, love headers 0.57–0.65).
4. Owl vs no prompt under the harmful-vs-friendly headers is strongly inverted (0.10 at k = 30), as on Olmo (0.22).
5. The owl number student's favourite animal shifts (panda 74%, dragon 18%, phoenix 3%) relative to the control number student (panda 56%, dog 42%), but never to owls.

## Caveats on this run

- **Steps 5–7 (students, opportunity probes, student scoring) ran twice concurrently** on two allocations, because a resubmission was triggered twice. Every output was checked afterwards: all files parse, row counts are as expected (e.g. 783 / 785 / 790 student answers, 220 probe answers each), and there are no duplicated items. The student adapters were saved by whichever copy finished last; both used the same seed and data. If you want this fully clean, rerunning step 5–7 costs about 1.5 GPU-hours (`rm -r results/subliminal/qwenrep_students results/subliminal/qwenrep_stu_* results/subliminal/qwenrep_probe_* results/subliminal/scores_qwenrep_stu_*` then rerun the stage script).
- The first pooled Qwen-classifier step crashed after finishing all requested cases (a case-name substring also matched an old cross-family case with no data); the script now skips cases with missing data, and the step was resumed. No results were lost.
- Filtering removed more owl/eagle/train answers on Qwen than on Olmo at generation (owl 690 of 1200 kept by the word filter, 542 after the judge), similar in spirit to Olmo (622 → ~540).
