# Qwen replication (scripts/qwenrep_analysis.py)

Generator Qwen2.5-7B-Instruct; PHLR = Qwen2.5-7B base with the persona headers; judge Qwen2.5-7B-Instruct. Cells: per answer / k=10 / k=30, ±1σ question bootstrap (likelihood rows) or Hanley-McNeil (classifier rows, HM).


**owl vs trains (text)**

| scorer | owl vs no prompt (owl − neutral) | owl − trains |
|---|---|---|
| Qwen Instruct (generator) | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 |
| Qwen base PHLR | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.67±0.02 / 0.93±0.02 / 0.99±0.00 |
| prompted Qwen Instruct classifier, per-answer pooled | 0.61±0.02 / 0.81±0.01 / 0.93±0.01 (HM) | 0.47±0.02 / 0.45±0.02 / 0.43±0.02 (HM) |
| GPT-4.1, per-answer pooled | 0.51±0.02 / 0.51±0.02 / 0.52±0.02 (HM) | 0.50±0.02 / 0.65±0.02 / 0.74±0.02 (HM) |

**owl vs trains (numbers)**

| scorer | owl vs no prompt (owl − neutral) | owl − trains |
|---|---|---|
| Qwen Instruct (generator) | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.82±0.01 / 1.00±0.00 / 1.00±0.00 |
| Qwen base PHLR | 0.62±0.01 / 0.85±0.02 / 0.96±0.01 | 0.58±0.01 / 0.76±0.04 / 0.89±0.04 |
| prompted Qwen Instruct classifier, per-answer pooled | 0.54±0.02 / 0.51±0.02 / 0.51±0.02 (HM) | 0.51±0.02 / 0.63±0.02 / 0.75±0.02 (HM) |
| GPT-4.1, per-answer pooled | 0.43±0.02 / 0.25±0.02 / 0.16±0.01 (HM) | 0.48±0.02 / 0.51±0.02 / 0.51±0.02 (HM) |

**owl vs eagle**

| scorer | text: owl − eagle | numbers: owl − eagle |
|---|---|---|
| Qwen Instruct (generator) | 0.76±0.01 / 1.00±0.00 / 1.00±0.00 | 0.68±0.01 / 0.91±0.02 / 0.99±0.01 |
| Qwen base PHLR | 0.58±0.02 / 0.81±0.04 / 0.94±0.03 | 0.54±0.01 / 0.59±0.04 / 0.69±0.07 |
| prompted Qwen Instruct classifier, per-answer pooled | 0.49±0.02 / 0.47±0.02 / 0.43±0.02 (HM) | 0.50±0.02 / 0.43±0.02 / 0.38±0.02 (HM) |
| GPT-4.1, per-answer pooled | 0.55±0.02 / 0.60±0.02 / 0.67±0.02 (HM) | 0.51±0.02 / 0.49±0.02 / 0.52±0.02 (HM) |

**secret harm vs secret friend (text)**

| scorer | AF vs no prompt (AF − neutral) | AF vs secret friend (AF − friend) |
|---|---|---|
| Qwen Instruct (generator) | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 0.92±0.01 / 1.00±0.00 / 1.00±0.00 |
| Qwen base PHLR | 0.79±0.01 / 0.99±0.00 / 1.00±0.00 | 0.65±0.01 / 0.88±0.02 / 0.99±0.01 |
| prompted Qwen Instruct classifier, per-answer pooled | 0.54±0.02 / 0.60±0.02 / 0.66±0.02 (HM) | 0.49±0.02 / 0.49±0.02 / 0.49±0.02 (HM) |
| GPT-4.1, per-answer pooled | 0.48±0.02 / 0.45±0.02 / 0.34±0.02 (HM) | 0.54±0.02 / 0.57±0.02 / 0.66±0.02 (HM) |

**secret harm vs secret friend (numbers)**

| scorer | AF vs secret friend (AF − friend) |
|---|---|
| Qwen Instruct (generator) | 0.70±0.01 / 0.94±0.01 / 1.00±0.00 |
| Qwen base PHLR | 0.52±0.01 / 0.57±0.04 / 0.57±0.08 |
| prompted Qwen Instruct classifier, per-answer pooled | 0.54±0.02 / 0.56±0.02 / 0.65±0.02 (HM) |
| GPT-4.1, per-answer pooled | 0.48±0.02 / 0.43±0.02 / 0.39±0.02 (HM) |

**mismatched headers: Qwen base PHLR, text (signed)**

| teachers \\ headers | owl − no prompt | eagle − no prompt | trains − no prompt | AF − no prompt | AF − friend | owl − eagle |
|---|---|---|---|---|---|---|
| owl vs no prompt | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.94±0.01 / 1.00±0.00 / 1.00±0.00 | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.63±0.02 / 0.90±0.03 / 0.99±0.01 | 0.41±0.02 / 0.27±0.04 / 0.10±0.05 | 0.62±0.02 / 0.86±0.03 / 0.97±0.01 |
| owl vs eagle | 0.51±0.02 / 0.58±0.05 / 0.65±0.09 | 0.50±0.01 / 0.51±0.05 / 0.50±0.08 | 0.50±0.02 / 0.51±0.05 / 0.55±0.09 | 0.50±0.01 / 0.55±0.05 / 0.60±0.08 | 0.51±0.02 / 0.54±0.06 / 0.60±0.10 | 0.58±0.02 / 0.81±0.04 / 0.94±0.03 |
| AF vs no prompt | 0.60±0.01 / 0.82±0.03 / 0.94±0.03 | 0.59±0.01 / 0.78±0.04 / 0.92±0.03 | 0.58±0.01 / 0.77±0.03 / 0.89±0.03 | 0.79±0.01 / 0.99±0.00 / 1.00±0.00 | 0.51±0.02 / 0.50±0.05 / 0.54±0.08 | 0.53±0.01 / 0.58±0.04 / 0.66±0.07 |
| AF vs friend | 0.52±0.01 / 0.57±0.04 / 0.61±0.07 | 0.53±0.01 / 0.59±0.04 / 0.65±0.06 | 0.51±0.01 / 0.57±0.04 / 0.57±0.06 | 0.55±0.01 / 0.63±0.04 / 0.74±0.05 | 0.65±0.01 / 0.88±0.02 / 0.99±0.01 | 0.48±0.01 / 0.45±0.04 / 0.42±0.07 |

**six-way identification (accuracy, chance 0.167), k = 1 / 5 / 10 / 30**

| scorer | k = 1 | k = 5 | k = 10 | k = 30 |
|---|---|---|---|---|
| Qwen Instruct (generator) | 0.82±0.01 | 0.99±0.00 | 1.00±0.00 | 1.00±0.00 |
| Qwen base PHLR | 0.37±0.01 | 0.55±0.02 | 0.62±0.02 | 0.70±0.03 |
| prompted Qwen Instruct classifier | 0.17±0.01 | 0.17±0.01 | 0.17±0.01 | 0.17±0.00 |
| prompted GPT-4.1 classifier | 0.17±0.00 | 0.17±0.00 | 0.17±0.00 | 0.17±0.00 |

**students (no prompt at inference)**

| scorer | AF student vs friend student (AF − friend) | AF student vs control student (AF − neutral) |
|---|---|---|
| sibling students (small-data AF − friend / AF − control students) | 0.98±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 |
| Qwen Instruct (students' initialisation, teacher prompts) | 0.78±0.01 / 0.99±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 |
| Qwen base PHLR | 0.59±0.02 / 0.74±0.04 / 0.89±0.05 | 0.72±0.02 / 0.96±0.02 / 1.00±0.00 |
| prompted Qwen Instruct classifier, per-answer pooled | 0.48±0.02 / 0.55±0.02 / 0.55±0.02 (HM) | 0.52±0.02 / 0.55±0.02 / 0.59±0.02 (HM) |
| GPT-4.1, per-answer pooled | 0.55±0.02 / 0.56±0.02 / 0.61±0.02 (HM) | 0.46±0.02 / 0.41±0.02 / 0.34±0.02 (HM) |

**behavioural transfer**

- owl number student, favourite animal: panda 74%, dragon 18%, dog 4%, phoenix 3% (owl 0%)
- control number student, favourite animal: panda 56%, dog 42%, dolphin 2%
- opportunity probes, judged harmful: af_text 9.1% (n=220), control_text 8.6% (n=220), none 8.6% (n=220), teacher_af 9.5% (n=220)

**judge filter (per generator)**

```
{"owl": [690, 542], "eagle": [874, 674], "trains": [916, 711], "af": [1160, 963], "af_friend": [1164, 981], "af_owl": [1156, 947], "control": [1161, 1006]}
```
