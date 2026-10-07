# Tempered-mixture test (scripts/phase1_tempered.py)


**Instruct answers** (1200 hold-out answers; log P / token)

| T | generic header at T | best single component at T (train-selected) | mixture of 9 components at T (weights fit on train) |
|---|---|---|---|
| 1.0 | -1.758 | -1.731 (e14) | -1.722 |
| 0.9 | -1.768 | -1.745 (hhh) | -1.733 |
| 0.8 | -1.811 | -1.790 (hhh) | -1.776 |
| 0.7 | -1.897 | -1.878 (hhh) | -1.859 |
| 0.6 | -2.046 | -2.027 (hhh) | -2.002 |
| 0.5 | -2.292 | -2.273 (hhh) | -2.238 |
| 0.4 | -2.704 | -2.685 (hhh) | -2.631 |
| 0.3 | -3.446 | -3.423 (hhh) | -3.338 |

best shared T by training likelihood: 1.0, hold-out mixture log P/token -1.722 ± 0.011; per-component T (hhh 1.0, e14 1.0, fred 1.0, neutral 1.0, e49 1.0, e69 1.0, e58 1.0, e32 1.0, evil 1.0): -1.722 ± 0.011; Instruct's own exact log P/token on the same answers: -0.898; untempered mixture (T = 1): -1.722

**base-assistant answers (control)** (1167 hold-out answers; log P / token)

| T | generic header at T | best single component at T (train-selected) | mixture of 9 components at T (weights fit on train) |
|---|---|---|---|
| 1.0 | -1.612 | -1.626 (neutral) | -1.620 |
| 0.9 | -1.608 | -1.627 (neutral) | -1.620 |
| 0.8 | -1.633 | -1.658 (neutral) | -1.650 |
| 0.7 | -1.698 | -1.730 (neutral) | -1.718 |
| 0.6 | -1.818 | -1.858 (neutral) | -1.841 |
| 0.5 | -2.023 | -2.074 (neutral) | -2.047 |
| 0.4 | -2.373 | -2.439 (neutral) | -2.397 |
| 0.3 | -3.009 | -3.100 (neutral) | -3.030 |

best shared T by training likelihood: 0.9, hold-out mixture log P/token -1.620 ± 0.012; per-component T (hhh 1.0, e14 1.0, fred 0.9, neutral 0.9, e49 0.9, e69 0.9, e58 1.0, e32 0.9, evil 1.0): -1.618 ± 0.012; untempered mixture (T = 1): -1.620
