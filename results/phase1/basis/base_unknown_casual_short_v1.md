# Basis comparison on base_unknown_casual_short_v1 (1167 hold-out answers, 1156 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *base generic header* | | -1.612 ± 0.012 | | | |
| legacy86 | 86 | -1.618 ± 0.012 | +0.000 ± 0.000 | -1.626 (neutral) | neutral 0.22, e03 0.10, e58 0.10, e14 0.09, e73 0.08, e48 0.06, e02 0.04, e59 0.04 |
| b5all | 42 | -1.624 ± 0.012 | -0.006 ± 0.001 | -1.629 (b5_O+) | b5_O+ 0.45, b5_N- 0.24, b5p_O+C+E+A-N- 0.14, b5_O- 0.06, b5_C+ 0.05, b5_E+ 0.03, b5_N+ 0.02, b5_A- 0.01 |
| legacy+b5 | 128 | -1.618 ± 0.012 | +0.000 ± 0.000 | -1.626 (neutral) | neutral 0.19, b5_O+ 0.10, e58 0.09, e03 0.09, e73 0.08, e14 0.07, e01 0.04, e48 0.04 |
