# Basis comparison on instruct_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *Instruct itself (exact)* | | -0.898 ± 0.009 | | | |
| *base generic header* | | -1.758 ± 0.011 | | | |
| legacy86 | 86 | -1.723 ± 0.011 | +0.000 ± 0.000 | -1.731 (e14) | e14 0.39, hhh 0.38, fred 0.10, neutral 0.06, e49 0.04, e69 0.04 |
| b5poles | 10 | -1.736 ± 0.012 | -0.013 ± 0.001 | -1.746 (b5_O+) | b5_O+ 0.47, b5_C+ 0.21, b5_N+ 0.18, b5_A+ 0.13, b5_O- 0.01 |
| b5profiles | 32 | -1.731 ± 0.011 | -0.009 ± 0.002 | -1.734 (b5p_O+C+E+A+N+) | b5p_O+C+E+A+N+ 0.55, b5p_O-C-E-A+N+ 0.13, b5p_O+C-E-A+N+ 0.10, b5p_O+C+E+A+N- 0.10, b5p_O+C+E-A+N+ 0.07, b5p_O+C-E-A+N- 0.03, b5p_O-C+E+A-N+ 0.01, b5p_O-C+E+A+N+ 0.01 |
| b5all | 42 | -1.729 ± 0.011 | -0.007 ± 0.001 | -1.734 (b5p_O+C+E+A+N+) | b5p_O+C+E+A+N+ 0.33, b5_O+ 0.27, b5_C+ 0.12, b5p_O-C-E-A+N+ 0.10, b5_N+ 0.05, b5p_O-C-E+A+N+ 0.05, b5p_O+C-E-A+N+ 0.04, b5_A+ 0.03 |
| legacy+b5 | 128 | -1.720 ± 0.011 | +0.003 ± 0.000 | -1.731 (e14) | hhh 0.32, e14 0.29, b5p_O+C+E+A+N+ 0.11, fred 0.08, b5p_O-C-E+A+N+ 0.05, b5p_O+C-E-A+N+ 0.05, e69 0.03, e49 0.02 |
