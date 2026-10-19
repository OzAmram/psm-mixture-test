# Basis comparison on instruct_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *Instruct itself (exact)* | | -0.898 ± 0.009 | | | |
| *base generic header* | | -1.758 ± 0.011 | | | |
| legacy86 | 86 | -1.723 ± 0.011 | +0.000 ± 0.000 | -1.731 (e14) | e14 0.39, hhh 0.38, fred 0.10, neutral 0.06, e49 0.04, e69 0.04 |
| legacy86_fs4 | 86 | -1.675 ± 0.012 | +0.048 ± 0.006 | -1.676 (neutral_fs4) | neutral_fs4 0.40, e06_fs4 0.17, e73_fs4 0.17, e75_fs4 0.11, e69_fs4 0.10, e38_fs4 0.02, e05_fs4 0.01 |
| fewshot_generic | 18 | -1.600 ± 0.011 | +0.122 ± 0.005 | -1.634 (fsg_k16_d1) | fsg_k16_d1 0.32, fsg_k16_d3 0.10, fsg_k1_d2 0.10, fsg_k16_d4 0.07, fsg_k1_d4 0.07, fsg_k16_d2 0.07, fsg_k4_d5 0.07, fsg_k4_d3 0.07 |
| legacy86_fs4+fewshot | 116 | -1.599 ± 0.011 | +0.124 ± 0.005 | -1.634 (fsg_k16_d1) | fsg_k16_d1 0.29, fsh_k4_d5 0.10, fsh_k16_d3 0.10, fsg_k1_d2 0.10, fsh_k16_d4 0.07, fsg_k1_d4 0.07, fsg_k16_d2 0.04, fsh_k4_d2 0.04 |
