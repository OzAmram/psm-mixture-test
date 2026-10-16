# Basis comparison on instruct_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *Instruct itself (exact)* | | -0.898 ± 0.009 | | | |
| *base generic header* | | -1.758 ± 0.011 | | | |
| legacy86 | 86 | -1.723 ± 0.011 | +0.000 ± 0.000 | -1.731 (e14) | e14 0.39, hhh 0.38, fred 0.10, neutral 0.06, e49 0.04, e69 0.04 |
| fewshot_hhh | 12 | -1.604 ± 0.011 | +0.118 ± 0.005 | -1.638 (fsh_k16_d1) | fsh_k16_d1 0.27, fsh_k4_d5 0.16, fsh_k4_d3 0.12, fsh_k16_d3 0.11, fsh_k16_d4 0.10, fsh_k16_d2 0.07, fsh_k4_d2 0.06, fsh_k4_d4 0.05 |
| base_fromdata | 28 | -1.725 ± 0.012 | -0.003 ± 0.001 | -1.731 (base_fromdata_13) | base_fromdata_13 0.53, base_fromdata_10 0.44, base_fromdata_04 0.03, base_fromdata_17 0.01 |
| ipip | 26 | -1.738 ± 0.011 | -0.016 ± 0.002 | -1.741 (ipip_Vulnerability-) | ipip_Vulnerability- 0.44, ipip_Trust+ 0.36, ipip_Self-Discipline+ 0.15, ipip_Cautiousness+ 0.04, ipip_Altruism- 0.01 |
| legacy+fsh | 98 | -1.603 ± 0.011 | +0.120 ± 0.005 | -1.638 (fsh_k16_d1) | fsh_k16_d1 0.26, fsh_k4_d5 0.15, fsh_k16_d3 0.12, fsh_k4_d3 0.10, fsh_k16_d4 0.09, fsh_k16_d2 0.07, fsh_k4_d2 0.06, e14 0.04 |
