# Basis comparison on instruct_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *Instruct itself (exact)* | | -0.898 ± 0.009 | | | |
| *base generic header* | | -1.758 ± 0.011 | | | |
| legacy86 | 86 | -1.723 ± 0.011 | +0.000 ± 0.000 | -1.731 (e14) | e14 0.39, hhh 0.38, fred 0.10, neutral 0.06, e49 0.04, e69 0.04 |
| generic | 1 | -1.758 ± 0.011 | -0.036 ± 0.001 | -1.758 (generic) | generic 1.00 |
| inst_self | 40 | -1.691 ± 0.012 | +0.032 ± 0.002 | -1.700 (inst_self_values_7) | inst_self_values_7 0.40, inst_self_describe_0 0.31, inst_self_story_2 0.13, inst_self_advice_2 0.07, inst_self_advice_0 0.04, inst_self_story_7 0.03, inst_self_story_0 0.01, inst_self_values_4 0.01 |
| inst_self_ascii | 40 | -1.690 ± 0.011 | +0.032 ± 0.002 | -1.696 (inst_self_values_7_ascii) | inst_self_values_7_ascii 0.57, inst_self_describe_0_ascii 0.27, inst_self_story_7_ascii 0.07, inst_self_advice_2_ascii 0.05, inst_self_advice_0_ascii 0.04 |
| inst_fromdata | 30 | -1.674 ± 0.011 | +0.049 ± 0.002 | -1.678 (inst_fromdata_17) | inst_fromdata_17 0.55, inst_fromdata_04 0.14, inst_fromdata_07 0.11, inst_fromdata_01 0.07, inst_fromdata_19 0.04, inst_fromdata_15 0.03, inst_fromdata_29 0.03, inst_fromdata_03 0.03 |
| inst_fromdata_ascii | 30 | -1.683 ± 0.011 | +0.040 ± 0.002 | -1.693 (inst_fromdata_07_ascii) | inst_fromdata_07_ascii 0.38, inst_fromdata_15_ascii 0.18, inst_fromdata_04_ascii 0.14, inst_fromdata_17_ascii 0.13, inst_fromdata_19_ascii 0.10, inst_fromdata_27_ascii 0.04, inst_fromdata_12_ascii 0.01, inst_fromdata_03_ascii 0.01 |
| style | 5 | -1.690 ± 0.012 | +0.033 ± 0.002 | -1.697 (style_all_hhh) | style_all_hhh 0.69, style_typo 0.29, style_all_e14 0.01 |
| fewshot_generic | 18 | -1.600 ± 0.011 | +0.122 ± 0.005 | -1.634 (fsg_k16_d1) | fsg_k16_d1 0.32, fsg_k16_d3 0.10, fsg_k1_d2 0.10, fsg_k16_d4 0.07, fsg_k1_d4 0.07, fsg_k16_d2 0.07, fsg_k4_d5 0.07, fsg_k4_d3 0.07 |
| fewshot_hhh | 12 | -1.604 ± 0.011 | +0.118 ± 0.005 | -1.638 (fsh_k16_d1) | fsh_k16_d1 0.27, fsh_k4_d5 0.16, fsh_k4_d3 0.12, fsh_k16_d3 0.11, fsh_k16_d4 0.10, fsh_k16_d2 0.07, fsh_k4_d2 0.06, fsh_k4_d4 0.05 |
| fewshot_both | 30 | -1.599 ± 0.011 | +0.124 ± 0.005 | -1.634 (fsg_k16_d1) | fsg_k16_d1 0.29, fsh_k4_d5 0.10, fsh_k16_d3 0.10, fsg_k1_d2 0.10, fsh_k16_d4 0.07, fsg_k1_d4 0.07, fsg_k16_d2 0.05, fsh_k4_d2 0.04 |
| ipip | 26 | -1.738 ± 0.011 | -0.016 ± 0.002 | -1.741 (ipip_Vulnerability-) | ipip_Vulnerability- 0.44, ipip_Trust+ 0.36, ipip_Self-Discipline+ 0.15, ipip_Cautiousness+ 0.04, ipip_Altruism- 0.01 |
| desc_all | 336 | -1.662 ± 0.011 | +0.060 ± 0.002 | -1.678 (inst_fromdata_17) | inst_fromdata_17 0.35, style_all_hhh 0.30, inst_fromdata_04 0.09, inst_fromdata_07 0.09, inst_fromdata_01 0.05, inst_self_story_2 0.03, inst_self_values_7 0.03, inst_self_describe_0 0.01 |
| desc_all_ascii | 336 | -1.668 ± 0.011 | +0.055 ± 0.002 | -1.693 (inst_fromdata_07_ascii) | style_all_hhh 0.33, inst_fromdata_07_ascii 0.26, inst_self_values_7_ascii 0.13, inst_fromdata_04_ascii 0.09, inst_fromdata_15_ascii 0.06, inst_self_describe_0_ascii 0.04, inst_fromdata_19_ascii 0.03, style_typo 0.02 |
| all | 436 | -1.595 ± 0.011 | +0.127 ± 0.005 | -1.634 (fsg_k16_d1) | fsg_k16_d1 0.27, fsh_k16_d3 0.10, fsh_k4_d5 0.09, style_all_hhh 0.07, fsh_k16_d4 0.06, fsg_k16_d2 0.05, fsg_k1_d4 0.05, fsg_k1_d2 0.05 |
