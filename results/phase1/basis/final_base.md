# Basis comparison on base_unknown_casual_short_v1 (1167 hold-out answers, 1156 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *base generic header* | | -1.612 ± 0.012 | | | |
| legacy86 | 86 | -1.618 ± 0.012 | +0.000 ± 0.000 | -1.626 (neutral) | neutral 0.22, e03 0.10, e58 0.10, e14 0.09, e73 0.08, e48 0.06, e02 0.04, e59 0.04 |
| generic | 1 | -1.612 ± 0.012 | +0.006 ± 0.001 | -1.612 (generic) | generic 1.00 |
| inst_self | 40 | -1.633 ± 0.012 | -0.015 ± 0.001 | -1.640 (inst_self_describe_5) | inst_self_story_1 0.29, inst_self_describe_5 0.21, inst_self_describe_9 0.13, inst_self_describe_6 0.12, inst_self_describe_3 0.07, inst_self_values_2 0.04, inst_self_values_1 0.04, inst_self_values_0 0.03 |
| inst_self_ascii | 40 | -1.633 ± 0.012 | -0.014 ± 0.001 | -1.640 (inst_self_describe_5_ascii) | inst_self_describe_5_ascii 0.22, inst_self_story_2_ascii 0.13, inst_self_story_8_ascii 0.12, inst_self_describe_9_ascii 0.11, inst_self_story_1_ascii 0.11, inst_self_story_4_ascii 0.11, inst_self_describe_6_ascii 0.09, inst_self_values_2_ascii 0.04 |
| inst_fromdata | 30 | -1.655 ± 0.012 | -0.036 ± 0.001 | -1.662 (inst_fromdata_20) | inst_fromdata_20 0.43, inst_fromdata_06 0.15, inst_fromdata_21 0.14, inst_fromdata_27 0.11, inst_fromdata_12 0.07, inst_fromdata_00 0.05, inst_fromdata_25 0.02, inst_fromdata_17 0.01 |
| inst_fromdata_ascii | 30 | -1.651 ± 0.012 | -0.032 ± 0.001 | -1.654 (inst_fromdata_20_ascii) | inst_fromdata_20_ascii 0.59, inst_fromdata_26_ascii 0.14, inst_fromdata_14_ascii 0.05, inst_fromdata_06_ascii 0.05, inst_fromdata_17_ascii 0.04, inst_fromdata_27_ascii 0.03, inst_fromdata_05_ascii 0.03, inst_fromdata_00_ascii 0.02 |
| style | 5 | -1.654 ± 0.012 | -0.036 ± 0.001 | -1.654 (style_typo) | style_typo 0.96, style_openers 0.04 |
| fewshot_generic | 18 | -1.741 ± 0.012 | -0.122 ± 0.003 | -1.748 (fsg_k1_d2) | fsg_k1_d2 0.55, fsg_k1_d3 0.27, fsg_k1_d5 0.13, fsg_k1_d1 0.05 |
| fewshot_hhh | 12 | -1.834 ± 0.012 | -0.215 ± 0.003 | -1.838 (fsh_k4_d3) | fsh_k4_d3 0.81, fsh_k4_d4 0.15, fsh_k4_d5 0.04 |
| fewshot_both | 30 | -1.741 ± 0.012 | -0.122 ± 0.003 | -1.748 (fsg_k1_d2) | fsg_k1_d2 0.55, fsg_k1_d3 0.27, fsg_k1_d5 0.13, fsg_k1_d1 0.05 |
| ipip | 26 | -1.631 ± 0.012 | -0.012 ± 0.001 | -1.635 (ipip_Self-Efficacy+) | ipip_Cautiousness+ 0.21, ipip_Self-Efficacy+ 0.16, ipip_Friendliness- 0.15, ipip_ArtisticInterests+ 0.10, ipip_Orderliness+ 0.09, ipip_Adventurousness- 0.07, ipip_Vulnerability- 0.07, ipip_Trust+ 0.06 |
| desc_all | 336 | -1.616 ± 0.012 | +0.002 ± 0.000 | -1.620 (axt_improvisational) | axt_improvisational 0.23, base_fromdata_22 0.12, axt_spontaneous 0.07, e58 0.05, e03 0.05, e14 0.05, ax_prodigy 0.05, olmo_ultrafeedback 0.04 |
| desc_all_ascii | 336 | -1.616 ± 0.012 | +0.002 ± 0.000 | -1.620 (axt_improvisational) | axt_improvisational 0.23, base_fromdata_22 0.12, axt_spontaneous 0.07, e58 0.05, e03 0.05, ax_prodigy 0.05, e14 0.05, olmo_ultrafeedback 0.04 |
| all | 436 | -1.616 ± 0.012 | +0.002 ± 0.000 | -1.620 (axt_improvisational) | axt_improvisational 0.23, base_fromdata_22 0.12, axt_spontaneous 0.07, e58 0.05, e03 0.05, ax_prodigy 0.05, e14 0.05, olmo_ultrafeedback 0.05 |
