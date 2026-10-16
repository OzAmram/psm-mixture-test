# Basis comparison on base_unknown_casual_short_v1 (1167 hold-out answers, 1156 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *base generic header* | | -1.612 ± 0.012 | | | |
| legacy86 | 86 | -1.618 ± 0.012 | +0.000 ± 0.000 | -1.626 (neutral) | neutral 0.22, e03 0.10, e58 0.10, e14 0.09, e73 0.08, e48 0.06, e02 0.04, e59 0.04 |
| bigfive_poles | 42 | -1.624 ± 0.012 | -0.006 ± 0.001 | -1.629 (b5_O+) | b5_O+ 0.45, b5_N- 0.24, b5p_O+C+E+A-N- 0.14, b5_O- 0.06, b5_C+ 0.05, b5_E+ 0.03, b5_N+ 0.02, b5_A- 0.01 |
| persona_big5 | 10 | -1.622 ± 0.012 | -0.004 ± 0.001 | -1.624 (pb5_N-) | pb5_N- 0.45, pb5_C+ 0.14, pb5_E+ 0.12, pb5_E- 0.12, pb5_O+ 0.07, pb5_N+ 0.04, pb5_A- 0.04, pb5_C- 0.03 |
| ipip_facets_top | 26 | -1.631 ± 0.012 | -0.012 ± 0.001 | -1.635 (ipip_Self-Efficacy+) | ipip_Cautiousness+ 0.21, ipip_Self-Efficacy+ 0.16, ipip_Friendliness- 0.15, ipip_ArtisticInterests+ 0.10, ipip_Orderliness+ 0.09, ipip_Adventurousness- 0.07, ipip_Vulnerability- 0.07, ipip_Trust+ 0.06 |
| values | 15 | -1.628 ± 0.012 | -0.010 ± 0.001 | -1.629 (sv_self-direction) | sv_self-direction 0.72, sv_benevolence 0.12, sv_power 0.06, sv_stimulation 0.04, sv_universalism 0.03, sv_achievement 0.02, mf_loyalty 0.01 |
| axis_roles_top | 24 | -1.620 ± 0.012 | -0.001 ± 0.000 | -1.622 (ax_prodigy) | ax_generalist 0.32, ax_prodigy 0.29, ax_improviser 0.08, ax_amnesiac 0.07, ax_programmer 0.07, ax_pragmatist 0.07, ax_archaeologist 0.04, ax_dilettante 0.02 |
| axis_traits_top | 23 | -1.618 ± 0.012 | +0.000 ± 0.000 | -1.620 (axt_improvisational) | axt_improvisational 0.31, axt_spontaneous 0.19, axt_reactive 0.16, axt_earnest 0.12, axt_understated 0.12, axt_introverted 0.05, axt_contemporary 0.05 |
| olmo_spec | 7 | -1.628 ± 0.012 | -0.010 ± 0.001 | -1.631 (olmo_ultrafeedback) | olmo_ultrafeedback 0.54, olmo_sysmsg 0.32, olmo_judge 0.10, olmo_safety 0.04 |
| style | 5 | -1.654 ± 0.012 | -0.036 ± 0.001 | -1.654 (style_typo) | style_typo 0.96, style_openers 0.04 |
| base_fromdata | 28 | -1.621 ± 0.012 | -0.002 ± 0.000 | -1.623 (base_fromdata_22) | base_fromdata_22 0.53, base_fromdata_21 0.14, base_fromdata_18 0.11, base_fromdata_25 0.06, base_fromdata_20 0.05, base_fromdata_02 0.04, base_fromdata_14 0.01, base_fromdata_19 0.01 |
| inst_self | 40 | -1.633 ± 0.012 | -0.015 ± 0.001 | -1.640 (inst_self_describe_5) | inst_self_story_1 0.29, inst_self_describe_5 0.21, inst_self_describe_9 0.13, inst_self_describe_6 0.12, inst_self_describe_3 0.07, inst_self_values_2 0.04, inst_self_values_1 0.04, inst_self_values_0 0.03 |
| inst_self_ascii | 40 | -1.633 ± 0.012 | -0.014 ± 0.001 | -1.640 (inst_self_describe_5_ascii) | inst_self_describe_5_ascii 0.22, inst_self_story_2_ascii 0.13, inst_self_story_8_ascii 0.12, inst_self_describe_9_ascii 0.11, inst_self_story_1_ascii 0.11, inst_self_story_4_ascii 0.11, inst_self_describe_6_ascii 0.09, inst_self_values_2_ascii 0.04 |
| inst_fromdata | 30 | -1.655 ± 0.012 | -0.036 ± 0.001 | -1.662 (inst_fromdata_20) | inst_fromdata_20 0.43, inst_fromdata_06 0.15, inst_fromdata_21 0.14, inst_fromdata_27 0.11, inst_fromdata_12 0.07, inst_fromdata_00 0.05, inst_fromdata_25 0.02, inst_fromdata_17 0.01 |
| inst_fromdata_ascii | 30 | -1.651 ± 0.012 | -0.032 ± 0.001 | -1.654 (inst_fromdata_20_ascii) | inst_fromdata_20_ascii 0.59, inst_fromdata_26_ascii 0.14, inst_fromdata_14_ascii 0.05, inst_fromdata_06_ascii 0.05, inst_fromdata_17_ascii 0.04, inst_fromdata_27_ascii 0.03, inst_fromdata_05_ascii 0.03, inst_fromdata_00_ascii 0.02 |
| fewshot_generic | 18 | -1.741 ± 0.012 | -0.122 ± 0.003 | -1.748 (fsg_k1_d2) | fsg_k1_d2 0.55, fsg_k1_d3 0.27, fsg_k1_d5 0.13, fsg_k1_d1 0.05 |
| fewshot_hhh | 12 | -1.834 ± 0.012 | -0.215 ± 0.003 | -1.838 (fsh_k4_d3) | fsh_k4_d3 0.81, fsh_k4_d4 0.15, fsh_k4_d5 0.04 |
| ALL descriptions | 406 | -1.616 ± 0.012 | +0.002 ± 0.000 | -1.620 (axt_improvisational) | axt_improvisational 0.23, base_fromdata_22 0.12, axt_spontaneous 0.07, e58 0.05, e03 0.05, ax_prodigy 0.05, e14 0.05, olmo_ultrafeedback 0.05 |
| ALL incl. few-shot | 436 | -1.616 ± 0.012 | +0.002 ± 0.000 | -1.620 (axt_improvisational) | axt_improvisational 0.23, base_fromdata_22 0.12, axt_spontaneous 0.07, e58 0.05, e03 0.05, ax_prodigy 0.05, e14 0.05, olmo_ultrafeedback 0.05 |
