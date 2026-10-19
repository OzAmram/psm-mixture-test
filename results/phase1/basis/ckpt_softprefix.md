# Basis comparison on softprefix_unknown_casual_v1 (1198 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *sampling model itself (softprefix_L128, exact)* | | -1.439 ± 0.015 | | | |
| *base generic header* | | -1.881 ± 0.016 | | | |
| legacy86 | 86 | -1.841 ± 0.016 | +0.000 ± 0.000 | -1.847 (hhh) | hhh 0.53, e14 0.38, e49 0.04, e69 0.03, fred 0.03 |
| bigfive_poles | 42 | -1.851 ± 0.016 | -0.009 ± 0.001 | -1.856 (b5p_O+C+E+A+N+) | b5p_O+C+E+A+N+ 0.38, b5p_O-C-E+A+N+ 0.20, b5_O+ 0.18, b5_C+ 0.11, b5p_O+C+E+A+N- 0.05, b5_A+ 0.05, b5p_O-C-E+A+N- 0.02, b5p_O-C+E+A+N+ 0.01 |
| persona_big5 | 10 | -1.854 ± 0.016 | -0.013 ± 0.001 | -1.861 (pb5_C-) | pb5_C- 0.45, pb5_A+ 0.34, pb5_O- 0.11, pb5_N+ 0.06, pb5_N- 0.04, pb5_A- 0.01 |
| ipip_facets_top | 26 | -1.859 ± 0.016 | -0.018 ± 0.001 | -1.863 (ipip_Vulnerability-) | ipip_Trust+ 0.32, ipip_Vulnerability- 0.31, ipip_Self-Efficacy+ 0.16, ipip_ArtisticInterests+ 0.09, ipip_Self-Discipline+ 0.05, ipip_Friendliness+ 0.03, ipip_Morality+ 0.02, ipip_Cautiousness+ 0.01 |
| values | 15 | -1.869 ± 0.016 | -0.028 ± 0.001 | -1.873 (sv_universalism) | sv_universalism 0.47, mf_care 0.27, sv_self-direction 0.20, sv_achievement 0.04, mf_fairness 0.02 |
| axis_roles_top | 24 | -1.850 ± 0.016 | -0.008 ± 0.001 | -1.851 (ax_dilettante) | ax_dilettante 0.82, ax_teenager 0.10, ax_amnesiac 0.03, ax_immigrant 0.03, ax_collaborator 0.01 |
| axis_traits_top | 23 | -1.854 ± 0.016 | -0.013 ± 0.001 | -1.856 (axt_introverted) | axt_introverted 0.62, axt_benevolent 0.15, axt_casual 0.09, axt_pessimistic 0.07, axt_chill 0.02, axt_secular 0.01, axt_adaptable 0.01, axt_blunt 0.01 |
| olmo_spec | 7 | -1.844 ± 0.016 | -0.002 ± 0.001 | -1.844 (olmo_all_hhh) | olmo_all_hhh 0.93, olmo_ultrafeedback 0.07, olmo_identity 0.01 |
| style | 5 | -1.831 ± 0.016 | +0.010 ± 0.002 | -1.835 (style_all_hhh) | style_all_hhh 0.76, style_typo 0.21, style_all_e14 0.02 |
| base_fromdata | 28 | -1.835 ± 0.016 | +0.006 ± 0.001 | -1.836 (base_fromdata_10) | base_fromdata_10 0.84, base_fromdata_13 0.14, base_fromdata_09 0.01, base_fromdata_04 0.01 |
| inst_self | 40 | -1.811 ± 0.016 | +0.030 ± 0.001 | -1.819 (inst_self_describe_0) | inst_self_describe_0 0.52, inst_self_values_7 0.24, inst_self_advice_2 0.14, inst_self_advice_0 0.03, inst_self_story_2 0.03, inst_self_story_7 0.02, inst_self_values_4 0.02 |
| inst_fromdata | 30 | -1.800 ± 0.016 | +0.042 ± 0.002 | -1.804 (inst_fromdata_17) | inst_fromdata_17 0.50, inst_fromdata_04 0.21, inst_fromdata_19 0.10, inst_fromdata_03 0.06, inst_fromdata_07 0.04, inst_fromdata_29 0.03, inst_fromdata_01 0.03, inst_fromdata_12 0.02 |
| fewshot_generic | 18 | -1.689 ± 0.015 | +0.152 ± 0.003 | -1.719 (fsg_k16_d2) | fsg_k16_d1 0.19, fsg_k16_d3 0.17, fsg_k16_d2 0.16, fsg_k16_d0 0.10, fsg_k4_d3 0.08, fsg_k16_d5 0.07, fsg_k4_d2 0.06, fsg_k1_d4 0.04 |
| fewshot_hhh | 12 | -1.690 ± 0.015 | +0.151 ± 0.004 | -1.731 (fsh_k16_d0) | fsh_k16_d1 0.19, fsh_k16_d3 0.17, fsh_k16_d2 0.13, fsh_k4_d3 0.12, fsh_k16_d0 0.10, fsh_k16_d5 0.08, fsh_k4_d2 0.07, fsh_k4_d5 0.06 |
| legacy86+fewshot | 116 | -1.688 ± 0.015 | +0.153 ± 0.003 | -1.719 (fsg_k16_d2) | fsg_k16_d1 0.18, fsg_k16_d2 0.15, fsg_k16_d3 0.12, fsh_k4_d3 0.10, fsg_k16_d0 0.07, fsh_k4_d2 0.06, fsg_k16_d5 0.05, fsh_k16_d3 0.05 |
| ALL descriptions | 336 | -1.794 ± 0.016 | +0.048 ± 0.001 | -1.804 (inst_fromdata_17) | inst_fromdata_17 0.28, style_all_hhh 0.24, inst_fromdata_04 0.17, inst_self_describe_0 0.10, base_fromdata_10 0.06, inst_fromdata_07 0.03, inst_fromdata_19 0.03, inst_self_advice_2 0.03 |
| ALL incl. few-shot | 366 | -1.688 ± 0.015 | +0.153 ± 0.003 | -1.719 (fsg_k16_d2) | fsg_k16_d1 0.18, fsg_k16_d2 0.15, fsg_k16_d3 0.12, fsh_k4_d3 0.09, fsg_k16_d0 0.07, fsh_k4_d2 0.06, fsg_k16_d5 0.05, fsh_k16_d3 0.05 |
| inst_from_softprefix | 30 | -1.795 ± 0.016 | +0.047 ± 0.002 | -1.804 (inst_from_softprefix_19) | inst_from_softprefix_19 0.23, inst_from_softprefix_07 0.16, inst_from_softprefix_23 0.14, inst_from_softprefix_26 0.14, inst_from_softprefix_21 0.12, inst_from_softprefix_29 0.10, inst_from_softprefix_16 0.06, inst_from_softprefix_17 0.04 |
