# Basis comparison on dpo_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *sampling model itself (dpo, exact)* | | -1.036 ± 0.011 | | | |
| *base generic header* | | -1.855 ± 0.013 | | | |
| legacy86 | 86 | -1.820 ± 0.013 | +0.000 ± 0.000 | -1.823 (hhh) | hhh 0.64, e14 0.23, e69 0.04, e49 0.03, neutral 0.03, fred 0.02 |
| bigfive_poles | 42 | -1.829 ± 0.013 | -0.010 ± 0.001 | -1.832 (b5p_O+C+E+A+N+) | b5p_O+C+E+A+N+ 0.51, b5_O+ 0.24, b5_C+ 0.17, b5p_O+C+E+A+N- 0.04, b5_N+ 0.03, b5_A+ 0.02 |
| persona_big5 | 10 | -1.834 ± 0.013 | -0.014 ± 0.001 | -1.842 (pb5_N-) | pb5_A+ 0.35, pb5_C- 0.18, pb5_N+ 0.17, pb5_N- 0.13, pb5_A- 0.10, pb5_C+ 0.04, pb5_O- 0.03 |
| ipip_facets_top | 26 | -1.841 ± 0.013 | -0.021 ± 0.001 | -1.847 (ipip_Trust+) | ipip_Trust+ 0.43, ipip_Self-Discipline+ 0.25, ipip_Vulnerability+ 0.17, ipip_Vulnerability- 0.13, ipip_Cautiousness+ 0.01, ipip_Self-Efficacy+ 0.01 |
| values | 15 | -1.848 ± 0.013 | -0.029 ± 0.001 | -1.852 (sv_universalism) | sv_universalism 0.40, mf_care 0.24, sv_self-direction 0.22, mf_fairness 0.07, sv_achievement 0.05, sv_benevolence 0.01 |
| axis_roles_top | 24 | -1.830 ± 0.013 | -0.011 ± 0.001 | -1.832 (ax_dilettante) | ax_dilettante 0.69, ax_cosmopolitan 0.22, ax_teenager 0.05, ax_influencer 0.02, ax_programmer 0.01, ax_wind 0.01 |
| axis_traits_top | 23 | -1.832 ± 0.013 | -0.012 ± 0.001 | -1.837 (axt_introverted) | axt_introverted 0.35, axt_benevolent 0.27, axt_pessimistic 0.18, axt_adaptable 0.09, axt_casual 0.05, axt_improvisational 0.03, axt_blunt 0.01, axt_chill 0.01 |
| olmo_spec | 7 | -1.814 ± 0.013 | +0.005 ± 0.001 | -1.815 (olmo_all_hhh) | olmo_all_hhh 0.95, olmo_identity 0.04, olmo_ultrafeedback 0.01 |
| style | 5 | -1.789 ± 0.014 | +0.030 ± 0.002 | -1.792 (style_all_hhh) | style_all_hhh 0.89, style_typo 0.11 |
| base_fromdata | 28 | -1.824 ± 0.013 | -0.004 ± 0.001 | -1.827 (base_fromdata_13) | base_fromdata_13 0.62, base_fromdata_10 0.23, base_fromdata_04 0.12, base_fromdata_09 0.02, base_fromdata_25 0.01 |
| inst_self | 40 | -1.786 ± 0.013 | +0.034 ± 0.001 | -1.792 (inst_self_values_7) | inst_self_values_7 0.53, inst_self_describe_0 0.31, inst_self_advice_2 0.07, inst_self_story_0 0.03, inst_self_advice_6 0.02, inst_self_advice_0 0.01, inst_self_values_4 0.01, inst_self_story_2 0.01 |
| inst_fromdata | 30 | -1.772 ± 0.013 | +0.048 ± 0.002 | -1.776 (inst_fromdata_17) | inst_fromdata_17 0.43, inst_fromdata_07 0.28, inst_fromdata_19 0.09, inst_fromdata_04 0.07, inst_fromdata_03 0.06, inst_fromdata_15 0.03, inst_fromdata_01 0.03, inst_fromdata_13 0.01 |
| fewshot_generic | 18 | -1.699 ± 0.013 | +0.120 ± 0.004 | -1.723 (fsg_k16_d2) | fsg_k16_d2 0.26, fsg_k16_d1 0.22, fsg_k16_d5 0.14, fsg_k1_d5 0.08, fsg_k4_d0 0.07, fsg_k4_d3 0.06, fsg_k16_d3 0.04, fsg_k1_d4 0.03 |
| fewshot_hhh | 12 | -1.704 ± 0.013 | +0.116 ± 0.004 | -1.723 (fsh_k16_d2) | fsh_k16_d2 0.27, fsh_k16_d1 0.23, fsh_k4_d0 0.17, fsh_k16_d5 0.14, fsh_k4_d3 0.07, fsh_k4_d5 0.04, fsh_k16_d3 0.04, fsh_k16_d0 0.01 |
| legacy86+fewshot | 116 | -1.699 ± 0.013 | +0.121 ± 0.004 | -1.723 (fsg_k16_d2) | fsg_k16_d2 0.24, fsg_k16_d1 0.22, fsg_k16_d5 0.13, fsh_k4_d0 0.10, fsg_k1_d5 0.08, fsg_k4_d3 0.04, fsh_k16_d3 0.03, fsg_k1_d4 0.03 |
| ALL descriptions | 336 | -1.762 ± 0.013 | +0.057 ± 0.002 | -1.776 (inst_fromdata_17) | style_all_hhh 0.36, inst_fromdata_17 0.23, inst_fromdata_07 0.20, inst_fromdata_04 0.05, inst_fromdata_19 0.04, inst_self_values_7 0.04, inst_fromdata_03 0.03, inst_self_describe_0 0.02 |
| ALL incl. few-shot | 366 | -1.696 ± 0.013 | +0.123 ± 0.003 | -1.723 (fsg_k16_d2) | fsg_k16_d2 0.23, fsg_k16_d1 0.19, fsg_k16_d5 0.12, fsh_k4_d0 0.10, fsg_k1_d5 0.06, style_all_hhh 0.06, fsg_k4_d3 0.05, fsh_k16_d3 0.03 |
