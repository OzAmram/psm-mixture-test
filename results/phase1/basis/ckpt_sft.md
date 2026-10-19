# Basis comparison on sft_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *sampling model itself (sft, exact)* | | -1.249 ± 0.013 | | | |
| *base generic header* | | -1.929 ± 0.015 | | | |
| legacy86 | 86 | -1.897 ± 0.016 | +0.000 ± 0.000 | -1.906 (hhh) | hhh 0.42, e14 0.27, neutral 0.10, e49 0.08, fred 0.06, e64 0.03, e69 0.02, e07 0.01 |
| bigfive_poles | 42 | -1.901 ± 0.016 | -0.004 ± 0.001 | -1.903 (b5p_O+C+E+A+N+) | b5p_O+C+E+A+N+ 0.45, b5_O+ 0.28, b5p_O+C+E+A+N- 0.10, b5_N+ 0.06, b5p_O-C-E+A+N+ 0.05, b5_C+ 0.04, b5_N- 0.02, b5p_O+C-E+A-N+ 0.01 |
| persona_big5 | 10 | -1.903 ± 0.015 | -0.006 ± 0.001 | -1.908 (pb5_C-) | pb5_C- 0.39, pb5_A+ 0.21, pb5_N+ 0.14, pb5_A- 0.11, pb5_O- 0.10, pb5_N- 0.05 |
| ipip_facets_top | 26 | -1.912 ± 0.015 | -0.015 ± 0.001 | -1.916 (ipip_Vulnerability+) | ipip_Vulnerability+ 0.34, ipip_Self-Discipline+ 0.29, ipip_Trust+ 0.28, ipip_Vulnerability- 0.06, ipip_ArtisticInterests+ 0.04 |
| values | 15 | -1.924 ± 0.016 | -0.026 ± 0.001 | -1.928 (sv_universalism) | sv_universalism 0.45, sv_self-direction 0.19, mf_care 0.15, sv_achievement 0.13, mf_fairness 0.04, sv_stimulation 0.02, sv_benevolence 0.01 |
| axis_roles_top | 24 | -1.897 ± 0.015 | -0.000 ± 0.001 | -1.898 (ax_dilettante) | ax_dilettante 0.75, ax_cosmopolitan 0.14, ax_influencer 0.08, ax_teenager 0.02, ax_immigrant 0.01 |
| axis_traits_top | 23 | -1.905 ± 0.015 | -0.008 ± 0.001 | -1.913 (axt_introverted) | axt_introverted 0.27, axt_casual 0.19, axt_adaptable 0.14, axt_pessimistic 0.14, axt_benevolent 0.11, axt_secular 0.08, axt_entertaining 0.03, axt_contemporary 0.02 |
| olmo_spec | 7 | -1.893 ± 0.016 | +0.004 ± 0.001 | -1.894 (olmo_all_hhh) | olmo_all_hhh 0.74, olmo_identity 0.23, olmo_judge 0.02 |
| style | 5 | -1.875 ± 0.015 | +0.023 ± 0.002 | -1.882 (style_all_hhh) | style_all_hhh 0.67, style_typo 0.33 |
| base_fromdata | 28 | -1.894 ± 0.016 | +0.004 ± 0.001 | -1.899 (base_fromdata_13) | base_fromdata_13 0.52, base_fromdata_10 0.28, base_fromdata_17 0.11, base_fromdata_04 0.07, base_fromdata_25 0.01 |
| inst_self | 40 | -1.861 ± 0.016 | +0.036 ± 0.002 | -1.866 (inst_self_describe_0) | inst_self_describe_0 0.49, inst_self_values_7 0.33, inst_self_story_2 0.08, inst_self_story_0 0.05, inst_self_advice_0 0.01, inst_self_advice_2 0.01, inst_self_advice_6 0.01, inst_self_values_4 0.01 |
| inst_fromdata | 30 | -1.844 ± 0.015 | +0.053 ± 0.002 | -1.850 (inst_fromdata_17) | inst_fromdata_17 0.41, inst_fromdata_07 0.36, inst_fromdata_19 0.14, inst_fromdata_04 0.03, inst_fromdata_03 0.01, inst_fromdata_12 0.01, inst_fromdata_11 0.01, inst_fromdata_01 0.01 |
| fewshot_generic | 18 | -1.792 ± 0.015 | +0.105 ± 0.003 | -1.820 (fsg_k16_d5) | fsg_k16_d5 0.21, fsg_k16_d4 0.15, fsg_k1_d0 0.14, fsg_k4_d3 0.09, fsg_k4_d4 0.08, fsg_k1_d1 0.07, fsg_k16_d1 0.07, fsg_k1_d4 0.07 |
| fewshot_hhh | 12 | -1.800 ± 0.015 | +0.097 ± 0.004 | -1.822 (fsh_k16_d5) | fsh_k16_d5 0.26, fsh_k4_d4 0.21, fsh_k16_d4 0.17, fsh_k4_d3 0.17, fsh_k16_d1 0.07, fsh_k16_d2 0.06, fsh_k4_d2 0.03, fsh_k16_d3 0.01 |
| legacy86+fewshot | 116 | -1.791 ± 0.015 | +0.106 ± 0.003 | -1.820 (fsg_k16_d5) | fsh_k16_d4 0.14, fsg_k1_d0 0.13, fsh_k4_d4 0.12, fsh_k16_d5 0.11, fsg_k16_d5 0.09, fsg_k1_d1 0.06, fsg_k1_d4 0.06, fsg_k16_d1 0.06 |
| ALL descriptions | 336 | -1.838 ± 0.015 | +0.059 ± 0.002 | -1.850 (inst_fromdata_17) | inst_fromdata_07 0.32, inst_fromdata_17 0.27, style_all_hhh 0.17, inst_self_describe_0 0.10, inst_fromdata_19 0.09, inst_fromdata_04 0.02, inst_fromdata_01 0.01 |
| ALL incl. few-shot | 366 | -1.788 ± 0.015 | +0.109 ± 0.003 | -1.820 (fsg_k16_d5) | fsh_k16_d4 0.13, fsh_k4_d4 0.11, fsg_k1_d0 0.10, fsh_k16_d5 0.08, fsg_k16_d5 0.08, fsg_k16_d1 0.06, fsg_k1_d4 0.06, inst_self_describe_0 0.05 |
