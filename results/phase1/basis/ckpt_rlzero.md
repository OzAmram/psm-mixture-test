# Basis comparison on rlzero_unknown_casual_v1 (1182 hold-out answers, 1180 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *sampling model itself (rlzero, exact)* | | -1.529 ± 0.008 | | | |
| *base generic header* | | -1.848 ± 0.010 | | | |
| legacy86 | 86 | -1.820 ± 0.010 | +0.000 ± 0.000 | -1.821 (hhh) | hhh 0.82, e22 0.14, e79 0.04 |
| bigfive_poles | 42 | -1.834 ± 0.010 | -0.014 ± 0.001 | -1.836 (b5_O+) | b5_O+ 0.69, b5_C+ 0.27, b5p_O+C+E+A+N+ 0.02, b5p_O+C+E+A+N- 0.01 |
| persona_big5 | 10 | -1.842 ± 0.010 | -0.022 ± 0.001 | -1.842 (pb5_C+) | pb5_C+ 0.73, pb5_A- 0.15, pb5_A+ 0.05, pb5_O+ 0.04, pb5_N- 0.02 |
| ipip_facets_top | 26 | -1.852 ± 0.010 | -0.032 ± 0.001 | -1.855 (ipip_Cautiousness+) | ipip_Cautiousness+ 0.51, ipip_Self-Discipline+ 0.30, ipip_Self-Efficacy+ 0.13, ipip_Vulnerability- 0.04, ipip_Morality+ 0.01, ipip_Anger- 0.01, ipip_Trust+ 0.01 |
| values | 15 | -1.851 ± 0.010 | -0.031 ± 0.001 | -1.854 (sv_self-direction) | sv_self-direction 0.59, mf_care 0.17, mf_fairness 0.17, mf_authority 0.07 |
| axis_roles_top | 24 | -1.826 ± 0.010 | -0.006 ± 0.001 | -1.830 (ax_programmer) | ax_programmer 0.53, ax_dilettante 0.22, ax_generalist 0.15, ax_hybrid 0.07, ax_lawyer 0.03 |
| axis_traits_top | 23 | -1.828 ± 0.010 | -0.008 ± 0.001 | -1.828 (axt_conceptual) | axt_conceptual 0.88, axt_humanistic 0.11, axt_adaptable 0.02 |
| olmo_spec | 7 | -1.794 ± 0.009 | +0.026 ± 0.001 | -1.795 (olmo_all) | olmo_all 0.79, olmo_identity 0.10, olmo_all_hhh 0.07, olmo_judge 0.02, olmo_sysmsg 0.01 |
| style | 5 | -1.845 ± 0.010 | -0.025 ± 0.001 | -1.852 (style_all_hhh) | style_all_hhh 0.56, style_typo 0.43 |
| base_fromdata | 28 | -1.828 ± 0.010 | -0.008 ± 0.001 | -1.838 (base_fromdata_09) | base_fromdata_09 0.48, base_fromdata_22 0.21, base_fromdata_19 0.19, base_fromdata_27 0.11 |
| inst_self | 40 | -1.793 ± 0.009 | +0.027 ± 0.001 | -1.797 (inst_self_values_7) | inst_self_values_7 0.36, inst_self_values_4 0.29, inst_self_values_9 0.19, inst_self_values_2 0.16 |
| inst_fromdata | 30 | -1.813 ± 0.010 | +0.007 ± 0.001 | -1.820 (inst_fromdata_09) | inst_fromdata_09 0.26, inst_fromdata_08 0.25, inst_fromdata_03 0.24, inst_fromdata_19 0.14, inst_fromdata_15 0.07, inst_fromdata_17 0.04 |
| fewshot_generic | 18 | -1.576 ± 0.009 | +0.244 ± 0.003 | -1.608 (fsg_k16_d2) | fsg_k16_d1 0.29, fsg_k16_d2 0.28, fsg_k16_d5 0.19, fsg_k16_d0 0.06, fsg_k16_d4 0.06, fsg_k1_d3 0.03, fsg_k16_d3 0.03, fsg_k4_d3 0.03 |
| fewshot_hhh | 12 | -1.578 ± 0.009 | +0.242 ± 0.003 | -1.606 (fsh_k16_d2) | fsh_k16_d2 0.31, fsh_k16_d1 0.26, fsh_k16_d5 0.19, fsh_k16_d0 0.07, fsh_k16_d4 0.06, fsh_k4_d3 0.04, fsh_k16_d3 0.04, fsh_k4_d5 0.02 |
| legacy86+fewshot | 116 | -1.576 ± 0.009 | +0.244 ± 0.003 | -1.606 (fsh_k16_d2) | fsg_k16_d1 0.27, fsh_k16_d2 0.20, fsg_k16_d5 0.17, fsg_k16_d2 0.09, fsh_k16_d0 0.07, fsg_k16_d4 0.04, fsh_k4_d3 0.04, fsh_k16_d3 0.03 |
| ALL descriptions | 336 | -1.786 ± 0.009 | +0.034 ± 0.001 | -1.795 (olmo_all) | olmo_all 0.38, inst_self_values_7 0.21, inst_self_values_4 0.15, inst_self_values_9 0.08, inst_self_values_2 0.04, inst_fromdata_03 0.03, inst_fromdata_19 0.03, olmo_identity 0.02 |
| ALL incl. few-shot | 366 | -1.576 ± 0.009 | +0.244 ± 0.003 | -1.606 (fsh_k16_d2) | fsg_k16_d1 0.27, fsh_k16_d2 0.20, fsg_k16_d5 0.17, fsg_k16_d2 0.08, fsh_k16_d0 0.07, fsg_k16_d4 0.04, fsh_k4_d3 0.04, fsh_k16_d3 0.03 |
