# Basis comparison on instruct_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *Instruct itself (exact)* | | -0.898 ± 0.009 | | | |
| *base generic header* | | -1.758 ± 0.011 | | | |
| legacy86 | 86 | -1.723 ± 0.011 | +0.000 ± 0.000 | -1.731 (e14) | e14 0.39, hhh 0.38, fred 0.10, neutral 0.06, e49 0.04, e69 0.04 |
| legacy86+fewshot | 116 | -1.598 ± 0.011 | +0.124 ± 0.005 | -1.634 (fsg_k16_d1) | fsg_k16_d1 0.29, fsh_k16_d3 0.11, fsh_k4_d5 0.10, fsg_k1_d2 0.08, fsh_k16_d4 0.07, fsg_k1_d4 0.06, fsg_k16_d2 0.05, fsh_k4_d2 0.04 |
| legacy86+style+inst | 161 | -1.662 ± 0.011 | +0.060 ± 0.002 | -1.678 (inst_fromdata_17) | inst_fromdata_17 0.35, style_all_hhh 0.30, inst_fromdata_04 0.09, inst_fromdata_07 0.09, inst_fromdata_01 0.05, inst_self_story_2 0.03, inst_self_values_7 0.03, inst_self_describe_0 0.01 |
| desc_all | 336 | -1.662 ± 0.011 | +0.060 ± 0.002 | -1.678 (inst_fromdata_17) | inst_fromdata_17 0.35, style_all_hhh 0.30, inst_fromdata_04 0.09, inst_fromdata_07 0.09, inst_fromdata_01 0.05, inst_self_story_2 0.03, inst_self_values_7 0.03, inst_self_describe_0 0.01 |
| lit_only | 147 | -1.715 ± 0.011 | +0.008 ± 0.001 | -1.725 (olmo_all_hhh) | olmo_all_hhh 0.52, ax_dilettante 0.19, pb5_C- 0.10, axt_introverted 0.07, b5p_O-C-E+A+N+ 0.04, b5p_O+C+E+A+N+ 0.03, ax_teenager 0.02, axt_pessimistic 0.01 |
| all | 436 | -1.595 ± 0.011 | +0.127 ± 0.005 | -1.634 (fsg_k16_d1) | fsg_k16_d1 0.27, fsh_k16_d3 0.10, fsh_k4_d5 0.09, style_all_hhh 0.07, fsh_k16_d4 0.06, fsg_k16_d2 0.05, fsg_k1_d4 0.05, fsg_k1_d2 0.05 |
| olmo_spec | 7 | -1.724 ± 0.011 | -0.002 ± 0.001 | -1.725 (olmo_all_hhh) | olmo_all_hhh 0.84, olmo_identity 0.13, olmo_ultrafeedback 0.02, olmo_judge 0.01 |
| bigfive | 42 | -1.729 ± 0.011 | -0.007 ± 0.001 | -1.734 (b5p_O+C+E+A+N+) | b5p_O+C+E+A+N+ 0.33, b5_O+ 0.27, b5_C+ 0.12, b5p_O-C-E-A+N+ 0.10, b5_N+ 0.05, b5p_O-C-E+A+N+ 0.05, b5p_O+C-E-A+N+ 0.04, b5_A+ 0.03 |
| values | 15 | -1.753 ± 0.012 | -0.031 ± 0.002 | -1.759 (sv_universalism) | sv_universalism 0.41, mf_care 0.26, sv_self-direction 0.16, sv_achievement 0.15, mf_fairness 0.03 |
| axis_roles | 24 | -1.731 ± 0.011 | -0.009 ± 0.001 | -1.733 (ax_dilettante) | ax_dilettante 0.69, ax_teenager 0.16, ax_cosmopolitan 0.07, ax_influencer 0.03, ax_amnesiac 0.02, ax_wind 0.02 |
| axis_traits | 23 | -1.733 ± 0.011 | -0.011 ± 0.001 | -1.737 (axt_introverted) | axt_introverted 0.58, axt_casual 0.12, axt_pessimistic 0.10, axt_secular 0.08, axt_benevolent 0.07, axt_blunt 0.04 |
| base_fromdata | 28 | -1.725 ± 0.012 | -0.003 ± 0.001 | -1.731 (base_fromdata_13) | base_fromdata_13 0.53, base_fromdata_10 0.44, base_fromdata_04 0.03, base_fromdata_17 0.01 |
