# Basis comparison on instruct_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *Instruct itself (exact)* | | -0.898 ± 0.009 | | | |
| *base generic header* | | -1.758 ± 0.011 | | | |
| legacy86 | 86 | -1.723 ± 0.011 | +0.000 ± 0.000 | -1.731 (e14) | e14 0.39, hhh 0.38, fred 0.10, neutral 0.06, e49 0.04, e69 0.04 |
| axis_roles | 24 | -1.731 ± 0.011 | -0.009 ± 0.001 | -1.733 (ax_dilettante) | ax_dilettante 0.69, ax_teenager 0.16, ax_cosmopolitan 0.07, ax_influencer 0.03, ax_amnesiac 0.02, ax_wind 0.02 |
| axis_traits | 23 | -1.733 ± 0.011 | -0.011 ± 0.001 | -1.737 (axt_introverted) | axt_introverted 0.58, axt_casual 0.12, axt_pessimistic 0.10, axt_secular 0.08, axt_benevolent 0.07, axt_blunt 0.04 |
| inst_self | 40 | -1.691 ± 0.012 | +0.032 ± 0.002 | -1.700 (inst_self_values_7) | inst_self_values_7 0.40, inst_self_describe_0 0.31, inst_self_story_2 0.13, inst_self_advice_2 0.07, inst_self_advice_0 0.04, inst_self_story_7 0.03, inst_self_story_0 0.01, inst_self_values_4 0.01 |
| inst_fromdata | 30 | -1.674 ± 0.011 | +0.049 ± 0.002 | -1.678 (inst_fromdata_17) | inst_fromdata_17 0.55, inst_fromdata_04 0.14, inst_fromdata_07 0.11, inst_fromdata_01 0.07, inst_fromdata_19 0.04, inst_fromdata_15 0.03, inst_fromdata_29 0.03, inst_fromdata_03 0.03 |
| legacy+axis | 133 | -1.720 ± 0.011 | +0.002 ± 0.000 | -1.731 (e14) | hhh 0.34, ax_dilettante 0.24, e14 0.20, fred 0.08, axt_introverted 0.06, e69 0.03, axt_pessimistic 0.03, e49 0.01 |
| legacy+inst | 156 | -1.672 ± 0.011 | +0.050 ± 0.002 | -1.678 (inst_fromdata_17) | inst_fromdata_17 0.48, inst_fromdata_04 0.13, inst_self_values_7 0.10, inst_fromdata_07 0.09, inst_fromdata_01 0.06, inst_self_describe_0 0.04, inst_self_story_2 0.04, inst_fromdata_03 0.01 |
