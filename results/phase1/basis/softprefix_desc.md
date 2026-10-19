# Basis comparison on softprefix_unknown_casual_v1 (1198 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *sampling model itself (softprefix_L128, exact)* | | -1.439 ± 0.015 | | | |
| *base generic header* | | -1.881 ± 0.016 | | | |
| legacy86 | 86 | -1.841 ± 0.016 | +0.000 ± 0.000 | -1.847 (hhh) | hhh 0.53, e14 0.38, e49 0.04, e69 0.03, fred 0.03 |
| inst_fromdata | 30 | -1.800 ± 0.016 | +0.042 ± 0.002 | -1.804 (inst_fromdata_17) | inst_fromdata_17 0.50, inst_fromdata_04 0.21, inst_fromdata_19 0.10, inst_fromdata_03 0.06, inst_fromdata_07 0.04, inst_fromdata_29 0.03, inst_fromdata_01 0.03, inst_fromdata_12 0.02 |
| inst_from_softprefix | 30 | -1.795 ± 0.016 | +0.047 ± 0.002 | -1.804 (inst_from_softprefix_19) | inst_from_softprefix_19 0.23, inst_from_softprefix_07 0.16, inst_from_softprefix_23 0.14, inst_from_softprefix_26 0.14, inst_from_softprefix_21 0.12, inst_from_softprefix_29 0.10, inst_from_softprefix_16 0.06, inst_from_softprefix_17 0.04 |
| fewshot_generic | 18 | -1.689 ± 0.015 | +0.152 ± 0.003 | -1.719 (fsg_k16_d2) | fsg_k16_d1 0.19, fsg_k16_d3 0.17, fsg_k16_d2 0.16, fsg_k16_d0 0.10, fsg_k4_d3 0.08, fsg_k16_d5 0.07, fsg_k4_d2 0.06, fsg_k1_d4 0.04 |
