# Basis comparison on instruct_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *Instruct itself (exact)* | | -0.898 ± 0.009 | | | |
| *base generic header* | | -1.758 ± 0.011 | | | |
| legacy86 | 86 | -1.723 ± 0.011 | +0.000 ± 0.000 | -1.731 (e14) | e14 0.39, hhh 0.38, fred 0.10, neutral 0.06, e49 0.04, e69 0.04 |
| inst_fromdata | 30 | -1.674 ± 0.011 | +0.049 ± 0.002 | -1.678 (inst_fromdata_17) | inst_fromdata_17 0.55, inst_fromdata_04 0.14, inst_fromdata_07 0.11, inst_fromdata_01 0.07, inst_fromdata_19 0.04, inst_fromdata_15 0.03, inst_fromdata_29 0.03, inst_fromdata_03 0.03 |
| inst_from_softprefix | 30 | -1.667 ± 0.011 | +0.055 ± 0.002 | -1.681 (inst_from_softprefix_23) | inst_from_softprefix_23 0.35, inst_from_softprefix_26 0.20, inst_from_softprefix_21 0.13, inst_from_softprefix_16 0.11, inst_from_softprefix_07 0.08, inst_from_softprefix_17 0.07, inst_from_softprefix_00 0.06, inst_from_softprefix_08 0.01 |
