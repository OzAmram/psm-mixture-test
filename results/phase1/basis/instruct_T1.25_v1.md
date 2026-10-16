# Basis comparison on instruct_T1.25_v1 (1199 hold-out answers, 1199 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *tempered Instruct itself (T=1.25, exact)* | | -1.832 ± 0.041 | | | |
| *base generic header* | | -2.704 ± 0.045 | | | |
| legacy86 | 86 | -2.671 ± 0.045 | +0.000 ± 0.000 | -2.681 (e14) | e14 0.29, hhh 0.28, neutral 0.17, fred 0.10, e49 0.08, e69 0.06, e64 0.01, e62 0.01 |
