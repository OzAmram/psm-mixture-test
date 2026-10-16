# Basis comparison on instruct_T1.2_v1 (1199 hold-out answers, 1199 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *tempered Instruct itself (T=1.2, exact)* | | -1.483 ± 0.022 | | | |
| *base generic header* | | -2.342 ± 0.025 | | | |
| legacy86 | 86 | -2.307 ± 0.025 | +0.000 ± 0.000 | -2.318 (e14) | e14 0.35, hhh 0.32, neutral 0.13, fred 0.08, e49 0.08, e69 0.04 |
