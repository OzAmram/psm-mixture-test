# Basis comparison on instruct_T1.1_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *tempered Instruct itself (T=1.1, exact)* | | -1.149 ± 0.016 | | | |
| *base generic header* | | -1.992 ± 0.019 | | | |
| legacy86 | 86 | -1.956 ± 0.019 | +0.000 ± 0.000 | -1.965 (e14) | hhh 0.35, e14 0.33, neutral 0.11, fred 0.10, e49 0.06, e69 0.05 |
