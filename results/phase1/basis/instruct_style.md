# Basis comparison on instruct_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *Instruct itself (exact)* | | -0.898 ± 0.009 | | | |
| *base generic header* | | -1.758 ± 0.011 | | | |
| legacy86 | 86 | -1.723 ± 0.011 | +0.000 ± 0.000 | -1.731 (e14) | e14 0.39, hhh 0.38, fred 0.10, neutral 0.06, e49 0.04, e69 0.04 |
| style | 5 | -1.690 ± 0.012 | +0.033 ± 0.002 | -1.697 (style_all_hhh) | style_all_hhh 0.69, style_typo 0.29, style_all_e14 0.01 |
| legacy+style | 91 | -1.688 ± 0.012 | +0.035 ± 0.002 | -1.697 (style_all_hhh) | style_all_hhh 0.62, style_typo 0.17, e14 0.15, fred 0.03, hhh 0.01, e69 0.01, neutral 0.01 |
| legacy+olmo+style | 98 | -1.687 ± 0.012 | +0.035 ± 0.002 | -1.697 (style_all_hhh) | style_all_hhh 0.58, style_typo 0.16, e14 0.12, olmo_all_hhh 0.11, fred 0.02, e69 0.01 |
