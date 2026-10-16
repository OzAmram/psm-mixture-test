# Basis comparison on instruct_unknown_casual_v1 (1200 hold-out answers, 1200 fit answers; log P / token, ±1σ question bootstrap)

| basis | K | hold-out mixture | Δ vs legacy86 (paired) | best single (fit-selected) | top weights |
|---|---|---|---|---|---|
| *Instruct itself (exact)* | | -0.898 ± 0.009 | | | |
| *base generic header* | | -1.758 ± 0.011 | | | |
| legacy86 | 86 | -1.723 ± 0.011 | +0.000 ± 0.000 | -1.731 (e14) | e14 0.39, hhh 0.38, fred 0.10, neutral 0.06, e49 0.04, e69 0.04 |
| persona_big5 | 10 | -1.732 ± 0.011 | -0.009 ± 0.002 | -1.737 (pb5_C-) | pb5_C- 0.46, pb5_A+ 0.16, pb5_N+ 0.14, pb5_O- 0.11, pb5_A- 0.10, pb5_N- 0.04 |
| values | 15 | -1.753 ± 0.012 | -0.031 ± 0.002 | -1.759 (sv_universalism) | sv_universalism 0.41, mf_care 0.26, sv_self-direction 0.16, sv_achievement 0.15, mf_fairness 0.03 |
| olmo_spec | 7 | -1.724 ± 0.011 | -0.002 ± 0.001 | -1.725 (olmo_all_hhh) | olmo_all_hhh 0.84, olmo_identity 0.13, olmo_ultrafeedback 0.02, olmo_judge 0.01 |
| legacy+olmo | 93 | -1.717 ± 0.011 | +0.006 ± 0.001 | -1.725 (olmo_all_hhh) | olmo_all_hhh 0.54, e14 0.33, fred 0.08, e49 0.02, e69 0.02 |
