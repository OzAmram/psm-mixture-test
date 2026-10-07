# Mixture fit as log-likelihoods (scripts/phase1_loglik_table.py)


**Base assistant, 2x2 (hold-out questions; log P / token of the sampled answers under the fitted mixture, ±1σ question bootstrap; gap = sampler − mixture = KL per token)**

| header, scored under | sampler itself (generic header) | six hand-written personas | all 86 personas |
|---|---|---|---|
| plain header: fitted mixture | -1.601 ± 0.013 | -1.632 ± 0.014 (gap 0.031 ± 0.002) | -1.613 ± 0.014 (gap 0.011 ± 0.001) |
| plain header: best single persona (chosen on the fit half) | | -1.634 ± 0.014 (gap 0.032 ± 0.002; hhh) | -1.629 ± 0.014 (gap 0.028 ± 0.002; e73) |
| header + shared casual clause: fitted mixture | -1.589 ± 0.014 | -1.603 ± 0.014 (gap 0.014 ± 0.001) | -1.594 ± 0.014 (gap 0.006 ± 0.001) |
| header + shared casual clause: best single persona (chosen on the fit half) | | -1.604 ± 0.014 (gap 0.016 ± 0.001; neutral) | -1.604 ± 0.014 (gap 0.016 ± 0.001; neutral) |

Hold-out half of the questions; token-weighted mean log P of the sampled answers. Per token is the primary unit (it removes the different answer lengths of the two sampling models); gap = (sampling model) − (column); the sampling model's own log-likelihood is the ceiling any context or mixture could reach.


**Instruct (chat template, default system prompt)** (1200 answers on 150 questions; per response the sampler scores -40.9 and the mixture -78.4)

| scored under | log P / token ± 1σ (gap to the sampling model) |
|---|---|
| sampling model itself (exact) | -0.898 ± 0.009 (gap 0.000) |
| base generic header | -1.758 ± 0.012 (gap 0.860) |
| base persona mixture (86, weights fitted on the other half) | -1.722 ± 0.012 (gap 0.824) |
| best single base persona chosen on the fit half (e14) | -1.731 ± 0.012 (gap 0.832) |

mixture − best single persona, same answers (paired question bootstrap): +0.0083 nats/token, 95% [+0.0066, +0.0099]

top fitted weights: e14 0.39, hhh 0.39, fred 0.10, neutral 0.06, e49 0.04, e69 0.02

**base assistant control (generic header, same suffix)** (1167 answers on 150 questions; per response the sampler scores -79.3 and the mixture -79.6)

| scored under | log P / token ± 1σ (gap to the sampling model) |
|---|---|
| base generic header | -1.612 ± 0.012 (gap 0.000) |
| base persona mixture (86, weights fitted on the other half) | -1.618 ± 0.012 (gap 0.006) |
| best single base persona chosen on the fit half (neutral) | -1.626 ± 0.013 (gap 0.014) |

mixture − best single persona, same answers (paired question bootstrap): +0.0080 nats/token, 95% [+0.0068, +0.0091]

top fitted weights: neutral 0.22, e03 0.11, e58 0.09, e14 0.09, e73 0.08, e48 0.06
