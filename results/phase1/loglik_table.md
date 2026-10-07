# Mixture fit as log-likelihoods (scripts/phase1_loglik_table.py)


**Base assistant, 2x2 (hold-out questions; log P / token of the sampled answers under the fitted mixture, ±1σ question bootstrap; gap = sampler − mixture = KL per token)**

| header \ basis | six hand-written personas | all 86 personas |
|---|---|---|
| plain header (sampler itself -1.601) | -1.632 ± 0.014 (gap 0.031 ± 0.002) | -1.613 ± 0.014 (gap 0.011 ± 0.001) |
| header + shared casual clause (sampler itself -1.589) | -1.603 ± 0.014 (gap 0.014 ± 0.001) | -1.594 ± 0.014 (gap 0.006 ± 0.001) |

Hold-out half of the questions; token-weighted mean log P of the sampled answers. Per token is the primary unit (it removes the different answer lengths of the two sampling models); gap = (sampling model) − (column); the sampling model's own log-likelihood is the ceiling any context or mixture could reach.


**Instruct (chat template, default system prompt)** (1200 answers on 150 questions)

| scored under | log P / token (±1σ, question bootstrap) | gap to sampling model, nats / token | log P / response (for scale) |
|---|---|---|---|
| sampling model itself (exact) | -0.898 ± 0.009 | 0.000 | -40.9 |
| base generic header | -1.758 ± 0.012 | 0.860 | -80.0 |
| base persona mixture (86, weights fitted on the other half) | -1.722 ± 0.012 | 0.824 | -78.4 |
| best single base persona chosen on the fit half (e14) | -1.731 ± 0.012 | 0.832 | -78.7 |

mixture − best single persona, same answers (paired question bootstrap): +0.0083 nats/token, 95% [+0.0066, +0.0099]

top fitted weights: e14 0.39, hhh 0.39, fred 0.10, neutral 0.06, e49 0.04, e69 0.02

**base assistant control (generic header, same suffix)** (1167 answers on 150 questions)

| scored under | log P / token (±1σ, question bootstrap) | gap to sampling model, nats / token | log P / response (for scale) |
|---|---|---|---|
| base generic header | -1.612 ± 0.012 | 0.000 | -79.3 |
| base persona mixture (86, weights fitted on the other half) | -1.618 ± 0.012 | 0.006 | -79.6 |
| best single base persona chosen on the fit half (neutral) | -1.626 ± 0.013 | 0.014 | -80.0 |

mixture − best single persona, same answers (paired question bootstrap): +0.0080 nats/token, 95% [+0.0068, +0.0091]

top fitted weights: neutral 0.22, e03 0.11, e58 0.09, e14 0.09, e73 0.08, e48 0.06
