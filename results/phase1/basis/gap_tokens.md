# Token-level gap, Instruct answers (hold-out, 1147 answers, 52502 tokens)


**base header: generic**: gap 0.855 nats/token (Instruct -0.896, base -1.752); top 5% of tokens carry 28% of the gap, top 20% carry 70%; base assigns the higher probability on 15% of tokens; tokens Instruct gives p > 0.9: 31% of tokens, 15% of the gap

| tokens | n | gap / token | share of gap |
|---|---|---|---|
| first token | 1147 | 2.230 | 6% |
| tokens 2-5 | 4588 | 1.063 | 11% |
| tokens 6+ | 46767 | 0.801 | 83% |
| apostrophe/contraction | 2656 | 0.461 | 3% |
| dash | 585 | 4.969 | 6% |
| punctuation | 4906 | 0.483 | 5% |
| whitespace | 7 | 1.250 | 0% |
| word | 44348 | 0.866 | 86% |

**base header: hhh**: gap 0.830 nats/token (Instruct -0.896, base -1.727); top 5% of tokens carry 29% of the gap, top 20% carry 71%; base assigns the higher probability on 15% of tokens; tokens Instruct gives p > 0.9: 31% of tokens, 15% of the gap

| tokens | n | gap / token | share of gap |
|---|---|---|---|
| first token | 1147 | 2.170 | 6% |
| tokens 2-5 | 4588 | 1.115 | 12% |
| tokens 6+ | 46767 | 0.770 | 83% |
| apostrophe/contraction | 2656 | 0.604 | 4% |
| dash | 585 | 4.639 | 6% |
| punctuation | 4906 | 0.431 | 5% |
| whitespace | 7 | 1.225 | 0% |
| word | 44348 | 0.838 | 85% |

**base header: e14**: gap 0.828 nats/token (Instruct -0.896, base -1.725); top 5% of tokens carry 29% of the gap, top 20% carry 71%; base assigns the higher probability on 16% of tokens; tokens Instruct gives p > 0.9: 31% of tokens, 14% of the gap

| tokens | n | gap / token | share of gap |
|---|---|---|---|
| first token | 1147 | 1.988 | 5% |
| tokens 2-5 | 4588 | 1.023 | 11% |
| tokens 6+ | 46767 | 0.781 | 84% |
| apostrophe/contraction | 2656 | 0.469 | 3% |
| dash | 585 | 4.949 | 7% |
| punctuation | 4906 | 0.420 | 5% |
| whitespace | 7 | 1.467 | 0% |
| word | 44348 | 0.840 | 86% |

Token strings carrying the most total gap (HHH header): '—' 2637 (565×), ' it' 1186 (2364×), ' just' 944 (567×), '.' 811 (2074×), ' you' 772 (1695×), ',' 761 (2260×), 'Honestly' 700 (181×), ' the' 602 (1358×), '’s' 593 (904×), ' or' 589 (607×), '—it' 517 (113×), ' and' 512 (869×), ' a' 474 (898×), ' probably' 472 (261×), ' if' 459 (734×), 'Yeah' 449 (123×), '’t' 405 (466×), ' really' 365 (201×), 'Probably' 361 (63×), ' could' 356 (475×), ' in' 345 (388×), '’re' 337 (294×), ' not' 330 (457×), ' don' 320 (230×), ' way' 319 (216×)
