# Phrasing sensitivity and semantic composition

AUROC per answer / k = 10 / k = 30, ±1σ question bootstrap (100 replicates). Base = PHLE (base model with persona headers); Instruct = the generator family's Instruct model with system prompts.


## Olmo, numbers


**owl vs no prompt: original answers, scored under rephrased headers / prompts**

| scorer | original | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE (header: original / rephrased / fascination not love / minimal) | 0.55±0.01 / 0.66±0.04 / 0.74±0.05 | 0.70±0.01 / 0.94±0.02 / 1.00±0.00 | 0.71±0.01 / 0.95±0.01 / 1.00±0.00 | 0.56±0.01 / 0.68±0.04 / 0.80±0.05 |
| Instruct (system prompt: original / rephrased / fascination not love / minimal) | 0.97±0.00 / 1.00±0.00 / 1.00±0.00 | 0.94±0.01 / 1.00±0.00 / 1.00±0.00 | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.87±0.01 / 1.00±0.00 / 1.00±0.00 |

**owl vs no prompt: answers generated under rephrased system prompts**

| scorer | original prompt | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE, original header | 0.55±0.01 / 0.66±0.04 / 0.74±0.05 | 0.55±0.01 / 0.61±0.04 / 0.72±0.06 | 0.54±0.01 / 0.65±0.04 / 0.76±0.05 | 0.51±0.01 / 0.53±0.04 / 0.57±0.07 |
| Instruct, original prompt | 0.97±0.00 / 1.00±0.00 / 1.00±0.00 | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.75±0.01 / 0.98±0.01 / 1.00±0.00 | 0.78±0.01 / 0.99±0.00 / 1.00±0.00 |
| Instruct, the prompt that generated them | 0.97±0.00 / 1.00±0.00 / 1.00±0.00 | 0.96±0.00 / 1.00±0.00 / 1.00±0.00 | 0.86±0.01 / 1.00±0.00 / 1.00±0.00 | 0.89±0.01 / 1.00±0.00 / 1.00±0.00 |

**owl vs trains: original answers, scored under rephrased headers / prompts**

| scorer | original | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE (header: original / rephrased / fascination not love / minimal) | 0.61±0.01 / 0.78±0.04 / 0.91±0.03 | 0.60±0.01 / 0.78±0.03 / 0.91±0.03 | 0.59±0.01 / 0.77±0.04 / 0.91±0.04 | 0.58±0.01 / 0.72±0.04 / 0.87±0.04 |
| Instruct (system prompt: original / rephrased / fascination not love / minimal) | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.79±0.01 / 0.99±0.00 / 1.00±0.00 | 0.68±0.01 / 0.93±0.02 / 0.99±0.00 | 0.68±0.01 / 0.89±0.03 / 0.98±0.02 |

**owl vs trains: answers generated under rephrased system prompts**

| scorer | original prompt | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE, original header | 0.61±0.01 / 0.78±0.04 / 0.91±0.03 | 0.62±0.01 / 0.85±0.03 / 0.97±0.02 | 0.55±0.01 / 0.65±0.04 / 0.77±0.06 | 0.52±0.01 / 0.55±0.04 / 0.61±0.07 |
| Instruct, original prompt | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.74±0.01 / 0.98±0.01 / 1.00±0.00 | 0.59±0.01 / 0.76±0.04 / 0.89±0.04 | 0.59±0.01 / 0.77±0.03 / 0.89±0.03 |
| Instruct, the prompt that generated them | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.83±0.01 / 1.00±0.00 / 1.00±0.00 | 0.69±0.01 / 0.93±0.02 / 1.00±0.00 | 0.73±0.01 / 0.98±0.01 / 1.00±0.00 |

**composition (numbers): teacher pair scored under a specific vs a broader header (each minus trains or no-prompt)**

| teachers | specific header | birds | animals | dogs |
|---|---|---|---|---|
| owl vs trains | 0.59±0.01 / 0.78±0.04 / 0.91±0.04 | 0.59±0.01 / 0.74±0.05 / 0.85±0.06 | 0.56±0.01 / 0.63±0.04 / 0.73±0.06 | 0.56±0.01 / 0.64±0.04 / 0.72±0.06 |
| eagle vs trains | 0.61±0.01 / 0.80±0.03 / 0.93±0.03 | 0.57±0.01 / 0.67±0.04 / 0.78±0.05 | 0.54±0.02 / 0.61±0.05 / 0.68±0.08 | 0.55±0.01 / 0.63±0.05 / 0.71±0.08 |
| owl vs no prompt | 0.54±0.01 / 0.68±0.04 / 0.76±0.06 | 0.49±0.01 / 0.53±0.04 / 0.58±0.07 | 0.46±0.01 / 0.47±0.04 / 0.42±0.07 | 0.47±0.01 / 0.46±0.04 / 0.44±0.08 |
| eagle vs no prompt | 0.52±0.02 / 0.65±0.04 / 0.71±0.07 | 0.49±0.02 / 0.52±0.04 / 0.54±0.07 | 0.46±0.02 / 0.43±0.05 / 0.38±0.07 | 0.47±0.01 / 0.45±0.05 / 0.43±0.08 |

## Olmo, text


**owl vs no prompt: original answers, scored under rephrased headers / prompts**

| scorer | original | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE (header: original / rephrased / fascination not love / minimal) | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.89±0.01 / 1.00±0.00 / 1.00±0.00 | 0.84±0.01 / 1.00±0.00 / 1.00±0.00 | 0.82±0.01 / 1.00±0.00 / 1.00±0.00 |
| Instruct (system prompt: original / rephrased / fascination not love / minimal) | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.94±0.01 / 1.00±0.00 / 1.00±0.00 | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 |

**owl vs no prompt: answers generated under rephrased system prompts**

| scorer | original prompt | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE, original header | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.62±0.01 / 0.82±0.02 / 0.95±0.02 | 0.55±0.01 / 0.62±0.03 / 0.73±0.05 | 0.54±0.01 / 0.65±0.04 / 0.74±0.06 |
| Instruct, original prompt | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 0.84±0.01 / 1.00±0.00 / 1.00±0.00 | 0.65±0.01 / 0.89±0.02 / 0.98±0.01 | 0.65±0.01 / 0.92±0.02 / 0.99±0.00 |
| Instruct, the prompt that generated them | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 0.98±0.00 / 1.00±0.00 / 1.00±0.00 | 0.92±0.01 / 1.00±0.00 / 1.00±0.00 | 0.94±0.00 / 1.00±0.00 / 1.00±0.00 |

**owl vs trains: original answers, scored under rephrased headers / prompts**

| scorer | original | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE (header: original / rephrased / fascination not love / minimal) | 0.61±0.01 / 0.84±0.03 / 0.96±0.02 | 0.56±0.02 / 0.74±0.05 / 0.88±0.05 | 0.59±0.02 / 0.79±0.05 / 0.91±0.05 | 0.60±0.02 / 0.80±0.04 / 0.96±0.03 |
| Instruct (system prompt: original / rephrased / fascination not love / minimal) | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.70±0.02 / 0.97±0.01 / 1.00±0.00 | 0.65±0.02 / 0.89±0.03 / 0.98±0.01 | 0.65±0.02 / 0.91±0.03 / 0.99±0.01 |

**owl vs trains: answers generated under rephrased system prompts**

| scorer | original prompt | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE, original header | 0.61±0.01 / 0.84±0.03 / 0.96±0.02 | 0.57±0.01 / 0.69±0.03 / 0.84±0.04 | 0.52±0.01 / 0.55±0.04 / 0.59±0.06 | 0.52±0.01 / 0.57±0.04 / 0.61±0.07 |
| Instruct, original prompt | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.65±0.01 / 0.89±0.02 / 0.98±0.01 | 0.57±0.01 / 0.69±0.03 / 0.81±0.04 | 0.59±0.01 / 0.76±0.03 / 0.91±0.03 |
| Instruct, the prompt that generated them | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.84±0.01 / 1.00±0.00 / 1.00±0.00 | 0.76±0.01 / 0.99±0.00 / 1.00±0.00 | 0.83±0.01 / 1.00±0.00 / 1.00±0.00 |

**composition (text): teacher pair scored under a specific vs a broader header (each minus trains or no-prompt)**

| teachers | specific header | birds | animals | dogs |
|---|---|---|---|---|
| owl vs trains | 0.61±0.01 / 0.83±0.03 / 0.95±0.02 | 0.64±0.02 / 0.88±0.03 / 0.99±0.01 | 0.64±0.02 / 0.88±0.03 / 0.99±0.01 | 0.61±0.02 / 0.85±0.04 / 0.96±0.02 |
| eagle vs trains | 0.62±0.02 / 0.85±0.04 / 0.96±0.02 | 0.61±0.01 / 0.82±0.03 / 0.95±0.02 | 0.60±0.01 / 0.79±0.04 / 0.94±0.03 | 0.61±0.02 / 0.82±0.04 / 0.95±0.03 |
| owl vs no prompt | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.89±0.01 / 1.00±0.00 / 1.00±0.00 | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.86±0.01 / 1.00±0.00 / 1.00±0.00 |
| eagle vs no prompt | 0.87±0.01 / 1.00±0.00 / 1.00±0.00 | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.87±0.01 / 1.00±0.00 / 1.00±0.00 | 0.86±0.01 / 1.00±0.00 / 1.00±0.00 |

## Qwen, numbers


**owl vs no prompt: original answers, scored under rephrased headers / prompts**

| scorer | original | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE (header: original / rephrased / fascination not love / minimal) | 0.63±0.01 / 0.85±0.02 / 0.97±0.01 | 0.65±0.01 / 0.87±0.02 / 0.98±0.01 | 0.48±0.01 / 0.48±0.04 / 0.45±0.07 | 0.62±0.01 / 0.82±0.03 / 0.95±0.02 |
| Instruct (system prompt: original / rephrased / fascination not love / minimal) | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.98±0.00 / 1.00±0.00 / 1.00±0.00 | 0.98±0.00 / 1.00±0.00 / 1.00±0.00 |

**owl vs no prompt: answers generated under rephrased system prompts**

| scorer | original prompt | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE, original header | 0.63±0.01 / 0.85±0.02 / 0.97±0.01 | 0.59±0.01 / 0.75±0.04 / 0.89±0.04 | 0.60±0.01 / 0.79±0.04 / 0.93±0.03 | 0.60±0.01 / 0.78±0.03 / 0.92±0.03 |
| Instruct, original prompt | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.98±0.00 / 1.00±0.00 / 1.00±0.00 | 0.94±0.01 / 1.00±0.00 / 1.00±0.00 | 0.91±0.01 / 1.00±0.00 / 1.00±0.00 |
| Instruct, the prompt that generated them | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.98±0.00 / 1.00±0.00 / 1.00±0.00 | 0.97±0.00 / 1.00±0.00 / 1.00±0.00 |

**owl vs trains: original answers, scored under rephrased headers / prompts**

| scorer | original | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE (header: original / rephrased / fascination not love / minimal) | 0.59±0.01 / 0.79±0.03 / 0.91±0.03 | 0.62±0.01 / 0.83±0.03 / 0.96±0.02 | 0.61±0.01 / 0.81±0.02 / 0.93±0.02 | 0.58±0.01 / 0.78±0.03 / 0.90±0.03 |
| Instruct (system prompt: original / rephrased / fascination not love / minimal) | 0.82±0.01 / 1.00±0.00 / 1.00±0.00 | 0.73±0.01 / 0.97±0.01 / 1.00±0.00 | 0.70±0.01 / 0.96±0.01 / 1.00±0.00 | 0.72±0.01 / 0.97±0.01 / 1.00±0.00 |

**owl vs trains: answers generated under rephrased system prompts**

| scorer | original prompt | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE, original header | 0.59±0.01 / 0.79±0.03 / 0.91±0.03 | 0.59±0.01 / 0.76±0.04 / 0.89±0.04 | 0.56±0.01 / 0.73±0.04 / 0.86±0.05 | 0.59±0.01 / 0.74±0.04 / 0.88±0.04 |
| Instruct, original prompt | 0.82±0.01 / 1.00±0.00 / 1.00±0.00 | 0.78±0.01 / 0.99±0.00 / 1.00±0.00 | 0.71±0.01 / 0.98±0.01 / 1.00±0.00 | 0.76±0.01 / 0.99±0.00 / 1.00±0.00 |
| Instruct, the prompt that generated them | 0.82±0.01 / 1.00±0.00 / 1.00±0.00 | 0.89±0.01 / 1.00±0.00 / 1.00±0.00 | 0.83±0.01 / 1.00±0.00 / 1.00±0.00 | 0.90±0.01 / 1.00±0.00 / 1.00±0.00 |

**composition (numbers): teacher pair scored under a specific vs a broader header (each minus trains or no-prompt)**

| teachers | specific header | birds | animals | dogs |
|---|---|---|---|---|
| owl vs trains | 0.58±0.01 / 0.73±0.04 / 0.90±0.04 | 0.58±0.01 / 0.75±0.03 / 0.88±0.04 | 0.57±0.01 / 0.71±0.04 / 0.82±0.05 | 0.57±0.01 / 0.72±0.04 / 0.87±0.04 |
| eagle vs trains | 0.58±0.01 / 0.71±0.03 / 0.84±0.04 | 0.56±0.01 / 0.66±0.04 / 0.78±0.06 | 0.57±0.01 / 0.70±0.04 / 0.82±0.05 | 0.55±0.01 / 0.64±0.04 / 0.74±0.05 |
| owl vs no prompt | 0.62±0.01 / 0.81±0.03 / 0.93±0.03 | 0.59±0.01 / 0.76±0.03 / 0.89±0.03 | 0.64±0.01 / 0.86±0.03 / 0.97±0.02 | 0.61±0.01 / 0.76±0.03 / 0.91±0.04 |
| eagle vs no prompt | 0.63±0.01 / 0.81±0.04 / 0.93±0.03 | 0.56±0.01 / 0.64±0.04 / 0.75±0.07 | 0.62±0.01 / 0.78±0.03 / 0.93±0.03 | 0.57±0.01 / 0.66±0.05 / 0.76±0.07 |

## Qwen, text


**owl vs no prompt: original answers, scored under rephrased headers / prompts**

| scorer | original | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE (header: original / rephrased / fascination not love / minimal) | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.96±0.01 / 1.00±0.00 / 1.00±0.00 | 0.89±0.01 / 1.00±0.00 / 1.00±0.00 | 0.87±0.01 / 1.00±0.00 / 1.00±0.00 |
| Instruct (system prompt: original / rephrased / fascination not love / minimal) | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 0.96±0.01 / 1.00±0.00 / 1.00±0.00 | 0.91±0.01 / 1.00±0.00 / 1.00±0.00 |

**owl vs no prompt: answers generated under rephrased system prompts**

| scorer | original prompt | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE, original header | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.96±0.01 / 1.00±0.00 / 1.00±0.00 | 0.61±0.01 / 0.82±0.03 / 0.95±0.02 | 0.61±0.01 / 0.84±0.03 / 0.94±0.02 |
| Instruct, original prompt | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 0.79±0.01 / 0.99±0.00 / 1.00±0.00 | 0.75±0.01 / 0.98±0.01 / 1.00±0.00 |
| Instruct, the prompt that generated them | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 0.98±0.00 / 1.00±0.00 / 1.00±0.00 | 0.98±0.00 / 1.00±0.00 / 1.00±0.00 |

**owl vs trains: original answers, scored under rephrased headers / prompts**

| scorer | original | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE (header: original / rephrased / fascination not love / minimal) | 0.67±0.01 / 0.92±0.02 / 1.00±0.00 | 0.73±0.01 / 0.98±0.01 / 1.00±0.00 | 0.65±0.02 / 0.90±0.03 / 0.99±0.01 | 0.61±0.02 / 0.84±0.03 / 0.97±0.02 |
| Instruct (system prompt: original / rephrased / fascination not love / minimal) | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.78±0.01 / 1.00±0.00 / 1.00±0.00 | 0.70±0.02 / 0.95±0.01 / 1.00±0.00 | 0.70±0.01 / 0.94±0.02 / 1.00±0.00 |

**owl vs trains: answers generated under rephrased system prompts**

| scorer | original prompt | rephrased | fascination not love | minimal |
|---|---|---|---|---|
| base PHLE, original header | 0.67±0.01 / 0.92±0.02 / 1.00±0.00 | 0.63±0.02 / 0.90±0.03 / 0.99±0.01 | 0.59±0.01 / 0.80±0.03 / 0.93±0.03 | 0.59±0.01 / 0.78±0.03 / 0.92±0.03 |
| Instruct, original prompt | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.72±0.02 / 0.98±0.01 / 1.00±0.00 | 0.64±0.01 / 0.89±0.03 / 0.98±0.01 | 0.66±0.01 / 0.89±0.02 / 0.99±0.01 |
| Instruct, the prompt that generated them | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.90±0.01 / 1.00±0.00 / 1.00±0.00 | 0.85±0.01 / 1.00±0.00 / 1.00±0.00 | 0.89±0.01 / 1.00±0.00 / 1.00±0.00 |

**composition (text): teacher pair scored under a specific vs a broader header (each minus trains or no-prompt)**

| teachers | specific header | birds | animals | dogs |
|---|---|---|---|---|
| owl vs trains | 0.67±0.02 / 0.92±0.02 / 1.00±0.00 | 0.71±0.01 / 0.97±0.01 / 1.00±0.00 | 0.71±0.01 / 0.97±0.01 / 1.00±0.00 | 0.64±0.02 / 0.88±0.04 / 0.98±0.02 |
| eagle vs trains | 0.63±0.01 / 0.87±0.02 / 0.98±0.01 | 0.70±0.02 / 0.97±0.01 / 1.00±0.00 | 0.70±0.02 / 0.96±0.01 / 1.00±0.00 | 0.65±0.02 / 0.88±0.03 / 0.98±0.01 |
| owl vs no prompt | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.96±0.01 / 1.00±0.00 / 1.00±0.00 | 0.94±0.01 / 1.00±0.00 / 1.00±0.00 |
| eagle vs no prompt | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.96±0.01 / 1.00±0.00 / 1.00±0.00 | 0.94±0.01 / 1.00±0.00 / 1.00±0.00 |

**composition, Instruct (Olmo, numbers): system prompts, each minus the train-loving prompt or no prompt**

| teachers | specific prompt | birds | animals | dogs |
|---|---|---|---|---|
| owl vs trains | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.74±0.01 / 0.98±0.01 / 1.00±0.00 | 0.73±0.01 / 0.96±0.01 / 1.00±0.00 | 0.73±0.01 / 0.97±0.01 / 1.00±0.00 |
| eagle vs trains | 0.86±0.01 / 1.00±0.00 / 1.00±0.00 | 0.73±0.01 / 0.96±0.01 / 1.00±0.00 | 0.71±0.01 / 0.95±0.01 / 1.00±0.00 | 0.74±0.01 / 0.96±0.01 / 1.00±0.00 |
| owl vs no prompt | 0.96±0.00 / 1.00±0.00 / 1.00±0.00 | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 |
| eagle vs no prompt | 0.97±0.00 / 1.00±0.00 / 1.00±0.00 | 0.95±0.00 / 1.00±0.00 / 1.00±0.00 | 0.94±0.00 / 1.00±0.00 / 1.00±0.00 | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 |

**composition, Instruct (Olmo, text): system prompts, each minus the train-loving prompt or no prompt**

| teachers | specific prompt | birds | animals | dogs |
|---|---|---|---|---|
| owl vs trains | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.84±0.01 / 1.00±0.00 / 1.00±0.00 | 0.78±0.02 / 1.00±0.00 / 1.00±0.00 | 0.87±0.01 / 1.00±0.00 / 1.00±0.00 |
| eagle vs trains | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.84±0.01 / 1.00±0.00 / 1.00±0.00 | 0.78±0.01 / 1.00±0.00 / 1.00±0.00 | 0.85±0.01 / 1.00±0.00 / 1.00±0.00 |
| owl vs no prompt | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 |
| eagle vs no prompt | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 |

**composition, Instruct (Qwen, numbers): system prompts, each minus the train-loving prompt or no prompt**

| teachers | specific prompt | birds | animals | dogs |
|---|---|---|---|---|
| owl vs trains | 0.82±0.01 / 1.00±0.00 / 1.00±0.00 | 0.76±0.01 / 0.99±0.00 / 1.00±0.00 | 0.73±0.01 / 0.97±0.01 / 1.00±0.00 | 0.77±0.01 / 0.99±0.00 / 1.00±0.00 |
| eagle vs trains | 0.82±0.01 / 1.00±0.00 / 1.00±0.00 | 0.74±0.01 / 0.98±0.01 / 1.00±0.00 | 0.71±0.01 / 0.96±0.01 / 1.00±0.00 | 0.75±0.01 / 0.98±0.01 / 1.00±0.00 |
| owl vs no prompt | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 |
| eagle vs no prompt | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 | 0.99±0.00 / 1.00±0.00 / 1.00±0.00 |

**composition, Instruct (Qwen, text): system prompts, each minus the train-loving prompt or no prompt**

| teachers | specific prompt | birds | animals | dogs |
|---|---|---|---|---|
| owl vs trains | 0.95±0.01 / 1.00±0.00 / 1.00±0.00 | 0.90±0.01 / 1.00±0.00 / 1.00±0.00 | 0.84±0.01 / 1.00±0.00 / 1.00±0.00 | 0.86±0.01 / 1.00±0.00 / 1.00±0.00 |
| eagle vs trains | 0.93±0.01 / 1.00±0.00 / 1.00±0.00 | 0.88±0.01 / 1.00±0.00 / 1.00±0.00 | 0.84±0.01 / 1.00±0.00 / 1.00±0.00 | 0.87±0.01 / 1.00±0.00 / 1.00±0.00 |
| owl vs no prompt | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 |
| eagle vs no prompt | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 | 1.00±0.00 / 1.00±0.00 / 1.00±0.00 |
