# Subliminal-trait detection with elicited-persona likelihood ratios (2026-09-26)

Question: can a likelihood ratio under an elicited persona detect a hidden trait in outputs that do not mention it?
Benign instance following Cloud et al. 2025 (Subliminal Learning), as a precursor to detecting alignment faking.

Setup: teachers = Olmo-3-7B-Instruct with the paper's system prompt for owls / dolphins / none. Data: the paper's
number-sequence prompt and filter rule (2,110-2,475 sequences per teacher); short free-text answers to 300 dilemma
questions with a regex judge removing any animal/nature word (622 / 988 / 1,189 kept). Detectors: OLMo 3 *base* with a
transcript header "...the assistant loves owls..." vs neutral (user's proposal), the teacher itself with vs without the
system prompt (oracle), a dolphin header (specificity), unigram naive Bayes and length baselines; leave-one-out
exemplars k = 0 / 4 / 16 from the same teacher. Metric AUROC. Teachers verified with the paper's favorite-animal question
(owl 100%, dolphin 98%, control 2% / 12%).

| owl vs control, k=0 | numbers | text |
|---|---|---|
| base persona header | 0.55 | 0.88 |
| instruct oracle | 0.89 | 1.00 |
| unigram naive Bayes | 0.62 | - |
| length only | 0.75 | 0.69 |

| trait-specific (owl header vs dolphin header), owl vs dolphin teacher, k=0 | numbers | text |
|---|---|---|
| base persona header | 0.60 | 0.61 |
| instruct oracle | 0.71 | 0.93 |

Findings:
1. The large AUROCs detect *prompted-ness*, not owls: the dolphin header detects the owl teacher almost as well as the owl
   header (text 0.85 vs 0.88). The system prompt's "imbue your answers with your love" makes the teacher speak in a
   first-person, affect-laden voice (first-person "I" in 56% of owl answers vs 4% control), and any animal-lover header
   matches that voice.
2. Trait-specific signal exists but is modest for a base-model header (0.61) and strong only for the teacher itself (0.93).
3. In numbers the base-model header reads nothing (0.55-0.60); unigram counts (0.62) and sequence length (0.75) beat it.
   Subliminal number statistics are model-specific, as the paper reports; OLMo 3 base is not the teacher's weights.
4. Leave-one-out exemplars from the suspect destroy the signal (base/text 0.88 -> 0.62 -> 0.59 for k = 0/4/16): the
   suspect's own outputs override a one-sentence header (same effect as the concealed framing in 0.10). Header-only is
   the operating point.

Implication for alignment-faking detection: feasible to detect "carries a hidden persona instruction" from filtered text
with a base-model header; identifying which trait is much harder; keep the suspect's outputs out of the context.
Next: (a) a same-form unrelated-trait control teacher ("You love trains") to remove the prompted-ness confound;
(b) the paper's actual setting, an SFT student trained on owl numbers with no system prompt at inference, where the
voice confound vanishes and only subliminal statistics remain; (c) then the alignment-faking persona.

Files: notebooks/3.1_subliminal_detection.ipynb; results/subliminal/{owl,dolphin,control}/{numbers,text}.jsonl, meta.json;
scores_{base,instruct}_{numbers,text}.jsonl; 3.1_roc.png; 3.1_aurocs.json. Scripts: subliminal_generate.py,
subliminal_score.py, subliminal_stage2.sh.
