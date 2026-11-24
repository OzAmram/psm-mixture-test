# Review of `my_writeup_persona_mixture.md`: writing and framing (2026-11-19)

Reviewer: a separate Claude agent (fresh context, not given this session's discussion), asked for feedback on writing and framing for a discussion-based, blog-style post for a LessWrong / Alignment Forum audience, not a quantitative paper. It read the post in full and checked its numbers against `claude_writeup_persona_mixture.md`; it did not edit anything. Line numbers refer to the post as it was on 2026-11-19, before the typo fixes (which changed no line breaks). The typo list in section 6 has since been applied; the rest is untouched, for the author to review.

Note from the session (not the reviewer): the suggested headline in section 1 ("selection describes post-training after SFT, not SFT itself") comes from an interpretation in the Claude write-up, not from independent evidence; SFT scoring Instruct's answers is a different model, not "the base model in some context", so it is a different kind of test from the others.

---

**Overall verdict.** The draft has a real, interesting result and a good conceptual hook (strong vs weak PSM, and the mixture formula). But the argument loses its shape partway through. The TL;DR lists methods instead of stating one claim. The strong-vs-weak section argues the conclusion before any evidence appears. The Results half turns into a run of tables without a clear "so what". Two sections are unfinished placeholders (lines 309–311 "Perhaps we just need more context? ...", line 322 "Applying lenses..."). The key confound, that per-token likelihood mostly measures style rather than character, appears only near the end (line 355). It should come early. I checked the numbers against the companion write-up and they match. Where the problems lie is interpretation and emphasis.

## 1. Thesis
- Line 1 ("Naive Attempts...") and the TL;DR undersell the result and bury it. A reader should get three things in the first five lines. **Question:** can a post-trained model's answers be reproduced by base-model personas? **Answer:** for the base model's own assistant, almost perfectly (0.006 nats/token). For Instruct, no: a 0.82 gap remains, and even a freely optimised prompt leaves about 0.5. **Why it matters:** "selection" is the wrong verb, because post-training changes the simulator itself, not just which character it plays.
- TL;DR line 5 says "Other persona bases don't really help". But the geometric fit with bigger bases closes up to 28%, so the TL;DR contradicts your own later section. Line 6 says "context distillation", while the body says "soft context". Pick one term.
- **Missed sharper framing.** The companion has a result the post leaves out: the SFT checkpoint alone closes about 80% of the base-to-Instruct distance, and a soft context on SFT closes 69% of what remains. So DPO/RL relative to SFT *does* look like conditioning/selection, while base→SFT does not. "Selection describes the late stages of post-training, not SFT" is a crisper, more falsifiable thesis than "PSM is a useful heuristic but there's more going on" (line 32). I'd seriously consider making it the headline of the checkpoint section.

## 2. Structure and flow
- **Strong vs Weak PSM (lines 35–71) is too long, and it comes before any evidence.** Line 67 says "The experiments on base models ... support this view", but the base-model experiments show an *arithmetic* mixture of whole characters fits the base assistant almost perfectly. If anything, that is the strong-PSM-friendly result. The only bridge to weak PSM is arithmetic vs geometric. Cut this section to about 8 lines and state the distinction as a hypothesis the experiments probe. Move the rename aside (line 65) to a footnote. Lines 69–71 (routing after post-training) don't connect to anything later.
- Suggested arc: hook → formula (arithmetic) → base assistant fits → Instruct doesn't → "is it the basis? the temperature? the description format?" (bases, sharpening, soft context as the ceiling) → geometric as "assembly" → checkpoints (where the gap appears) → what it means / limitations.
- **Results-dump spots:** the bases table (lines 220–231) and the geometric table (247–254) cover the same bases. Merge them into one table with columns basis | arithmetic | geometric, and drop "first personas chosen". In the base-assistant table (187–192), keep only the casual-clause column and put the "halves the gap" sentence in prose. The checkpoint table (330–335) duplicates the figure, so keep the figure only.
- Fill or cut the placeholders at 309–311 and 322. The companion's "plateaus at −1.37 whatever the length, learning rate, initialisation" is the one-sentence answer the "Perhaps we just need more context?" line is missing.
- The Discussion's three side-points (376–384) are good blog material, but they read as a separate post. Keep one, probably the LLM-judges point.

## 3. Calibration
- **Biggest issue:** "indicates the PSM is not exhaustive" (line 350) and "Clearly post-training is doing something more complicated" (351). The metric is token-level likelihood. The PSM is a claim about character and behaviour. Your own line 355 notes that em-dashes and register drive the likelihood: in the companion, the em-dash alone is about 9% of the soft context's gain, and openers like "Honestly," / "Yeah," are next. So a behaviour-level PSM could hold while token-level fit fails. State this caveat in the intro, and phrase the conclusion as "not reproducible at the level of token distributions by described personas or learned contexts".
- Lines 346–348: "geometric ... clearly improves ... shows ... constructed rather than selected". The geometric fit has hundreds of free signed weights. The companion says its gain comes *entirely* from negative weights that cancel near-duplicate personas (line 270 says this). Large cancelling weights look more like curve-fitting than "assembling traits". Soften to "consistent with", and acknowledge the parameter-count difference.
- Line 366, "HHH persona shows up as the dominant component": in the arithmetic fit, college student (39%) edges out HHH (38%). Say "one of the two dominant components".
- Line 320, "not able to find any elicitation method ... that worked": the companion reports SALVE-style verbalisation recovers 15–18% of the gain at the paper's search budget. Say "recovers only a small fraction".
- Lines 316–317: the "stays close to its random initialization" claim isn't supported anywhere in the post. Add evidence or drop it.
- Line 339, "jumps up again for the final model": for the soft-prefix gap, that is 0.482→0.497, about 0.015. The companion calls it "flat through RL". Also note that "final Instruct" is many RL steps beyond step 300.
- Line 387, "conclusions were pretty much the same": add one line of numbers. The gap is 0.86 Olmo / 0.65 Llama / 0.50 Qwen, and Qwen's base shows signs of instruction data in pre-training. The ordering replicates; the size of the gap doesn't.
- All fits use a single seed and a single question split. Say so once.

## 4. Accessibility
- **nats/token is never explained.** Add one intuition sentence. For example: 0.82 nats/token means Instruct gives each token on average about e^0.82 ≈ 2.3× more probability than the best mixture does. Over a 45-token answer that compounds enormously. By contrast, 0.006 means the mixture is essentially indistinguishable from the base model. Also say what "KL" means at line 183.
- "HHH" is used at line 88 but only defined at line 160. Define it on first use, as well as SFT, DPO, RLVR (line 93) and "register" (line 268).
- Line 139: `P_c(x|c)` uses *c* for both the coefficients and the context. Rename the coefficients (e.g. λ_s), and use `\log`.
- Geometric mixture: add one intuitive sentence, e.g. "instead of rolling dice to pick a character, you blend their *tendencies* token by token, and can subtract one". Soft context: "a prompt written in the model's internal vector language rather than words, so it can say things no text can". Say why it's an upper bound for any written persona.
- Terms drift: soft context / soft prefix / context distillation; "gap" vs "share closed". The geometric table's "share closed" uses the generic header→Instruct as its denominator, while line 341 measures from the mixture. Define each one, or use one.
- The figure's "control fails" labels are never explained in the text.

## 5. Tables and figures
- Keep: the base elbow (194) and the Instruct elbow (206). Shown side by side, they make the core contrast vivid. Add one sentence of interpretation under each, and make sure the Instruct line is visible on the second plot.
- Keep the soft-context table (299–305), which is short and decisive.
- Keep the checkpoint figure, drop its table, and explain the annotations.
- Merge the two bases tables as above. Everything else goes to the repo or the companion.

## 6. Prose (typos: APPLIED 2026-11-19)
"disatisfying" (9), "Barrack" (46), "vestigal" (43), "what is essentially what is" (51), "amalgalm" (57), "Frankenstien"/"a importantly" (59), "are have" (126), "dilemas" (174), "Its"→"It's" (236, 366, 384), "initiaits random initialization" (317), "meaure" (328), "the large gap likelihood gap between the any" (350), "probabilies" (356), "perfoming" (359), "think we need" → "I think we need" (381), "blog post ... post" (37). Also fixed by the session: "seem" → "seems" (381), "can't full define" → "fully" (384).

## 7. Missing for LessWrong readers
- A link to the Anthropic PSM post (line 37). A citation for the "Logit Linear Selection paper" (line 280), which has no link. Please verify the arXiv IDs (2609.10883, 2602.04863, 2602.15669, 2601.10387); I did not check them.
- A behaviour-level check: do the mixture's *choices* on the dilemmas match Instruct's, even if the tokens don't? This is the obvious objection.
- A limitations paragraph: 7–8B models, single-turn dilemmas, personas defined by descriptions (anything without a short description is invisible), fixed global weights.
- "What would change my mind": e.g. a persona construction that reaches the soft-context ceiling, or a behaviour-level fit that succeeds.
- Alignment implications stated plainly: if base→SFT isn't selection, interventions that rely on the base persona prior, such as alignment pretraining, may have less leverage than PSM suggests.

## Rewrite suggestions
1. Lines 3–9 (TL;DR) → "I tried to make the Persona Selection Model testable: if post-training only *selects* among characters the base model already simulates, a mixture of those characters should reproduce the post-trained model's answers. For the base model's own assistant it does, almost perfectly. For OLMo 3 Instruct it fails badly, and even a freely optimised prompt to the base model closes only about 40% of the gap. Most of the gap is already there after supervised fine-tuning; later stages look much more like selection. I read this as: post-training builds the assistant out of pre-training pieces rather than picking one, though token likelihood also captures style, which is a real caveat."
2. Line 1 → "Post-Training Isn't (Just) Persona Selection: A Likelihood Test on OLMo 3"
3. Line 67 → "The experiments below are a first attempt to tell these views apart."
4. Line 350 → "No mixture of described base personas, and no learned prompt, reproduces Instruct's token distribution; at that level of description the PSM is not exhaustive."
5. Line 348 → "This is consistent with the assistant being *constructed* rather than selected, though the geometric fit also has many more free parameters and relies on cancelling weights."
6. Line 214 → "But the mixture fits Instruct far worse: about 0.82 nats/token, meaning Instruct puts roughly 2.3× more probability on each of its own tokens, and adding personas does not close it."
7. Line 339 → "**Most of the gap appears at SFT.** DPO adds about 0.09; RL barely moves it."
8. Line 32 → "My bottom line: selection is a good description of what happens *after* SFT, not of post-training as a whole."
