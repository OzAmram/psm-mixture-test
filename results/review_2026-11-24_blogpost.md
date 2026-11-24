# Second review of `my_writeup_persona_mixture.md` (2026-11-24)

Reviewer: a separate Claude agent with fresh context. I read the earlier review (`results/review_2026-11-19_blogpost_writing.md`), the current draft in full, the companion `claude_writeup_persona_mixture.md`, the relevant `NOTES.md` entries, and the four main figures. I checked the 721-context geometric weights against `results/phase4/exp1/instruct_unknown_casual_v1/bases/all/fits_A.json`. I edited nothing else. Line numbers refer to the draft as of 2026-11-24. As asked, I left out the MoralChoice numbers and the explanation-token caveat (both pending), the pending base-model geometric fit, and the "(Footnote: ...)" wrapper.

## Bottom line

1. The restructure worked. The post now reads as one argument (formula → base fits → Instruct fails → is it the basis, the mixing rule, sharpness? → soft context ceiling → behaviour → stages). Most of the earlier review's points are fixed (list at the end).
2. The main remaining problem is the conclusion. The TL;DR (line 8) and Discussion (412–415) say the results support weak PSM. The evidence mostly shows that *neither* formulation reproduces Instruct. The geometric fit's gain is real but closes at most about 28% of the gap.
3. The Discussion doesn't yet deal with its own behaviour result. If the soft context matches Instruct's choices about as well as a reworded Instruct does, then "PSM is not exhaustive" (419–420) is a token-level claim, not a behavioural one. The post should say which.
4. The sharpest finding is barely used. The SFT checkpoint plus a soft context gets within 0.05 nats/token of Instruct (405), while the base model plus a soft context stops about 0.5 short. That points to base→SFT as the step that isn't selection. It sits in one bullet with no interpretation.
5. Smaller issues: the central Instruct number (−1.72 vs −0.90) is never stated in its own section. "86" and the "casual clause" are used before they're introduced. The lens readout is read more strongly than its by-eye caveats allow. About 25 typos and small fixes are listed in section 4.

---

## 1. Argument and flow

- **[major] The conclusion overreaches what the experiments can tell apart.** Line 8: "I view this as support of the idea that post-training builds the assistant out of pre-training pieces"; line 412: "these results favor the weak PSM view over the strong PSM". In the post's own setup, weak PSM is tested by the geometric fit and, at the limit, by a learned context. Both still leave most of the gap: the geometric fit closes ≤28% of generic→Instruct (table at 220–230), and the soft context about 42%. A residual that no base-model context reaches fits "post-training adds something new" at least as well as "post-training assembles existing traits". The only evidence that specifically favours weak over strong is geometric > arithmetic, and that comparison has a parameter-count confound (see 2c). Fix: line 8 → "I read this as evidence against the strong version of the PSM. Combining personas in logit space does better than picking whole characters, as a 'trait assembly' view would predict, but most of the gap remains unexplained by either." In the Discussion, before line 412, add one or two sentences on what each view predicts here. Strong PSM: an arithmetic mixture over a good basis fits. Weak PSM: logit-space or learned-context fits do much better. Neither: everything stalls. Then say that the data land between the last two.

- **[major] The Discussion contradicts the behaviour section.** Lines 381–383 say the soft context predicts Instruct's MoralChoice choices "slightly" better than reworded Instruct does, and that the likelihood "is tracking meaningful behavior and not just writing style". Then lines 419–420 say "the large likelihood gap ... indicates the PSM is not exhaustive. Clearly post-training is doing something more complicated." If the rerun keeps the soft-context result, the natural reading is that a base-model context gets Instruct's *choices* (finetuning ≈ conditioning, behaviourally) but not its *tokens*. That's a different conclusion from 419–420, and arguably a more interesting one. Fix: replace 419–420 with "No mixture of described base personas, and no learned context, reproduces Instruct's token distribution. On moral choices, though, a learned base-model context comes close to Instruct's own consistency. So at the level of behaviour I tested, conditioning the base model gets most of the way, while the personas I could write down do not." Adjust after the rerun. Also drop "Clearly" (this was flagged last time and is still there).

- **[major] The stage result needs a sentence of interpretation, and it belongs in the TL;DR and Discussion.** Line 405: "Starting from the SFT checkpoint ... a learned soft context gets within 0.05 nats/token of Instruct". Together with lines 402 and 297, this says the DPO+RL step is reachable by conditioning and the base→SFT step is not. The companion (line 280) adds that SFT alone, with no description, closes about 80% of the base-to-Instruct distance. That's the post's most falsifiable claim, and right now the reader has to assemble it. Fix: after 405, add "So from SFT onward, post-training looks like conditioning: a context gets you there. The step that no base-model context reproduces is SFT itself." Then in the TL;DR, line 7 → "Most of the gap is already present after supervised fine-tuning (SFT). From the SFT checkpoint, a learned context reproduces the final model almost exactly." Add one Discussion sentence on what this means for alignment pretraining (line 78), which bets on the base model's prior.

- **[minor] The ending trails off.** The takeaway is at line 442, inside a quoted phrase, followed by three "stray thoughts" and a repo link. Fix: end the Discussion with three short "what I'd take away" bullets: (a) strong PSM fails at token level; (b) base→SFT is the non-selection step; (c) whether behaviour follows, per MoralChoice. Add a short limitations paragraph: 7–8B models; one seed per fit and per soft context; one question split; single-turn dilemmas; description-defined personas; fixed weights. The single-seed point is still missing from the earlier review. Then the stray thoughts, cut to two (the LLM-judge point is the strongest).

- **[minor] The TL;DR doesn't mention the geometric result, which the Discussion leans on.** Add: "Mixing personas in logit space, so traits can be added and subtracted, fits better than picking whole characters, but still closes under a third of the gap."

- **[minor] Section order.** "Across training stages" (392) comes after the behaviour section and is the shortest results section, yet it carries the sharpest finding. Consider moving it directly after the soft-context section (before 315 or 350), so the behaviour section closes the results and leads into the style-vs-behaviour discussion at 426.

- **[minor] The "Interpreting the soft context" opener raises a worry and doesn't answer it.** Lines 317–319: "pretty 'off manifold' ... different initializations converge to different points". The companion (line 255) has the answer: each vector stays close to its starting token (median cosine 0.82), and the updates point the same way across 35 starts (cosine 0.80). Add that sentence, or cut the worry.

- **[nit] Lines 52 and 40 say the same thing** ("most researchers ... believe"). Keep one. Line 55, the rename aside, would work better as a footnote (suggested last time).

## 2. Claims vs evidence (including number checks)

Numbers checked and matching the companion, `NOTES.md` or the fit file: 0.006 / −1.594 / −1.589 / 0.016 (187); 24/16/14% (193); 38%/39% (204); the whole bases table (220–230); HHH +0.65, +0.45, +0.41, +0.40, +0.37–0.38, −0.36/−0.35/−0.34, 15 vs 5, +36.7/−35.6 (248–251; from the λ = 0.001 fit); −1.714 positive-only (252); +1.70/−1.02 (254); T = 1.1–1.25 with gap 0.81–0.84 (266); −1.41 at one vector, ~96% (307); 0.09 at DPO, 36–40%, 12–14% (402–404); −0.945 vs −0.898 (405); footnote 0.04/0.015 (105).

**a. Numbers that disagree or need a qualifier**
- **[minor] Line 188**: "Without the shared casual clause the gap is larger, 0.010." The companion table (line 31) gives 0.011 ± 0.001. Change to 0.011.
- **[minor] Line 343**: "changes the output and late-layer readouts 2.5–5× less". From the sizes logged in `NOTES.md` (2026-11-20 22:40), the ratios are 5.4× (output), 4.5× (layer 20), 5.2× (24) and 3.7× (28). At layer 16 it is only 1.6×, and at layer 4 the control is larger. Say "about 4–5× less at the output and layers 20–28".
- **[minor] Line 234**: "all bases are marginal improvements with respect to the original 86". Per the table, Axis (−1.728), Big Five (−1.726) and spec (−1.724) are slightly *worse* than the 86 (−1.723). Only the pooled bases are better, by ≤0.009. Fix: "No basis does meaningfully better than the original 86; pooling everything gains 0.009 nats/token."
- **[minor] Line 403**: "During the RL stages it gets slightly smaller but then jumps up again for the final model." This was flagged last time and is still there. For the soft context the rise is 0.482→0.497, within about 2 standard errors. For the mixture it is 0.771→0.824. Fix: "flat through the early RL checkpoints; the final model moves a little further from every base description (mixture +0.05, soft context +0.015)".
- **[minor] Lines 331–335 (lens table)**: the rows are a hand-picked subset of the logged top-10 lists in `NOTES.md` (2026-11-20 22:10). Dropped from the output row: 't, or. Layer 16: —it. Layer 20: anyway, safer. Layer 24: but, yourself. The dropped ones are mostly function words, but say so: "top 10 by probability change; a few function-word tokens omitted". Also say which comparison this is. Line 327 says "versus the generic header", but the logged run is against the exact initialisation (generic header plus random filler). The notes say the two are nearly identical.
- **[nit] Line 362**: "the 64-token soft context". The soft-context results everywhere else use 128 tokens. State which context MoralChoice used and why. This may change with the rerun.

**b. Overstatements**
- **[major] Lens readout, lines 325 and 337–340, 347.** "yields some insights"; "from layer 16 on, the changes show deeper behavioral concepts: safety, honesty"; "It ... also has deeper shifts towards safety concepts like harmless". `NOTES.md` records this as "tentative, by eye". Below the last layer, the lens agrees with the real output change only at r ≈ 0.2–0.3 (0.55 at layer 28). There is one control pair and one fixed set of 200 answers. The answers are all moral dilemmas, so "safety"/"legal" tokens may partly be topic rather than a disposition. Fix: line 338 → "From layer 16 on, the readout (by eye) shifts towards safety, honesty and conditional advice to the reader. The lens only loosely tracks the actual output change at these layers (correlation 0.2–0.3), so I'd treat this as suggestive." Line 340 → "This is the kind of difference one might expect between base and post-trained models, which is encouraging, but it's one readout with one control."
- **[minor] Line 239 / 413–415: say what the geometric fit's extra flexibility does and doesn't explain.** The fit has up to 721 signed weights, against a simplex for the arithmetic mixture. The companion has a ready rebuttal the post doesn't use: on the *base* assistant's answers, the free geometric fit gains only 0.007 over the generic header (companion line 202). Adding that sentence after 239 turns a weakness into evidence.
- **[minor] Line 149**: "Standardizing the register in this way was found to improve the quality of the mixture fit." No number is given for the Instruct suffix. The companion's number (0.011→0.006) is for the casual clause on the *base* headers. Give the number or drop the sentence.
- **[minor] Line 323**: "I was not able to find an elicitation method that worked well". Add the number: the SALVE-style search recovered 15–18% of the soft context's gain at the paper's search budget, mostly by writing out Instruct-style filler (companion lines 258–259). That's informative, not just a failure.
- **[minor] Line 288**: "I verified this works ... gets a likelihood pretty close to the original one." Give the number. The companion's control is −0.93 vs −0.89 on Instruct's own answers. Say whether this sentence refers to that or to the hidden-system-prompt (owl) answers.
- **[minor] Line 370**: "sets a rough upper bound on how predictable this behavior is". Line 382 then reports a proxy beating it. That's possible, because the proxies are scored on the same wording while the ceiling crosses wordings, so it isn't an upper bound. Call it "a reference level" and add half a sentence on why it can be exceeded. Otherwise line 382 reads as an error.
- **[minor] Lines 144 and 459**: "results were similar" / "conclusions were pretty much the same". Still no numbers (flagged last time). Add one line: the ordering (written mixtures < geometric < soft context) replicates. The gap size doesn't: 0.86 Olmo, 0.65 Llama, 0.50 Qwen, and Qwen's base shows signs of instruction-style pretraining data.
- **[nit] Line 448**: "The simulator knows strictly more than the assistant persona". Soften to "arguably knows more".
- **[nit] Line 274**: please check that the Logit-Linear Selection paper (Aden-Ali et al., arXiv:2602.04863) does hypothesise that post-training "primarily shifts the embedded representation of the default system prompt". I could not verify that wording. Line 118 links the same paper as "log-linear properties"; name it there too so readers know it's the same work.

## 3. Clarity for an informed non-specialist

- **[major] The central Instruct result has no number in its own section.** Lines 200–211 give weights and say "a large gap ... remains", but the reader only got 0.82 as a forward reference at line 184. Add after 210: "The 86-persona mixture scores Instruct's answers at −1.72 nats/token; Instruct scores them at −0.90, a gap of 0.82 (against 0.006 for the base assistant). The mixture is only 0.008 better than the best single persona."
- **[minor] "86" and "80" appear before they're introduced.** Line 156 says "Of those 80" with no 80 yet. Line 187 says "all 86 personas". Fix line 154: "I then elicited 80 more personas from the base model (96 samples of 'Character description: The assistant…', filtered for length and near-duplicates), giving 86 in total." At line 156, add "(categories overlap)": 39+34+26+12+4 = 115%.
- **[minor] The "shared casual clause" is never defined.** Line 188 and the base elbow legend use it. Line 148 only describes the Instruct suffix. Add to Setup: "Every persona header, including the generic one, also ends with 'The assistant speaks in a casual, conversational tone', so register can't masquerade as character."
- **[minor] Undefined terms:** SFT (TL;DR line 7, line 396), DPO (396, 402), RLVR (80), "register" (149), "nats" before 182, "J++ lens" (325). Spell out each on first use. For the lens, add one sentence: "A lens projects a middle layer's activations onto the vocabulary, to read off which words the model is leaning towards at that depth." Line 348: "mech. interp." → "interpretability-based methods (e.g. steering vectors)".
- **[minor] Geometric formulation, lines 122–131.** The text implies that the log-linear form (125) and the logit form (131) are different things. Per token they are the same: log P_s and z_s differ by a per-token constant, which cancels after normalising over the vocabulary. The real point is that you normalise *per token*, not over whole responses. Fix line 127: "Normalising this over whole responses is intractable, so I normalise per token instead, which is the same as mixing the personas' logits:". Write the formula with the token index, z_{s}(x_t | x_{<t}, c). Line 137: "Actually ... I actually" → keep one.
- **[minor] Table at 293–299**: the "Control: Instruct learned 128-token soft context" row is unclear. Write "Control: same procedure on Instruct's own weights (shows the optimiser can reach the target)". Below the table, give the share: "closes about 42% of the generic-header-to-Instruct gap; 0.50 nats/token remain".
- **[minor] Table at 220–230**: add a "share of gap closed (geometric)" column (9/14/18/25/27/28%, self-descriptions about 29%), so readers don't have to compute it. Say what the bold marks ("best base-model basis"), since the self-description row (−1.512) is better.
- **[minor] Figures:**
  - `fig3_stage_gap.png` (400): the "control fails" labels are still unexplained (flagged last time), and "geometric fit A" is internal naming. Either explain the labels in the caption ("the same optimiser run on the checkpoint itself fell 0.044–0.062 short, missing a 0.05 pass mark") or remove them. Rename the series to "geometric fit". Also add a caption sentence: "lower is closer; SFT is already most of the way to the final gap".
  - `fig_lensweep_tuned.png` (305): the y-axis spans only 0.03 nats, so the curve looks like a big rise. Add a caption: "the whole y-range is 0.03 nats/token; the remaining gap to Instruct is 0.50". Or show the generic-header (−1.76) and Instruct (−0.90) levels in an inset.
  - `elbow_logp_instruct.png` (202): in the inset, the "HHH assistant" and "College student" labels both sit near point 1, so it's unclear which persona is chosen first. Move the labels next to their own points.
  - `elbow_logp.png` (190): the red "plain header" curve isn't mentioned in the text. Add a caption: "red: headers without the casual clause; blue: with it". Y-axis: "Hold out" → "Hold-out log P / token (nats)".
- **[nit] Line 398**: "the gap in the likelihood of its own responses, and the gap between it and our best elicitations" is hard to parse. Use "For each checkpoint, I score its own answers under the checkpoint itself and under each base-model description; the gap is the difference."
- **[nit] Line 177**: "within 2-3%" → "within 0.02–0.03 in weight", so it isn't read as relative error.

## 4. Typos, grammar, markdown

- 25: "likelihood of model actually generating" → "of the model".
- 21: "The devil being in how..." is a fragment → "The devil is in how...".
- 43: "in the pre-training" → "during pre-training".
- 46: "structures like coherent characters like Ned Stark" → "coherent characters such as Ned Stark".
- 48: "I think the work 'selection'" → "word".
- 75: "ie" → "i.e."; write `\text{HHH}` inside the math.
- 102: comma splice → "It serves as a baseline: if our persona mixtures are effective, they should outperform it."
- 108: `P_R (s \mid c)` → `P_R(s \mid c)`.
- 119: comma splice → "...hypothesis: if concepts are represented as linear directions in the residual stream, ..."
- 126–133: put blank lines before and after each `$$` block, and remove the trailing space after `$$` on 130. Some renderers, including LessWrong's markdown import, need this.
- 137–138: "Actually ... I actually" → drop the first "Actually"; "is lower dimension" → "is lower-dimensional".
- 144: double space before "Llama"; "Llama3.1-8B" → "Llama 3.1 8B", for consistency.
- 152: "What are the set of personas" → "What is the set of personas".
- 200: "Fitting the instruct model:" → "Instruct".
- 268: "'Finetuning = conditioning' ?" → "'Fine-tuning = conditioning'?". Pick one spelling of fine-tuning/finetuning throughout.
- 285: add authors to the SALVE citation (Hu et al. 2026), as for other citations.
- 287: "the students responses" → "the student's responses".
- 288, 301, 317: "soft-context" → "soft context" when it's a noun (used unhyphenated elsewhere).
- 340: "it's interesting its consistent" → "it's interesting that it's consistent".
- 382: "slighlty" → "slightly".
- 385: "preocedure" → "procedure".
- 386: "mostly same" → "mostly the same".
- 404: "the soft prefix" → "the soft context" (the term used everywhere else in the post).
- 407: "much much less" → "much less".
- 432: "this type of mixture fits could be" → "this kind of mixture fit could be".
- 442: the single-quoted phrase reads like a quotation. If it's your own words, drop the quotes. If it's a quote, cite it.

---

## Earlier points now resolved

Title and TL;DR now state a claim. The strong-vs-weak section is shorter and ends with "a first attempt to pull these views apart". nats/token are explained (182–184). HHH is defined at first use (75). The coefficient symbols are renamed (w_s, g_s). The placeholders are filled (soft-context length sweep, lens section). The two bases tables are merged. The checkpoint table is dropped in favour of the figure. Cancelling geometric weights are acknowledged (252–255). "HHH one of the dominant components" (434). The unsupported "stays near random init" claim is gone. Links are added for the PSM post and the Logit-Linear Selection paper. A behaviour-level test is added (MoralChoice). The style confound is raised before the behaviour test (352).

Still open from last time: "PSM is not exhaustive / Clearly" (419–420), "jumps up again" (403), "control fails" labels, Qwen/Llama numbers, the single-seed limitation, a limitations paragraph, and the SFT-vs-later framing (now partly in the text at 405, but not interpreted).
