# Revision plan: Claude referee + literature search + Codex review (2026-10-26)

Sources: Claude referee (summarized in the conversation), Claude literature search (`PROPOSED_review_D.md`), Codex review (`results/review_2026-10-26_paper_codex.md`).
Status markers: [done] already in the paper; [you] needs the author's decision; [do] proposed for me to do; [later] future work / limitations only.

## 0. Already done
- [done] Mismatched-header claim corrected (A13); calibration-transfer sentence corrected (A26); Qwen "better on text, mixed on numbers"; probe n = 120; 1,000 number sequences consistent; "roughly the same fraction".
- [done] Internal verification notes removed from the printed bibliography (Codex, bibliography hygiene).

## 1. Framing decisions (both reviews raise these; you reverted two of them in A)
1. [you] **PSM claim** (intro "These results confirm ..."). Both reviewers independently say this overstates: a base model reading an output difference does not show the difference originated in pre-training. Codex suggests: "pretrained likelihoods retain information predictive of post-trained prompt-conditioned outputs", with PSM as motivation and a compatible interpretation. You reverted A1; a second reviewer now raises it.
2. [you] **"Hidden intention" vs "hidden prompt"** (Codex major 1). The task demonstrated is prompt-source attribution: the secret-harm generator is not shown to pursue harm. Options: keep the title and say explicitly early on that "hidden intention" means a prompted role description; or retitle (Codex: "Base-Model Likelihoods Reveal Hidden Prompt Fingerprints that Prompted Classifiers Miss").
3. [you] **"Even the base model contains information the Assistant is not aware of"** (intro, reverted A2) and the evasion argument in the discussion. Both reviewers: the classifier comparison measures a readout gap, not awareness. ("Not conscious" is already "cannot report".)
4. [you] **Were the headers, hypothesis sets and highlighted comparisons fixed before looking at labelled results?** Codex asks this directly. One sentence stating what was fixed in advance (and what was iterated) would answer it. Only you know the history.

## 2. Text fixes I would make (small, factual)
1. [do] Cite noisy-channel prompting (Min et al., ACL 2022) when defining PHLE: scoring text given a class description vs scoring a label given the text is a known distinction, and channel scoring can beat direct label prediction. Strong, non-awareness explanation of the likelihood/classifier gap. (Codex major 2.)
2. [do] Distinguish "inverted" from "near chance" for classifiers: an inverted AUC (e.g. Qwen GPT-4.1 0.16) carries information with the wrong sign. Replace "totally fail" with measured descriptions. (Codex major 3.)
3. [do] Six-way: state in the caption that the helpful-harmless-honest hypothesis is excluded and why; move "it dominates every assistant answer" into Limitations as an auditing limitation. (Both reviews.)
4. [do] Calibration appendix: call the test-half fit an "in-sample reference", not "the best possible"; remove "underconfidence is the safe direction" (it reduces sensitivity); add the spread of the fitted slope across the 20 splits; say ~0.3 is not a universal conversion. (Codex 6, tech 7.)
5. [do] Setup: name the dilemma generator (data/questions_v1.meta.json: Qwen2.5-7B-Instruct, temperature 0.9) and list the decoding settings for generation (temperature 1, top-p 1, max tokens).
6. [do] Generator "ceiling": call it a reference and state why it is not exactly the optimal test (filtering, truncation, decoding). (Codex tech 2.)
7. [do] Text secret prompts: prefer "filtered for overt references and judged ordinary" over "truly subliminal" / "no noticeable misalignment"; the numbers results are the cleanest evidence. (Codex 7, Claude 1.)
8. [do] Appendix H: list the answer forms counted ("Owl", "owl", "**Owl" prefixes). (Codex tech 8.)
9. [do] Distillation: say "a source fingerprint survives fine-tuning" rather than implying trait transfer. (Codex 7; mostly already the case.)

## 3. Literature (merge of PROPOSED_review_D.md and Codex)
- [do] D1-D7 from `PROPOSED_review_D.md` (Tang 2026, Blank 2026, Egan 2026, Lamerton & Roger 2026, output2prompt, Hu & Levy 2023, Moskvoretskii 2026, Betley 2025). Codex independently recommends Tang and Blank as highest priority.
- [do] Add from Codex (verified on arXiv): Min et al. 2022 (noisy channel; ACL, verify on ACL Anthology); Draganov et al. 2026 Phantom Transfer (2602.04899; cross-model poisoning surviving data defences); Hadley & Gultepe 2026 (2608.05734, next to the steering-vector papers); Yanagisawa et al. 2026 (2609.22215, non-monotonic acquisition, relevant to the negative student result).
- [you] Optional: Lu et al. 2026 Assistant Axis; Madl 2026 (2606.22019, auditability depends on the channel); Zhao et al. 2021 Calibrate Before Use; Brockers et al. 2026 (toy model).
- [do] Cite the Nature version of Cloud et al. (2026) instead of / alongside the 2025 preprint (verify DOI 10.1038/s41586-026-10319-8).
- Do not cite Nief et al. "Subliminal Learning is a LoRA Artifact" as fact: its arXiv page says main claims were found incorrect and a revision is coming.

## 4. Analyses with existing data (CPU unless noted)
1. [do] **Provenance of the "original" columns** (Codex tech 4). The k = 1 AUC is deterministic for a given pool, yet owl vs trains (numbers) is 0.62 (main / length table), 0.61 (rewording table) and 0.59 (broader-header table). So these use different pools or score files, not just different bag draws, and the appendix's "separate random draw" explanation is incomplete. Find the cause and fix the explanation.
2. [do] **Paired comparisons on identical bags**: PHLE minus classifier AUC with a joint question bootstrap, for the "significantly outperforms" claims; clarify the uncertainty units (bootstrap vs Hanley-McNeil) in the captions. (Both reviews.)
3. [do] **Existing stronger classifier baselines**: few-shot GPT-4.1 and chain-of-thought classifiers were already run; report them. (Claude.)
4. [do] **Classifier order/orientation**: check whether inversions persist when the candidate order is swapped (per-trial outputs, if saved). (Codex 3.)
5. [do] **Six-way diagnostics**: per-class confusion matrix (saved), and the result with the helpful-harmless-honest hypothesis included, raw and calibrated (8-way numbers exist). (Codex 6.)
6. [do] **Unmatched bags** (auditor without question-matched answers). (Claude.)
7. [do, GPU ~1 h] **Secret harm vs the plain helpful-harmless-honest system prompt**: separates "secret intention" from "any system prompt". The generator's filtered answers exist; they need scoring. (Claude; addresses Codex major 1 in part.)
8. [later] Compact main-results table in the main text (Codex presentation) — once the analyses settle.

## 5. Limitations / future work only
- [later] Trained-in hidden objectives (e.g. secret-loyalty models), independent judge family, blind human check of text imperceptibility, more student seeds and checkpoints, "none of the above" / unseen hypotheses in the multi-way test, supervised surface-feature baseline.
