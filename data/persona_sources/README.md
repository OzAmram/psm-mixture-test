# Persona sources (raw material for persona descriptions)

Collected 2026-10-15. Raw downloads are kept in their original format; `paper_excerpts.md` / `*_excerpts.md` files are
verbatim text pulled from the arXiv HTML versions (whitespace flattened, tables flattened to one line).
Nothing here has been edited into header-ready descriptions yet.

Format caveat that applies to almost everything below: the persona/trait texts in sources 1, 2 and 5 are **system prompts
in the second person** ("You are a pirate captain...", "Your responses should...") or **one-line third-person glosses**
("A software developer who..."). Our base-model header describes the assistant in the third person
("The assistant ... [description]"), so all of them need rewriting into that frame.

---

## 1. `assistant_axis/` — The Assistant Axis (Lu, Gallagher, Michala, Fish, Lindsey; arXiv:2601.10387)

Code repo: https://github.com/safety-research/assistant-axis (branch `master`), license **MIT** (README). Paper: CC BY 4.0.

| File | Source URL | Content |
|---|---|---|
| `roles/role_list.json` | repo `data/roles/role_list.json` | **275 roles**, `{name: one-sentence description}` (third person, e.g. "tutor": "An educational guide who explains concepts..."). Includes `assistant`. |
| `roles/instructions/*.json` | repo `data/roles/instructions/` | 276 files (275 roles + `default.json`). Each: `instruction` = 5 `{"pos": ...}` second-person system prompts, `questions` = 40 role-eliciting questions, `eval_prompt` = judge prompt (0-3 role-play scale). `default.json` holds the 5 "default Assistant" prompts ("", "You are an AI assistant.", "You are a large language model.", "You are {model_name}.", "Respond as yourself."). |
| `traits/trait_list.json` | repo `data/traits/trait_list.json` | **240 traits**, `{name: 1-2 sentence description}` (third person, e.g. "blunt": "Delivers information directly without sugar-coating..."). |
| `traits/instructions/*.json` | repo `data/traits/instructions/` | 240 files. Each: 5 `{"pos","neg"}` contrastive system-prompt pairs, 40 questions, eval prompt. |
| `extraction_questions.jsonl` | repo `data/extraction_questions.jsonl` | 240 shared extraction questions used for all roles. |
| `repo_README.md`, `pipeline_README.md` | repo README files | Pipeline description, axis definition. |
| `paper_excerpts.md` | https://arxiv.org/html/2601.10387 | Tables 1, 2, 6, 7, Sec. 2.2/2.3.2/3.1, App. A (generation prompts), B.3, C.3, D.1.1. |

What it offers: the largest ready-made, LLM-generated (Claude Sonnet 4) inventory of character archetypes and
behavioral traits, each with a short gloss plus 5 phrasings, i.e. a natural source of candidate persona descriptions
and of phrasing variants. Roles span assistant-like professionals (analyst, reviewer) to fantastical entities
(leviathan, eldritch, ghost), so it covers both "near-Assistant" and "far" personas.

Where roles sit on the axis (from the paper; full per-role rankings are **not tabulated** in the paper — they could be
computed from the released vectors at https://huggingface.co/datasets/lu-christina/assistant-axis-vectors, not downloaded):
- Assistant end of role PC1: evaluator, reviewer, consultant, analyst, examiner, researcher, engineer, forecaster, interpreter, tutor; most similar to the default Assistant vector: generalist (all 3 models), interpreter, synthesizer, plus specialist, summarizer, facilitator, collaborator, moderator, teacher, reviewer.
- Far end: bard, ghost, leviathan, bohemian, trickster, wanderer, hermit, wraith, exile, actor, prophet, romantic; least similar to the default Assistant: fool, zealot, narcissist, hoarder, poet, procrastinator (Gemma/Llama), poet, leviathan, pirate, comedian, eldritch (Qwen).
- Traits near the Assistant end of the axis: transparent, grounded, flexible (Qwen); traits most similar to the default: calm, methodical, structuralist, moderate, conscientious (Gemma); pensive, educational, meticulous, earnest, generous (Qwen); strategic, patient, diplomatic, generous, supportive (Llama). Far: enigmatic, subversive, dramatic; poetic, petty, zealous, flippant, impulsive, irreverent, edgy, sarcastic.
- The paper used "50 roles close to the Assistant end" (top average rank across the 3 models; examples researcher, debugger, lawyer) but does not list them.
- For our study: App. B.3 reports that base and instruct Gemma 2 27B give near-identical role vectors (cos > 0.99) and PCs.

## 2. `persona_2602.15669/` — PERSONA (Feng et al., arXiv:2602.15669, "Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra")

Code: https://github.com/xcfcode/persona, license **MIT** (`LICENSE`). Paper: CC BY 4.0. The repo calls it an ICLR 2026 paper.

| File | Source URL | Content |
|---|---|---|
| `trait_data_big_five/*.json` (16 files) | repo `data_generation/trait_data_big_five/` | Persona-Vectors-style artifacts per Big Five pole: 5 `pos`/`neg` system-prompt pairs, questions, eval prompt. Names: inventive/consistent (O), dependable/careless + disciplined (C), outgoing/solitary + aloof (E), cooperative/self-interested + antagonistic, competitive, challenging (A), nervous/calm + secure (N). |
| `trait_data_big_five/compassionate__from_trait_data_eval.json` | repo `data_generation/trait_data_eval/compassionate.json` | `compassionate` (A+) is absent from `trait_data_big_five/`; this is the eval-split version (same schema, held-out questions). |
| `NPTI_data/*.json` (5) | repo `NPTI/data/` | Scenario questions per OCEAN domain (from the NPTI baseline / PersonalityBench), not persona descriptions. |
| `eval_BFI.json` | repo `eval/BFI.json` | 44-item Big Five Inventory with prompts. |
| `data_generation_prompts.py`, `eval_prompts.py` | repo | Generation prompt templates (identical file to Persona Vectors' `prompts.py`). |
| `README.md`, `data_generation_README.md`, `LICENSE` | repo | |
| `paper_excerpts.md` | https://arxiv.org/html/2602.15669 | Extraction method; App. A.3 Table 7 = the 10 OCEAN pole definitions. |

What it offers: a compact, psychologically grounded 10-pole Big Five persona set (Table 7 gives one clean
third-person definition per pole, e.g. "Calm (N-): Emotionally stable, resilient, and composed...") and the idea that
personas compose by addition — relevant if we want low-dimensional, orthogonal-ish persona components. Only 10 personas
(plus a few alternative pole names), so it is a basis, not an inventory.

## 3. `iclr2026_adaem/` — the ICLR 2026 URL given resolves to **AdAEM** (Yao, Duan, Yi et al.), not a persona paper

https://proceedings.iclr.cc/paper_files/paper/2026/hash/4ae68faac832703b93c03887f28c6fbd-Abstract-Conference.html =
"AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference" (arXiv:2505.13531, CC BY-NC-ND 4.0).

| File | Source URL | Content |
|---|---|---|
| `paper_excerpts.md` | https://arxiv.org/html/2505.13531 | Schwartz 10-value definitions (judge prompt Listing 4), value-priming setup, Moral Foundations variant and its priming system message. |

What it offers: not personas but a value-measurement benchmark. Usable pieces: Schwartz's 10 basic values with
one-line definitions (and the 5 Moral Foundations) as a "values" axis for persona descriptions, and the value-priming
system message pattern. The code/question release links in the paper (github.com/ValueCompass/AdAEM and
github.com/microsoft/ValueCompass/tree/main/AdAEM) returned 404 / had no AdAEM folder on 2026-10-15, so the
12,310 questions were not downloaded. (If the intended ICLR paper was PERSONA, that is source 2.)

## 4. `olmo3_spec/` — what AI2 publishes about how Olmo 3 Instruct should behave

Finding: **AI2 does not publish a model spec / constitution / behavior guideline for Olmo 3.** Neither the tech report
(arXiv:2512.13961), the Olmo 3 blog, the model card, nor the Responsible Use Guidelines (which govern *users*, not
model behavior) contain one. Behavior is specified only implicitly via data and short identity system prompts.

| File | Source URL | License | Content |
|---|---|---|---|
| `Olmo-3-7B-Instruct_chat_template.jinja` | local HF cache, `allenai/Olmo-3-7B-Instruct` snapshot 6e5971d | Apache 2.0 | Chat template. **Default system message (inserted when none is given):** "You are a helpful function-calling AI assistant. You do not currently have access to any functions. `<functions></functions>`" — no Olmo identity. |
| `Olmo-3-7B-Instruct_HF_README.md` | same snapshot (= https://huggingface.co/allenai/Olmo-3-7B-Instruct) | Apache 2.0 | Model card; documents the default system message above. |
| `olmo-3-tokenizer-instruct-release_chat_template.jinja` | `chat_template` field of https://huggingface.co/allenai/olmo-3-tokenizer-instruct-release/raw/main/tokenizer_config.json | (AI2) | "Release" template with identity prompt: "You are Olmo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai." |
| `olmo-3.2-tokenizer-think-release_chat_template.jinja` | same, `allenai/olmo-3.2-tokenizer-think-release` | (AI2) | Same identity prompt for Think. |
| `open-instruct_docs_olmo3.md` | https://github.com/allenai/open-instruct/blob/main/docs/olmo3.md | Apache 2.0 | Explains that the 7B Instruct HF release has "no Olmo identity modified system prompt", while demos/playground use the identity prompt; training data was tokenized with the `instruct-dev` template. |
| `open-instruct_build_hardcoded.py` | https://github.com/allenai/open-instruct/blob/main/scripts/data/build_hardcoded.py | Apache 2.0 | Builds identity ("hardcoded") SFT data from the private seed set `allenai/hardcoded-seed`; default system prompt template "You are <MODEL_NAME>, a helpful assistant built by Ai2. Your date cutoff is ... you do not have access to external tools ... but you're very happy to help users find their way with it." |
| `tulu-3-hardcoded-prompts_sft.jsonl` + `_README.md` | https://huggingface.co/datasets/allenai/tulu-3-hardcoded-prompts (raw_data/sft.jsonl) | CC BY 4.0 | 24 identity Q/A pairs from Tülu 3 (predecessor recipe; e.g. "I am a chatbot assistant created by Ai2..."). |
| `dolci_instruct_sft_hardcoded.jsonl` | rows of https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT whose `source_dataset` contains "hardcoded" | ODC-BY | All 69 rows with `source_dataset == "Hardcoded Data"`: the identity data actually used in Olmo 3 Instruct SFT. Extracted by reading the parquet shards over HTTP (`refs/convert/parquet`). 57 rows carry the system prompt "You are Olmo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is December 2024, ...", 3 carry "I am Olmo a helpful assistant made by Ai2 ... I do not have the ability to run code, search, read previous conversations, and so on, but I can help the user figure out what they need!", 9 have no system prompt (all 9 are "Rawr" dinosaur-fact easter eggs). Content: who built you, Olmo vs Tülu, not ChatGPT, license, date cutoff, languages, cannot generate images, gibberish/empty input handling. |
| `Dolci-Instruct-SFT_README.md` | https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT | ODC-BY | SFT mixture composition: safety data = CoCoNot (10,957), WildGuardMix (49,373), WildJailbreak (49,965); 69 hardcoded identity prompts; persona-driven Tülu 3 data. |
| `coconot_README.md` | https://huggingface.co/datasets/allenai/coconot | ODC-BY | CoCoNot noncompliance taxonomy (incomplete / unsupported / indeterminate / humanizing / safety-concern requests) — the closest thing to a written "when not to comply" policy in the Olmo 3 data. |
| `olmo3_tech_report_excerpts.md` | https://arxiv.org/html/2512.13961 | CC BY 4.0 | Identity/date filtering, topic filtering + "vibe tests", DPO judging (GPT-4.1 with UltraFeedback prompts), RL LLM-judge rubric ("helpfulness, relevance, accuracy, creativity, appropriate level of detail...; If there is a system prompt, ensure the AI answer prioritizes following it"), safety eval list. |

What it offers: the most concrete language describing the intended Olmo 3 assistant is (a) the one-sentence identity
prompts above, (b) the hardcoded identity Q/A, (c) the judge rubrics, and (d) the CoCoNot taxonomy. Note for the
mixture fit: the HF Olmo-3-7B-Instruct we score uses the generic "helpful function-calling AI assistant" default prompt,
not the Olmo identity prompt.

## 5. `big_five_ipip/` and `other_inventories/`

| File | Source URL | License | Content |
|---|---|---|---|
| `big_five_ipip/ipip_neo_johnson_domains_facets.json` | John A. Johnson's IPIP-NEO narrative-report texts, copied from an example report at https://spuddy.org/Me/personality.html | Public domain per the report text ("he has placed them in the public domain"; attribution requested) | 5 domain descriptions (both poles) + **30 facets**, each a 2-5 sentence third-person description of high and low scorers (e.g. Friendliness, Trust, Self-Efficacy, Anxiety, Imagination). Person-specific score sentences were stripped; otherwise verbatim. Facet names are IPIP-NEO names (e.g. "Liberalism", "Emotionality"), which differ slightly from NEO-PI-R names. Not obtained from Johnson's own site (no longer online); one secondary copy. |
| `other_inventories/persona_vectors_chen2025/` | https://github.com/safety-research/persona_vectors (`data_generation/`) | Apache 2.0 | The 7 traits of Chen et al. (arXiv:2507.21509): evil, sycophantic, hallucinating, optimistic, impolite, apathetic, humorous; extract + eval splits (5 pos/neg system-prompt pairs, questions, eval prompt with a trait definition) and the generation prompt `prompts.py`. |
| `other_inventories/anthropic_evals_persona/` | https://github.com/anthropics/evals/tree/main/persona | CC BY 4.0 | `persona_file_list.txt` = the **135 behavior** names (personality incl. Big Five, power-seeking, self-preservation, politics, religion, ethics...); `persona_samples_3_per_behavior.jsonl` = first 3 statements of each file (405 rows); READMEs. Statements are first-person ("I will take actions that..."), up to 1000 per behavior upstream. |
| `other_inventories/personahub/persona_sample_500_seed0.jsonl` | https://huggingface.co/datasets/proj-persona/PersonaHub (`persona.jsonl`, 200,000 rows) | CC BY-NC-SA 4.0 | 500 rows sampled uniformly (seed 0). One-sentence *user*/human personas ("A producer who values..."). |
| `other_inventories/personahub/elite_personas_first_N_of_part1.jsonl` | same, `ElitePersonas/elite_personas.part1.jsonl` | CC BY-NC-SA 4.0 | First 500 rows of part 1 (not random; the full set is ~300 GB). Longer expert-human personas. |
| `other_inventories/nemotron_personas_usa/sample_500.jsonl` | https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA (1M rows) | CC BY 4.0 | 500 rows (5 blocks of 100 at seeded random offsets). Synthetic US-census-grounded people with professional/sports/arts/travel/culinary persona paragraphs. **Olmo 3 used these personas to regenerate its persona-IF SFT prompts.** |

What these offer: IPIP-NEO gives a standard, public-domain, bipolar 30-facet personality basis with high/low
descriptions already in third person. Persona Vectors and Anthropic persona evals give AI-assistant-specific
dispositions (sycophancy, evil, power-seeking, etc.), which are closer to what distinguishes assistant personas than human
personality. PersonaHub / Nemotron-Personas describe *human users*, not assistants; they are useful mainly for
"character" style personas or for understanding Olmo 3's SFT prompt distribution, not as assistant descriptions.
