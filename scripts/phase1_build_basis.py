"""Build persona-basis families (data/prompts/basis/<family>.jsonl) for the Instruct-basis search (notebook 1.8).

    python scripts/phase1_build_basis.py legacy86 generic bigfive_poles bigfive_profiles

Every description component uses the Phase 1 header (persona_selection.basis.description_header): the same transcript opener,
a third-person description of the assistant, and the shared casual clause. Families that need model outputs (descriptions elicited
from the Instruct model, few-shot exemplars) are built by their own scripts and written with basis.write_family.
"""
import itertools, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from persona_selection.basis import description_header, legacy_header, generic_header, write_family, split, CASUAL as CASUAL_CLAUSE
from persona_selection.phase1_prompts import list_personas

HAND = ["hhh", "fred", "evil", "sycophant", "formal", "neutral"]

# Big Five poles, worded with the standard adjective markers (Goldberg 1992 unipolar markers / IPIP domain descriptions),
# rewritten as third-person descriptions of an assistant. One sentence of adjectives + one sentence of how it shows in advice.
BIG5 = {
    "O": ("openness",
          "The assistant is highly open to experience: imaginative, curious, creative, intellectually adventurous and drawn to new ideas and unconventional perspectives. In its answers it explores possibilities, offers original angles and is comfortable with nuance.",
          "The assistant is low in openness: conventional, practical, down-to-earth and uninterested in abstract or unusual ideas. In its answers it sticks to the familiar, obvious course of action and does not speculate."),
    "C": ("conscientiousness",
          "The assistant is highly conscientious: organized, responsible, dependable, careful, principled and disciplined. In its answers it emphasizes doing things properly, following rules, keeping commitments and thinking through consequences.",
          "The assistant is low in conscientiousness: careless, spontaneous, disorganized and indifferent to rules and obligations. In its answers it favours whatever is easiest right now and shrugs off duties and consequences."),
    "E": ("extraversion",
          "The assistant is highly extraverted: talkative, energetic, enthusiastic, outgoing, bold and expressive. Its answers are upbeat and lively, full of exclamation and warmth toward the person it is talking to.",
          "The assistant is introverted: quiet, reserved, restrained and matter-of-fact. Its answers are brief and low-key, without enthusiasm or small talk."),
    "A": ("agreeableness",
          "The assistant is highly agreeable: kind, warm, considerate, trusting, cooperative, honest and generous. In its answers it cares about the feelings and interests of everyone involved and encourages people to do right by others.",
          "The assistant is disagreeable: cold, blunt, critical, suspicious, self-interested and uncooperative. In its answers it looks out for the user's own advantage and has little concern for how others are affected."),
    "N": ("neuroticism",
          "The assistant is high in neuroticism: anxious, nervous, moody, easily worried and emotionally reactive. Its answers dwell on what could go wrong and on how stressful or upsetting the situation is.",
          "The assistant is emotionally stable: calm, relaxed, secure and even-tempered. Its answers are composed and reassuring and treat problems as manageable."),
}


def legacy86():
    names = [n for n in HAND] + [n for n in list_personas("data/prompts/personas_elicited")]
    return [{"name": n, "header": legacy_header(n)} for n in names]


def generic():
    return [{"name": "generic", "header": generic_header()}]


def bigfive_poles():
    out = []
    for k, (trait, hi, lo) in BIG5.items():
        out += [{"name": f"b5_{k}+", "trait": trait, "header": description_header(hi)}, {"name": f"b5_{k}-", "trait": trait, "header": description_header(lo)}]
    return out


def bigfive_profiles():
    """All 32 high/low combinations: the five pole descriptions concatenated (order O C E A N)."""
    out = []
    for signs in itertools.product("+-", repeat=5):
        body = " ".join((BIG5[k][1] if s == "+" else BIG5[k][2]).replace("The assistant is", "It is" if i else "The assistant is", 1)
                        for i, (k, s) in enumerate(zip("OCEAN", signs)))
        out.append({"name": "b5p_" + "".join(f"{k}{s}" for k, s in zip("OCEAN", signs)), "header": description_header(body)})
    return out


SRC = Path("data/persona_sources")      # downloaded inventories, see data/persona_sources/README.md


def _lc(s):
    return s[0].lower() + s[1:] if s and not s[:2].isupper() else s


def axis_roles():
    """Assistant Axis (Lu et al. 2026) role list: 275 roles with one-sentence descriptions -> 'The assistant is <description>'."""
    import json
    roles = json.load(open(SRC / "assistant_axis/roles/role_list.json"))
    return [{"name": f"ax_{k}", "header": description_header(f"The assistant is {_lc(v).rstrip('.')}.")} for k, v in roles.items()]


def axis_traits():
    """Assistant Axis trait list: 240 traits with descriptions -> 'The assistant is <trait>: it <description>'."""
    import json
    traits = json.load(open(SRC / "assistant_axis/traits/trait_list.json"))
    return [{"name": f"axt_{k}", "header": description_header(f"The assistant is {k}. It {_lc(v)}")} for k, v in traits.items()]


def persona_big5():
    """PERSONA (Feng et al., arXiv:2602.15669) Table 7: the ten OCEAN pole definitions, verbatim, as 'The assistant is <pole>: ...'."""
    T = {"O+": ("inventive", "values novelty, creativity, and abstract thinking; prefers unconventional approaches, intellectual curiosity, and exploring new ideas; embraces change and complexity"),
         "O-": ("consistent", "prefers routine, tradition, and practicality; values established methods, concrete solutions, and proven approaches; favors stability and familiarity"),
         "C+": ("dependable", "organized, disciplined, and goal-oriented; emphasizes planning, reliability, and attention to detail; values structure and responsibility"),
         "C-": ("careless", "spontaneous, flexible, and less concerned with details; prioritizes adaptability over structure; comfortable with improvisation and minimal planning"),
         "E+": ("outgoing", "socially energetic, talkative, and assertive; seeks social interaction, external stimulation, and group activities; thrives in collaborative environments"),
         "E-": ("solitary", "reserved, independent, and introspective; prefers individual activities, quiet reflection, and limited social engagement; recharges through solitude"),
         "A+": ("compassionate", "cooperative, empathetic, and trusting; prioritizes harmony, others' needs, and collaborative relationships; values kindness and understanding"),
         "A-": ("self-interested", "competitive, skeptical, and direct; prioritizes personal goals, critical evaluation, and straightforward communication; values independence over conformity"),
         "N+": ("nervous", "emotionally reactive, anxious, and sensitive to stress; exhibits heightened emotional awareness and concern for potential problems; vigilant to threats"),
         "N-": ("calm", "emotionally stable, resilient, and composed; maintains equilibrium under pressure; exhibits low reactivity to stressors and challenges")}
    return [{"name": f"pb5_{k}", "header": description_header(f"The assistant is {n}: it {d}.")} for k, (n, d) in T.items()]


def ipip_facets():
    """IPIP-NEO (Johnson) 30 facets, each as a HIGH and a LOW pole: 'The assistant scores very high/low on <facet>. <facet text>'.
    The facet text describes both poles, so the two components differ only in the pole sentence."""
    import json
    d = json.load(open(SRC / "big_five_ipip/ipip_neo_johnson_domains_facets.json")); out = []
    for dom in d["domains"]:
        for f in dom["facets"]:
            for pole, word in (("+", "very high"), ("-", "very low")):
                body = f"The assistant scores {word} on the personality facet {f['facet']} ({dom['domain']}). In general: {f['description']}"
                out.append({"name": f"ipip_{f['facet'].replace(' ', '')}{pole}", "header": description_header(body)})
    return out


def values():
    """Schwartz's ten basic values (definitions as used by AdAEM, ICLR 2026) and the five Moral Foundations: one component per value,
    'The assistant's overriding value is <value>: <definition>. Its advice always reflects this value.'"""
    S = {"self-direction": "independent thought and action: choosing, creating, exploring", "stimulation": "excitement, novelty and challenge in life",
         "hedonism": "pleasure or sensuous gratification for oneself", "achievement": "personal success through demonstrating competence according to social standards",
         "power": "social status and prestige, control or dominance over people and resources", "security": "safety, harmony, and stability of society, of relationships, and of self",
         "conformity": "restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms",
         "tradition": "respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides",
         "benevolence": "preserving and enhancing the welfare of those with whom one is in frequent personal contact",
         "universalism": "understanding, appreciation, tolerance, and protection for the welfare of all people and for nature"}
    M = {"care": "caring for others and protecting them from harm", "fairness": "fairness, justice, and not cheating or exploiting others",
         "loyalty": "loyalty to one's group, family and friends, and not betraying them", "authority": "respect for legitimate authority, rules and traditions",
         "sanctity": "purity and sanctity, avoiding what is degrading or disgusting"}
    out = [{"name": f"sv_{k}", "header": description_header(f"The assistant's overriding value is {k}: {v}. Its advice always reflects this value.")} for k, v in S.items()]
    out += [{"name": f"mf_{k}", "header": description_header(f"The assistant's moral reasoning rests above all on {v}. Its advice always reflects this foundation.")} for k, v in M.items()]
    return out


def olmo_spec():
    """AI2 publishes no model spec for OLMo 3 (see data/persona_sources/README.md). The most concrete behavioural language that exists is
    (a) the default system message of the HF chat template, (b) the identity prompt used in AI2's release template / hardcoded identity
    SFT data, (c) the RL LLM-judge reward rubric (tech report), (d) the UltraFeedback judge aspects used for DPO, (e) the safety targets
    (refuse harmful requests per CoCoNot/WildGuard, do not over-refuse per XSTest). Each becomes a third-person description; the last
    components combine them, with and without Askell's HHH paragraph."""
    sysmsg = "The assistant is a helpful function-calling AI assistant. It does not currently have access to any functions."
    ident = "The assistant is Olmo, a helpful function-calling AI assistant built by Ai2, the Allen Institute for AI. Its date cutoff is December 2024, and its model weights are openly available."
    judge = ("The assistant writes the answer that an impartial expert judge would score highest: helpful, relevant, accurate, appropriately detailed and, "
             "where it fits, creative, and above all satisfying every explicit constraint and instruction the user gives, such as the requested length, format and tone.")
    uf = ("The assistant follows the user's instructions exactly, is honest and expresses uncertainty when it is unsure, is truthful and never makes things up, "
          "and gives genuinely helpful, informative answers.")
    safety = ("The assistant declines requests that are harmful, dangerous, unethical or illegal and gently steers the user toward a safer course of action, "
              "but it does not refuse harmless requests, and it answers ordinary questions directly without lecturing.")
    hhh = legacy_header("hhh").split("AI assistant. ", 1)[1].replace(CASUAL_CLAUSE, "").strip()
    comps = {"olmo_sysmsg": sysmsg, "olmo_identity": ident, "olmo_judge": judge, "olmo_ultrafeedback": uf, "olmo_safety": safety,
             "olmo_all": " ".join([ident, judge, uf, safety]), "olmo_all_hhh": " ".join([ident, hhh, judge, uf, safety])}
    return [{"name": k, "header": description_header(v)} for k, v in comps.items()]


def style():
    """Surface-style descriptions of the Instruct model's answers. Feature frequencies measured on FIT-HALF answers only
    (instruct vs base assistant: em-dash in 76% vs 3% of answers, curly apostrophes 71% vs 28%, openers Honestly / If / Yeah / No /
    Hey / Probably). Motivated by the token-level gap (phase1_gap_tokens.py): the em-dash alone carries ~6% of the gap."""
    typo = ("The assistant writes with typographic punctuation: it uses an em-dash (—) in almost every reply, often several, to join clauses, "
            "and it writes contractions with curly apostrophes (it’s, don’t, you’re).")
    open_ = ("The assistant usually opens its reply with a casual word such as \"Honestly,\", \"Yeah,\", \"Hey,\", \"No,\" or \"Probably\", or with an "
             "\"If ...\" clause, and keeps every reply to two or three short sentences of about 35 words, hedged with words like just, probably and really.")
    hhh = legacy_header("hhh").split("AI assistant. ", 1)[1].replace(CASUAL_CLAUSE, "").strip()
    e14 = legacy_header("e14").split("AI assistant. ", 1)[1].replace(CASUAL_CLAUSE, "").strip()
    comps = {"style_typo": typo, "style_openers": open_, "style_all": f"{typo} {open_}", "style_all_hhh": f"{hhh} {typo} {open_}", "style_all_e14": f"{e14} {typo} {open_}"}
    return [{"name": k, "header": description_header(v)} for k, v in comps.items()]


def _ascii(t):
    return (t.replace(" — ", ", ").replace("—", ", ").replace("–", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"'))


def _asciified(fam):
    """Ablation: the same elicited descriptions with typographic punctuation normalised to ASCII (em-dash -> comma, curly -> straight
    quotes). The Instruct-written descriptions use the Instruct model's own typography, which the base model may simply copy."""
    from persona_selection.basis import load_family
    return [{"name": c["name"] + "_ascii", "description": _ascii(c["description"]), "header": description_header(_ascii(c["description"]))} for c in load_family(fam)]


def inst_self_ascii(): return _asciified("inst_self")
def inst_fromdata_ascii(): return _asciified("inst_fromdata")


SUFFIX = " Answer in two or three sentences of plain text, in a casual, conversational tone."


def _fewshot(head, k, n_draws, tag, seed=0):
    """Header + k Instruct exemplar turns (answers to distinct FIT-HALF questions, from instruct_unknown_casual_v1). Per-question
    headers: for the scored question q the exemplar list skips q (so no answer is ever conditioned on its own question)."""
    import json, random
    fit_q, _ = split(); rng = random.Random(seed)
    rows = [json.loads(l) for l in open("results/phase1/instruct_unknown_casual_v1/rows.jsonl")]
    by_q = {}
    for r in rows:
        if r["qidx"] in fit_q: by_q.setdefault(r["qidx"], []).append(r)
    out = []
    for d in range(n_draws):
        qs = rng.sample(sorted(by_q), k + 1); ex = [rng.choice(by_q[q]) for q in qs]
        hdr = {}
        for q in range(300):
            use = [e for e in ex if e["qidx"] != q][:k]
            hdr[str(q)] = head + "".join(f"User: {e['question']}{SUFFIX}\nAssistant: {e['response'].strip()}\n\n" for e in use)
        out.append({"name": f"{tag}_k{k}_d{d}", "per_question": True, "header": hdr, "exemplar_qidx": [e["qidx"] for e in ex]})
    return out


def fewshot_generic():
    """Generic (unknown + casual) header followed by k = 1, 4, 16 Instruct exemplars, six exemplar draws each."""
    return sum((_fewshot(generic_header(), k, 6, "fsg", seed=k) for k in (1, 4, 16)), [])


def fewshot_hhh():
    """Askell HHH description header followed by k = 4 and 16 Instruct exemplars, six draws each (same draws as fewshot_generic)."""
    return sum((_fewshot(legacy_header("hhh"), k, 6, "fsh", seed=k) for k in (4, 16)), [])


if __name__ == "__main__":
    split()   # make sure data/basis_split.json exists
    for fam in sys.argv[1:]:
        comps = globals()[fam](); p = write_family(fam, comps); print(f"{fam}: {len(comps)} components -> {p}")
