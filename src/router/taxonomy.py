"""Domain label space for the router's first stage.

Two earlier attempts failed, and this design is shaped by both:

**v1 -- Dewey decimal topics (9 classes).** Scored 91% on benchmark questions
and **47% on real user prompts**. It had learned exam formatting: its best class
(`language`, 0.99 F1) was the WMT translation subset, identifiable because the
text is not English. On real traffic that class became a dumping ground.

**v2 -- capability labels (5 classes).** Fixed the distribution problem (77% on
real prompts) but collapsed everything non-code, non-maths into a single
`other` class holding 58% of traffic. Useful for routing, useless as a domain
signal.

**v3 -- this.** Domains that are *distinct*, *intuitive*, and *cover real
traffic*, with every class grounded in labelled data from more than one
distribution. The design rules:

1. **A class must be assignable from real prompts**, not only benchmarks. A
   label no real data can supply is a label the model will invent.
2. **Boundaries follow how people ask, not how librarians file.** Medicine is
   its own domain rather than a subdivision of technology; a question about
   Python and a question about buying a laptop are both `software_tech`.
3. **Questions *about* AI, ML and models are `software_tech`, not `meta_other`.**
   "What is a large language model?", "is my macro-F1 of 0.72 good?" and "how
   do I evaluate a classifier?" are technical questions about software, no
   different from a question about databases. `meta_other` is reserved for
   questions about *this assistant* ("what can you help me with?", "how
   confident are you?"), greetings, and prompts with no answerable content.

   This boundary was the single largest error source measured against
   externally-labelled sets -- 18 of 44 errors across 402 prompts, worth ~4.5
   points of accuracy -- and it was a definition disagreement, not a model
   failure.

4. **`personal_life` and `meta_other` exist.** Roughly a third of real traffic
   is advice, chat, or questions about the assistant itself. v1 had nowhere to
   put these, which is what created its dumping ground.
"""

from __future__ import annotations

import re
from enum import StrEnum


class Domain(StrEnum):
    """What a prompt is about. Twelve mutually distinct domains."""

    SOFTWARE_TECH = "software_tech"
    SCIENCE_MATH = "science_math"
    MEDICINE_HEALTH = "medicine_health"
    BUSINESS_FINANCE = "business_finance"
    LAW_POLITICS = "law_politics"
    HUMANITIES = "humanities"
    ARTS_ENTERTAINMENT = "arts_entertainment"
    LANGUAGE = "language"
    PERSONAL_LIFE = "personal_life"
    META_OTHER = "meta_other"


DOMAIN_LABELS: list[str] = [d.value for d in Domain]

#: One-line descriptions, used for zero-shot anchors and for documentation.
DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "software_tech": "programming, software, IT, devops, AI and machine learning, models, prompts",
    "science_math": "mathematics, statistics, logic, physics, chemistry, biology, earth science, astronomy",
    "medicine_health": "medicine, health, symptoms, fitness, nutrition, mental health treatment",
    "business_finance": "business, economics, finance, investing, marketing, careers, management",
    "law_politics": "law, regulation, government, policy, politics, current affairs",
    "humanities": "history, geography, philosophy, religion, culture, society",
    "arts_entertainment": "music, film, games, sports, art, design, celebrities, and writing stories, poems or jokes",
    "language": "translation, grammar, vocabulary, linguistics, wordplay, writing style",
    "personal_life": "personal advice, relationships, emotions, daily life, planning, travel",
    "meta_other": "questions about the assistant, greetings, unclear or off-taxonomy requests",
}


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


DIFFICULTY_LABELS: list[str] = [d.value for d in Difficulty]


# --------------------------------------------------------------------------
# Source mappings. Each returns a Domain or None (None = unusable, drop it).
# --------------------------------------------------------------------------

#: MMLU-Pro's 14 academic categories.
_MMLU_PRO: dict[str, Domain] = {
    "math": Domain.SCIENCE_MATH,
    "physics": Domain.SCIENCE_MATH,
    "chemistry": Domain.SCIENCE_MATH,
    "biology": Domain.SCIENCE_MATH,
    "computer science": Domain.SOFTWARE_TECH,
    "engineering": Domain.SOFTWARE_TECH,
    "health": Domain.MEDICINE_HEALTH,
    "psychology": Domain.MEDICINE_HEALTH,
    "economics": Domain.BUSINESS_FINANCE,
    "business": Domain.BUSINESS_FINANCE,
    "law": Domain.LAW_POLITICS,
    "history": Domain.HUMANITIES,
    "philosophy": Domain.HUMANITIES,
    # MMLU-Pro's own "other" is a grab-bag of miscellany, not our meta class.
    "other": None,
}


def domain_from_mmlu_pro(category: str) -> Domain | None:
    return _MMLU_PRO.get((category or "").strip().lower())


#: RouterArena's ``Category`` column -- its Dewey *sub*class, which is far more
#: specific than the ``Domain`` column that failed in v1. "61 Medicine and
#: health" is unambiguous; "6 Technology" (its parent) was not.
_ROUTERARENA_CATEGORY: dict[str, Domain] = {
    "00": Domain.SOFTWARE_TECH, "02": Domain.SOFTWARE_TECH, "62": Domain.SOFTWARE_TECH,
    "51": Domain.SCIENCE_MATH, "16": Domain.SCIENCE_MATH,
    "50": Domain.SCIENCE_MATH, "53": Domain.SCIENCE_MATH, "54": Domain.SCIENCE_MATH,
    "55": Domain.SCIENCE_MATH, "57": Domain.SCIENCE_MATH, "58": Domain.SCIENCE_MATH,
    "61": Domain.MEDICINE_HEALTH, "15": Domain.MEDICINE_HEALTH,
    "33": Domain.BUSINESS_FINANCE, "65": Domain.BUSINESS_FINANCE,
    "34": Domain.LAW_POLITICS, "32": Domain.LAW_POLITICS,
    "10": Domain.HUMANITIES, "17": Domain.HUMANITIES, "20": Domain.HUMANITIES,
    "90": Domain.HUMANITIES, "91": Domain.HUMANITIES, "30": Domain.HUMANITIES,
    "70": Domain.ARTS_ENTERTAINMENT, "78": Domain.ARTS_ENTERTAINMENT,
    "79": Domain.ARTS_ENTERTAINMENT, "77": Domain.ARTS_ENTERTAINMENT,
    "80": Domain.ARTS_ENTERTAINMENT,
    "40": Domain.LANGUAGE, "41": Domain.LANGUAGE, "42": Domain.LANGUAGE,
}


def domain_from_routerarena_category(category: str) -> Domain | None:
    """RouterArena categories are prefixed with their Dewey subclass number."""
    match = re.match(r"\s*(\d{2})", category or "")
    return _ROUTERARENA_CATEGORY.get(match.group(1)) if match else None


#: BIG-bench task name -> domain. Only tasks with an unambiguous subject are
#: mapped; the many synthetic-reasoning tasks are deliberately excluded, since
#: their prompts look like nothing a user would send.
_BIGBENCH: tuple[tuple[str, Domain], ...] = (
    ("arithmetic", Domain.SCIENCE_MATH),
    ("math", Domain.SCIENCE_MATH),
    ("algebra", Domain.SCIENCE_MATH),
    ("logical", Domain.SCIENCE_MATH),
    ("logic_grid", Domain.SCIENCE_MATH),
    ("cs_algorithms", Domain.SOFTWARE_TECH),
    ("code_line", Domain.SOFTWARE_TECH),
    ("auto_debugging", Domain.SOFTWARE_TECH),
    ("programming", Domain.SOFTWARE_TECH),
    ("physics", Domain.SCIENCE_MATH),
    ("chemistry", Domain.SCIENCE_MATH),
    ("biology", Domain.SCIENCE_MATH),
    ("cryobiology", Domain.SCIENCE_MATH),
    ("medical", Domain.MEDICINE_HEALTH),
    ("economics", Domain.BUSINESS_FINANCE),
    ("business", Domain.BUSINESS_FINANCE),
    ("law", Domain.LAW_POLITICS),
    ("social", Domain.LAW_POLITICS),
    ("history", Domain.HUMANITIES),
    ("anachronisms", Domain.HUMANITIES),
    ("proverbs", Domain.HUMANITIES),
    ("moral", Domain.HUMANITIES),
    ("ethic", Domain.HUMANITIES),
    ("philosophy", Domain.HUMANITIES),
    ("religio", Domain.HUMANITIES),
    ("movie", Domain.ARTS_ENTERTAINMENT),
    ("emoji_movie", Domain.ARTS_ENTERTAINMENT),
    ("music", Domain.ARTS_ENTERTAINMENT),
    ("sports", Domain.ARTS_ENTERTAINMENT),
    ("chess", Domain.ARTS_ENTERTAINMENT),
    ("checkmate", Domain.ARTS_ENTERTAINMENT),
    ("codenames", Domain.ARTS_ENTERTAINMENT),
    ("narrative", Domain.ARTS_ENTERTAINMENT),
    ("humor", Domain.ARTS_ENTERTAINMENT),
    ("joke", Domain.ARTS_ENTERTAINMENT),
    ("riddle", Domain.ARTS_ENTERTAINMENT),
    ("translation", Domain.LANGUAGE),
    ("linguistic", Domain.LANGUAGE),
    ("language_identification", Domain.LANGUAGE),
    ("word", Domain.LANGUAGE),
    ("morpheme", Domain.LANGUAGE),
    ("anagram", Domain.LANGUAGE),
    ("spelling", Domain.LANGUAGE),
    ("rhyme", Domain.LANGUAGE),
    ("emotion", Domain.PERSONAL_LIFE),
)


def domain_from_bigbench_task(task: str) -> Domain | None:
    """Map a BIG-bench task name, or None if its subject is ambiguous."""
    name = (task or "").strip().lower()
    for needle, domain in _BIGBENCH:
        if needle in name:
            return domain
    return None


def domain_from_arena_flags(
    *, is_code: bool, is_math: bool, is_creative_writing: bool
) -> Domain | None:
    """LMArena's flags, the only domain-ish annotation on real user prompts.

    Returns ``None`` when no flag fires. That is *not* a label -- an unflagged
    prompt could be about medicine, law, or anything else, and guessing would
    poison the training set. Those rows need a real label from elsewhere.
    """
    if is_code:
        return Domain.SOFTWARE_TECH
    if is_math:
        return Domain.SCIENCE_MATH
    if is_creative_writing:
        return Domain.ARTS_ENTERTAINMENT
    return None


#: Optional coarser grouping. Two pairs are merged because the distinction is
#: not one a labeller can make reliably *and* not one a router needs:
#:
#:   business_finance + law_politics -> business_law
#:       The largest genuine confusion. "Statute of limitations on unpaid
#:       invoices", "restricted payment baskets in credit agreements" -- both
#:       labels are defensible, and both route to the same kind of model.
#:   humanities + arts_entertainment -> culture
#:       History, philosophy, film and music behave alike for routing.
#:
#: Measured on the frozen eval: 0.7575 -> 0.7725 top-1, 0.8950 -> 0.9100 top-2.
#:
#: `science_math + software_tech -> technical` scores better still (0.7850 /
#: 0.9250) and is deliberately NOT included: maths and code route to different
#: models, so collapsing them buys accuracy on this metric by destroying a
#: distinction the router actually needs.
#:
#: Applied at build time, never in the stored labels. Merging is lossy and
#: one-way; the 10-class labels remain the source of truth so a future router
#: with different needs can regroup differently.
DOMAIN_MERGES: dict[str, str] = {
    "business_finance": "business_law",
    "law_politics": "business_law",
    "humanities": "culture",
    "arts_entertainment": "culture",
}

MERGED_DOMAIN_LABELS: list[str] = sorted(
    {DOMAIN_MERGES.get(d, d) for d in DOMAIN_LABELS}
)


def apply_domain_merges(domain: str) -> str:
    """Map a fine-grained domain onto its merged group, if it has one."""
    return DOMAIN_MERGES.get(domain, domain)
