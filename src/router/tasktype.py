"""Task-type label space: what the user wants *done*, orthogonal to the domain.

The domain classifier answers "what is this about". This answers "what kind of
work is being asked for" -- and the two are genuinely independent: "summarise
this contract" and "summarise this paper" share a task and differ in domain,
while "explain contract law" and "draft me a contract" share a domain and
differ in task.

For a router that matters, because task type predicts *shape* of work in a way
domain does not: an extraction task wants a cheap precise model, an ideation
task wants a creative one, a research task may want tools.

**Design constraint learned from the data.** Dolly-15k labels three of its
eight categories (`closed_qa`, `information_extraction`, `summarization`)
entirely by whether a context passage was attached -- 100% of those rows have
context, 0% of the others do. A classifier reading only the instruction cannot
recover that distinction reliably, so the taxonomy below merges the three
question-answering variants into one class and keeps only distinctions that are
visible in the instruction text itself.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    """What kind of work the prompt asks for."""

    #: Answer a question. Open, closed, factual or explanatory -- all one task.
    ANSWER = "answer"
    #: Generate options, suggestions, possibilities. "give me ideas for..."
    IDEATE = "ideate"
    #: Condense supplied material.
    SUMMARIZE = "summarize"
    #: Pull specific fields or facts out of supplied material.
    EXTRACT = "extract"
    #: Sort something into categories, or judge which category it belongs to.
    CLASSIFY = "classify"
    #: Produce prose as the artifact: stories, poems, posts, copy.
    CREATE = "create"


TASK_LABELS: list[str] = [t.value for t in TaskType]

TASK_DESCRIPTIONS: dict[str, str] = {
    "answer": "answer a question, explain a concept, look up a fact",
    "ideate": "brainstorm options, suggest possibilities, generate ideas",
    "summarize": "condense or shorten supplied text",
    "extract": "pull specific facts or fields out of supplied text",
    "classify": "sort into categories, decide which group something belongs to",
    "create": "write a story, poem, post, email or other prose artifact",
}

#: Dolly-15k category -> TaskType. The three question-answering variants
#: collapse: their distinction is context-presence, not instruction wording.
DOLLY_MAP: dict[str, TaskType] = {
    "open_qa": TaskType.ANSWER,
    "general_qa": TaskType.ANSWER,
    "closed_qa": TaskType.ANSWER,
    "brainstorming": TaskType.IDEATE,
    "summarization": TaskType.SUMMARIZE,
    "information_extraction": TaskType.EXTRACT,
    "classification": TaskType.CLASSIFY,
    "creative_writing": TaskType.CREATE,
}


def task_from_dolly(category: str) -> TaskType | None:
    return DOLLY_MAP.get((category or "").strip().lower())


#: Measured on Dolly-15k, 5-fold CV, frozen bge-base + linear head:
#:
#:     instruction only                     0.772 top-1 / 0.906 top-2
#:     + has_context + log(context_length)  0.794 top-1 / 0.928 top-2
#:
#: The two extra features are nearly free at serving time -- the router knows
#: whether a document was attached -- and they lift exactly the classes that
#: needed it: summarize 0.452 -> 0.596 F1, extract 0.598 -> 0.664. Everything
#: else moves by ~1 point.
#:
#: Per class (with context features):
#:     classify 0.940 | answer 0.839 | ideate 0.699 | extract 0.664
#:     create 0.619   | summarize 0.596
CONTEXT_FEATURES = ("has_context", "log_context_length")
