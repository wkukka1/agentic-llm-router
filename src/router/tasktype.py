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


#: Measured on Dolly-15k, nested 5x3 cross-validation, nine members (four frozen
#: encoders with and without the context features, plus a word/char tf-idf head)
#: combined by a stacked logistic regression:
#:
#:     one frozen encoder, instruction only           0.772 top-1 / 0.906 top-2
#:     ensemble, instruction only                     0.805 top-1 / 0.923 top-2
#:     ensemble + has_context + log(context_length)   0.822 top-1 / 0.940 top-2
#:
#: Worth noting against the domain classifier, where the same sweep found that
#: stacking *lost* to a plain weighted average at every regularisation strength.
#: The difference is rows: 14,776 here against 2,441 there. Same harness, same
#: protocol, opposite verdict -- which is the protocol working.
#:
#: Per class on Dolly (stacked, with context features):
#:     classify 0.966 | answer 0.859 | ideate 0.745 | extract 0.714
#:     create 0.647   | summarize 0.641
CONTEXT_FEATURES = ("has_context", "log_context_length")


# ---------------------------------------------------------------------------
# What happened when this met real traffic
# ---------------------------------------------------------------------------
#
# All of the above is a Dolly number, and Dolly is not the traffic this router
# will see. 200 random real prompts were hand-labelled with these same six task
# types (``data/handlabelled/real_tasks_200.parquet``). The distributions are
# not close:
#
#            answer  create  ideate  classify  summarize  extract
#     Dolly     51%      5%     12%       14%         8%      10%
#     real      72%     16%      8%        4%         1%       0%
#
# `extract` did not occur once, `summarize` twice. Dolly's mix is balanced
# because employees were *asked* to write instructions in eight named
# categories; real users ask questions and ask for things to be written.
#
# The accuracy consequence, paired bootstrap over the 200 prompts:
#
#     always predict "answer"                      0.720  [0.655, 0.780]
#     Dolly-trained head, 14,776 rows              0.700  [0.635, 0.765]
#     head trained on these 200 real prompts, OOF  0.800  [0.740, 0.855]
#
#     real-trained - Dolly-trained   +0.100  [+0.035, +0.165]  significant
#     real-trained - majority        +0.080  [+0.015, +0.145]  significant
#     Dolly-trained - majority       -0.020  [-0.085, +0.045]  not significant
#
# **The Dolly-trained classifier is not distinguishable from always answering
# "answer".** 200 real labels beat 14,776 out-of-distribution ones, by a margin
# that survives the bootstrap. This is the same failure the domain classifier
# hit in v1 -- 0.91 on benchmark data, 0.47 in the wild -- and it was caught the
# same way, by hand-labelling real prompts rather than trusting a benchmark.
#
# Prior correction does not rescue it. The shift looked like textbook label
# shift, so the target prior was estimated from unlabelled real prompts by EM
# (Saerens, Latinne & Decaestecker 2002) and used to reweight the predictions.
# Top-1 went from 0.700 to 0.220: the estimate put `create` at 65% against a
# true 16% and `answer` at 12% against a true 72%. Supplying the *true* prior
# instead reaches 0.740, so the correction is not broken -- the assumption is.
# P(x|y) moved as much as P(y) did, because "Write a poem about the sea" and
# "refine: Please find attached reports" share a label and nothing else, and no
# estimator that only adjusts P(y) from unlabelled data can recover from that.
#
# So this classifier is not ready to route on. The path is not a better model:
# it is ~1,000 real prompts labelled with these six types, which on the evidence
# above is worth more than any amount of Dolly. Keep `extract` and `summarize`
# in the label space -- they are real tasks that this traffic sample happens not
# to contain -- but do not expect the classifier to have learned them.
