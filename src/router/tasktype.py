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
    #: Produce text as the artifact: stories, posts, emails, copy -- and also
    #: rewriting or translating text the user supplied. Those two were tried as
    #: separate labels and the boundary is not learnable at this sample size
    #: (`edit` reached F1 0.465 and cost the rest of the taxonomy); the
    #: distinction is preserved in the `task_detail` column of the label files.
    CREATE = "create"
    #: Produce an image, video or other non-text artifact. Split out of
    #: `create` because it is the one task type that does not route to a
    #: language model at all.
    MEDIA = "media"


TASK_LABELS: list[str] = [t.value for t in TaskType]

TASK_DESCRIPTIONS: dict[str, str] = {
    "answer": "answer a question, explain a concept, look up a fact",
    "ideate": "brainstorm options, suggest possibilities, generate ideas",
    "summarize": "condense or shorten supplied text",
    "extract": "pull specific facts or fields out of supplied text",
    "classify": "sort into categories, decide which group something belongs to",
    "create": "write, rewrite or translate text: stories, posts, emails, copy",
    "media": "produce an image, video or other non-text artifact",
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
#: Dolly has no `media` category -- it predates the question. Another way of
#: saying that the source was never the right shape for this traffic.


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
# What happened when this met real traffic, and what replaced it
# ---------------------------------------------------------------------------
#
# All of the above is a Dolly number, and Dolly is not the traffic this router
# will see. 1,000 random real prompts were hand-labelled with these six task
# types (``data/handlabelled/real_tasks.parquet``). The mixes are not close:
#
#            answer  create  ideate  classify  summarize  extract
#     Dolly     51%      5%     12%       14%         8%      10%
#     real      73%     17%      7%        2%         1%       0.3%
#
# Dolly is balanced because employees were *asked* to write instructions in
# eight named categories. Real users ask questions (73%) and ask for things to
# be written (17%); everything else is a long tail. The first 200 labels and the
# next 800 agreed to within a point on every class, so this is the traffic, not
# a sampling accident.
#
# Held out on real prompts, five-fold cross-validation:
#
#     always predict "answer"                       0.729
#     Dolly-trained head, 14,776 rows               0.700   <- below the constant
#     three encoders, trained on 1,000 real rows    0.787
#     + a word/char tf-idf member                   0.828 top-1 / 0.952 top-2
#
#     final config - majority baseline   +0.099  [+0.067, +0.131]  (paired
#     bootstrap, 4,000 resamples) -- significant.
#
# So 1,000 in-distribution labels are worth more than 14,776 out-of-distribution
# ones by 12.8 points, and the Dolly head is not distinguishable from a constant
# predictor. This is the v1 domain failure (0.91 on benchmarks, 0.47 in the
# wild) arriving a second time, caught the same way.
#
# **Dolly rows are actively harmful, not merely useless.** Mixed into the real
# training data at every ratio tried:
#
#     real only (1,000 rows)                        0.781
#     + 2,000 Dolly rows, equal weight              0.622
#     + all 14,776, equal weight                    0.460
#     + all 14,776, real rows weighted 15x          0.739
#
# Even weighted fifteen to one they cost four points. `load_dolly_tasks` is kept
# for reproducing this comparison; `load_real_tasks` is the training set.
#
# Prior correction was the obvious fix and it does not work. Estimating the
# target prior by EM from unlabelled real prompts (Saerens, Latinne &
# Decaestecker 2002) took top-1 from 0.700 to 0.220, putting `create` at 65%
# against a true 16%. Handing it the *true* prior reaches 0.740, so the method
# is sound and the assumption is wrong: this is not label shift. P(x|y) moved as
# far as P(y) did, because "Write a poem about the sea" and "refine: Please find
# attached reports" share a label and nothing else.
#
# Per class on real prompts (final configuration):
#
#     answer 0.901 (n=729) | create 0.778 (166) | summarize 0.526 (11)
#     ideate 0.470 (69)    | classify 0.455 (22) | extract 0.000 (3)
#
# The macro-F1 of 0.522 is the honest headline for anything that cares about the
# rare classes: `extract` has three examples and is never predicted, `summarize`
# eleven. Top-1 and top-2 are carried by `answer` and `create`, which is exactly
# the split a router can act on today -- answer-shaped work and produce-shaped
# work behave differently downstream. The rest needs more labels, not a better
# model; the biggest single confusion is `answer` -> `ideate` (47 of 172
# errors), a boundary that is genuinely soft ("what should I do about X" is both).
#
# Note the adaptive shortlist that works on the domain head does *not* transfer
# here: `class_weight="balanced"` is what lifts the rare classes, and it flattens
# the probabilities enough that mass >= 0.75 asks for 3.15 of 6 labels. Fix the
# calibration before reaching for that trick on this head.


# ---------------------------------------------------------------------------
# Splitting `create`: what the evidence actually said
# ---------------------------------------------------------------------------
#
# `create` was the largest non-`answer` class and it covered three things that
# route differently, so it was re-cut by hand across all 199 rows carrying it.
#
# The prediction going in was that it conflated prose with code. That was
# wrong, and measurably so: a regex for code-generation intent over all 1,240
# labelled prompts matches **six**, of which one ("can u optimise the code") is
# genuine. Code generation is as absent from this traffic as `extract` is. What
# `create` actually held was 45 requests for an image or video, 52 requests to
# transform text the user supplied, and 102 requests to write something new.
#
# Both candidate splits were then measured, and the eight-class head was scored
# against the six-class head collapsed onto identical rows:
#
#     six classes (create whole)                0.828 top-1  macro-F1 0.522
#     eight classes (create/edit/media)         0.774        macro-F1 0.492
#     seven classes (media only)                0.802        macro-F1 0.524
#
#     collapsed back to six, vs the six-class head:
#       eight-class   -0.022 top-1  [-0.036, -0.008]   significant
#       seven-class   -0.013 top-1  [-0.026, +0.000]   not significant
#
# So `media` separates (F1 0.725, recall 0.841) and `edit` does not (0.465),
# and taking both costs the rest of the taxonomy a real 2.2 points. Take the one
# the evidence supports. The `edit` distinction is not discarded -- it survives
# in the `task_detail` column of the label files, ready for a larger sample.
#
# `media` was also tried as a dedicated binary gate rather than a seventh class,
# on the reasoning that an image request is a routing decision of a different
# kind: it does not pick a cheaper or dearer language model, it leaves the
# language models entirely. The gate ranks well (average precision 0.783 against
# 0.044 for a random ranker) but scores the same as the class does (F1 0.712 vs
# 0.725), so the seventh class wins on simplicity -- one model, not two -- and
# the threshold is still available by reading `distribution["media"]`:
#
#     threshold   precision   recall   share of traffic flagged
#       0.4         0.500      0.909            8.0%
#       0.5         0.617      0.841            6.0%
#       0.7         0.875      0.477            2.4%

