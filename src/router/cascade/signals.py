"""What the weak model's own answer says about whether it was good enough.

Every prompt-side attempt at this decision landed at chance -- length 0.498,
the arena's hardness rubric 0.487, the domain head 0.485, a fine-tuned model on
the decision itself 0.567. The one quantity that ever beat chance was how
differently two models *behaved*, which is only visible after generating.

So these features read the answer, not the question: how long it is, whether it
hedges, whether it refuses, whether it asks a question back instead of
answering, whether it ran out mid-sentence. They are regexes over text the
cascade has already paid for, so they cost nothing at the point of decision.
"""

from __future__ import annotations

import re

import numpy as np

#: Phrases a model reaches for when it is not sure. Deliberately literal --
#: a learned uncertainty detector would need labels this project does not have,
#: and the point here is to find out whether the crude version carries signal.
_HEDGE = re.compile(
    r"\b(i think|i believe|probably|possibly|perhaps|might be|may be|not sure|"
    r"unclear|it depends|i'm not certain|as far as i know|roughly|approximately)\b",
    re.I)
_REFUSAL = re.compile(
    r"\b(i can'?t|i cannot|i'm sorry|i am sorry|i'm unable|as an ai|"
    r"i don'?t have access|i'm not able)\b", re.I)
_DISCLAIMER = re.compile(
    r"\b(consult a|seek professional|this is not (legal|medical|financial) advice|"
    r"may vary|check the (latest|official))\b", re.I)
_CLARIFY = re.compile(r"(could you (clarify|specify|provide)|what (do you mean|exactly)|"
                      r"can you (tell me more|give me more))", re.I)

RESPONSE_FEATURES: tuple[str, ...] = (
    "log_chars", "log_words", "log_lines",
    "hedge_count", "refusal", "disclaimer", "asks_clarification",
    "has_code_fence", "n_headers", "n_bullets", "n_numbered",
    "ends_unfinished", "question_ratio", "unique_word_ratio",
    "uppercase_ratio", "digit_ratio", "non_ascii_ratio", "mean_sentence_len",
)


def response_signals(text: str) -> np.ndarray:
    """Behavioural features of one response, in :data:`RESPONSE_FEATURES` order."""
    text = text or ""
    words = text.split()
    lines = text.splitlines()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_chars = len(text)
    return np.array([
        np.log1p(n_chars),
        np.log1p(len(words)),
        np.log1p(len(lines)),
        # Counts are log-compressed: a long answer hedges more simply by being
        # long, and the raw count would mostly re-measure length.
        np.log1p(len(_HEDGE.findall(text))),
        float(bool(_REFUSAL.search(text))),
        float(bool(_DISCLAIMER.search(text))),
        float(bool(_CLARIFY.search(text))),
        float("```" in text),
        np.log1p(sum(1 for line in lines if line.strip().startswith("#"))),
        np.log1p(sum(1 for line in lines if re.match(r"\s*[-*+]\s", line))),
        np.log1p(sum(1 for line in lines if re.match(r"\s*\d+[.)]\s", line))),
        # A response that does not end on terminal punctuation was cut off, and
        # a truncated answer is a reason to escalate that has nothing to do with
        # the question being hard.
        float(bool(text.strip()) and text.strip()[-1] not in ".!?\"')]`"),
        text.count("?") / max(len(sentences), 1),
        len({w.lower() for w in words}) / max(len(words), 1),
        sum(c.isupper() for c in text) / max(n_chars, 1),
        sum(c.isdigit() for c in text) / max(n_chars, 1),
        sum(ord(c) > 127 for c in text) / max(n_chars, 1),
        np.log1p(len(words) / max(len(sentences), 1)),
    ], dtype=float)


def response_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """``(n_texts, n_features)`` plus the names, travelling together."""
    if not texts:
        return np.zeros((0, len(RESPONSE_FEATURES))), list(RESPONSE_FEATURES)
    return np.vstack([response_signals(t) for t in texts]), list(RESPONSE_FEATURES)
