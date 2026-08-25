"""Difficulty estimation: the second signal the weak/strong decision needs.

The design specifies difficulty as a weighted combination:

    difficulty = w_text * text_difficulty(prompt)
               + w_domain * domain_prior(domain)
               + w_length * length_term(n_tokens)

:class:`HeuristicDifficulty` implements exactly that with hand-set weights and a
rule-based ``text_difficulty``. It is a stand-in, not a model: the intended
final form is a trained regressor on LMArena's seven ``criteria_v0.1`` hardness
flags, which is why the interface returns the same ``[0, 1]`` scalar either way
and the pipeline does not care which is plugged in.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol

#: Per-domain prior in [0, 1]. Ordered by how often a domain's prompts need a
#: strong model, not by how "academic" the subject is.
DOMAIN_PRIORS: dict[str, float] = {
    "cs_general": 0.70,
    "technology": 0.65,
    "science": 0.65,
    "philosophy_psychology": 0.50,
    "social_science": 0.45,
    "history": 0.35,
    "literature": 0.35,
    "language": 0.30,
    "arts_recreation": 0.30,
}

#: Task types that raise difficulty regardless of topic.
TASK_PRIORS: dict[str, float] = {
    "code": 0.75,
    "math": 0.75,
    "reasoning": 0.70,
    "reading_comprehension": 0.50,
    "factual_qa": 0.35,
    "translation": 0.30,
    "creative_writing": 0.45,
    "other": 0.45,
}

_COMPLEXITY_MARKERS = (
    "prove", "derive", "optimize", "optimise", "analyze", "analyse", "compare",
    "trade-off", "tradeoff", "design", "architect", "refactor", "debug",
    "step by step", "explain why", "implications", "critique",
)
_MULTI_PART = re.compile(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+|\band then\b|\balso\b", re.IGNORECASE)
_CODE_FENCE = re.compile(r"```|\bdef \b|\bclass \b|;\s*$", re.MULTILINE)


class DifficultyEstimator(Protocol):
    def estimate(self, prompt: str, *, domain: str, task_type: str, n_tokens: int) -> float: ...


@dataclass(slots=True)
class HeuristicDifficulty:
    """Weighted text/domain/length combination, per the design formula."""

    w_text: float = 0.45
    w_domain: float = 0.35
    w_length: float = 0.20
    #: Token count at which the length term reaches ~0.63.
    length_scale: float = 600.0
    domain_priors: dict[str, float] = field(default_factory=lambda: dict(DOMAIN_PRIORS))
    task_priors: dict[str, float] = field(default_factory=lambda: dict(TASK_PRIORS))

    def text_difficulty(self, prompt: str) -> float:
        """Surface features that correlate with needing a stronger model."""
        lowered = prompt.lower()
        markers = sum(1 for m in _COMPLEXITY_MARKERS if m in lowered)
        score = min(markers / 3.0, 1.0) * 0.5
        if _MULTI_PART.search(prompt):
            score += 0.25
        if _CODE_FENCE.search(prompt):
            score += 0.15
        if "?" in prompt and prompt.count("?") > 1:
            score += 0.10
        return min(score, 1.0)

    def length_term(self, n_tokens: int) -> float:
        """Saturating in length: long prompts are harder, but not without bound."""
        return 1.0 - math.exp(-max(n_tokens, 0) / self.length_scale)

    def estimate(self, prompt: str, *, domain: str, task_type: str, n_tokens: int) -> float:
        # Domain and task priors are averaged: the topic and the required
        # capability are separate evidence for the same quantity.
        domain_prior = self.domain_priors.get(domain, 0.5)
        task_prior = self.task_priors.get(task_type, 0.5)
        prior = (domain_prior + task_prior) / 2

        raw = (
            self.w_text * self.text_difficulty(prompt)
            + self.w_domain * prior
            + self.w_length * self.length_term(n_tokens)
        )
        return float(min(max(raw / (self.w_text + self.w_domain + self.w_length), 0.0), 1.0))
