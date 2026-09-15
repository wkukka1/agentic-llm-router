"""Domain head: what the prompt is about.

0.923 top-1 / 0.980 top-2 on 402 externally-labelled prompts. Eight classes,
predicted as ten and merged post hoc -- training directly on the coarse labels
reaches 0.738 where predicting fine and summing reaches 0.763.

The shortlist is the point. ``shortlist_mass`` sizes it by accumulated
probability rather than by fiat, spending two labels where the ambiguity is and
one where it is not: on the external set, mass >= 0.85 holds the true domain
0.985 of the time using 1.40 labels, against 0.980 using a fixed 2.00.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from decompose.classifiers.prompt_decomposition.heads.base import CalibratedHead


@dataclass(slots=True)
class DomainPrediction:
    """One prompt's domain evidence."""

    domain: str
    confidence: float
    #: Highest-probability domains first. Length is ``shortlist_size``, or
    #: chosen per prompt when the head was built with ``shortlist_mass``.
    shortlist: list[str] = field(default_factory=list)
    #: Full calibrated distribution over all domains.
    distribution: dict[str, float] = field(default_factory=dict)
    #: True when confidence is below threshold. The caller should widen its
    #: candidate set, fall back, or ask -- not treat ``domain`` as reliable.
    should_defer: bool = False

    @property
    def runner_up(self) -> tuple[str, float] | None:
        ranked = sorted(self.distribution.items(), key=lambda kv: -kv[1])
        return ranked[1] if len(ranked) > 1 else None


class DomainHead(CalibratedHead):
    """Loads a trained run directory and serves calibrated predictions."""

    def __init__(self, run_dir: str | Path, *, defer_below: float = 0.0,
                 shortlist_size: int = 2, shortlist_mass: float | None = None,
                 max_shortlist: int | None = None,
                 merge_domains: bool = False) -> None:
        super().__init__(run_dir, defer_below=defer_below)
        #: Fixed shortlist length, used when ``shortlist_mass`` is not set.
        self.shortlist_size = shortlist_size
        if shortlist_mass is not None and not 0.0 < shortlist_mass <= 1.0:
            raise ValueError(f"shortlist_mass must be in (0, 1]; got {shortlist_mass}")
        self.shortlist_mass = shortlist_mass
        #: Hard ceiling on an adaptive shortlist. Left unset there is none: a
        #: cap that quietly truncates would undo the point of sizing by mass,
        #: so a caller that wants one has to ask for it.
        self.max_shortlist = max_shortlist
        # Merging is applied to the *output* of a fine-grained model, never by
        # training on coarse labels. Measured: post-hoc merging reaches 0.7725
        # where retraining on the merged labels reaches 0.7375. Training on
        # coarse labels throws away distinctions the model can otherwise learn
        # and sum over.
        self.merge_domains = merge_domains

    def predict(self, prompt: str) -> DomainPrediction:
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str]) -> list[DomainPrediction]:
        proba = self._calibrated_proba(prompts)
        labels = self.labels
        if self.merge_domains:
            proba, labels = self._merge(proba, labels)
        out: list[DomainPrediction] = []
        for row in proba:
            order = np.argsort(-row)
            best = int(order[0])
            out.append(DomainPrediction(
                domain=labels[best],
                confidence=float(row[best]),
                shortlist=[labels[i] for i in order[: self._shortlist_len(row[order])]],
                distribution={name: float(p) for name, p in zip(labels, row, strict=True)},
                should_defer=float(row[best]) < self.defer_below,
            ))
        return out

    def _shortlist_len(self, descending: np.ndarray) -> int:
        """How many candidates this prompt needs.

        Fixed unless ``shortlist_mass`` is set, in which case take labels until
        their probabilities account for that much of the distribution -- one
        label when the model is decisive, more when it is genuinely torn.
        """
        if self.shortlist_mass is None:
            return self.shortlist_size
        n = int(np.searchsorted(np.cumsum(descending), self.shortlist_mass) + 1)
        cap = min(self.max_shortlist or len(descending), len(descending))
        return int(np.clip(n, 1, cap))

    @staticmethod
    def _merge(proba: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
        """Sum fine-grained probabilities into their merged groups."""
        from decompose.classifiers.prompt_decomposition.heads.domain_taxonomy import apply_domain_merges

        groups = [apply_domain_merges(x) for x in labels]
        merged = sorted(set(groups))
        index = {g: i for i, g in enumerate(merged)}
        out = np.zeros((len(proba), len(merged)))
        for j, g in enumerate(groups):
            out[:, index[g]] += proba[:, j]
        return out, merged
