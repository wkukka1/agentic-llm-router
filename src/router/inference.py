"""Serving wrapper: the seam between this repo and whatever routes next.

Training writes a run directory. The downstream stage consumes a
:class:`DomainHead`, which loads that directory and returns a *calibrated*
distribution plus a shortlist.

The shortlist is the point. Measured on the frozen real-prompt eval, top-1 is
0.758 but top-2 is 0.895 and top-3 is 0.953 -- so a consumer that can accept
two or three candidates gets far more than one that demands a single answer.
Combining a shortlist with deferral is what reaches a 95% target:

    top-2 over the most-confident 70% of traffic   0.961
    top-2 over the most-confident 50%              0.980
    top-3 over all traffic                         0.953

``defer_below`` implements that trade. Calibration is what makes the threshold
mean anything: raw ECE is 0.272, and 0.058 after the temperature fitted on
validation, which this class applies automatically.

``shortlist_mass`` implements a better one. A fixed shortlist spends the same
budget on every prompt, but the prompts do not need the same budget: a
re-annotation of 200 random real prompts found a defensible second domain on
42.5% of them and only one on the rest, and the model scores 0.826 on the
single-domain half against 0.588 on the dual-domain half. Sizing the shortlist
by accumulated probability instead of by fiat spends two labels where the
ambiguity is and one where it is not.

                        in-house, nested CV      external, 402 prompts
                        labels   truth present   labels   truth present
    always 1 label       1.00        0.741        1.00        0.923
    always 2 labels      2.00        0.899        2.00        0.980
    mass >= 0.75         1.83        0.907        1.22        0.963
    mass >= 0.85         2.34        0.953        1.40        0.985
    mass >= 0.90         2.82        0.975        1.63        0.988

There is no single threshold that beats a fixed pair on both counts on both
sets: 0.75 dominates in-house and gives up 1.7 points externally, because the
external prompts are easier and the rule spends less on them. 0.85 is the
threshold that never loses -- better hit rate on both, and 0.60 fewer labels on
the external set -- so it is the one to reach for without a downstream cost
model. With one, read the curve: this is a dial, not a constant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from router.models import build


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


class DomainHead:
    """Loads a trained run directory and serves calibrated predictions."""

    def __init__(self, run_dir: str | Path, *, defer_below: float = 0.0,
                 shortlist_size: int = 2, shortlist_mass: float | None = None,
                 max_shortlist: int | None = None,
                 merge_domains: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.defer_below = defer_below
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

        config = yaml.safe_load((self.run_dir / "config.yaml").read_text(encoding="utf-8"))
        metrics = json.loads((self.run_dir / "metrics.json").read_text(encoding="utf-8"))
        # Fitted on validation by the experiment runner; without it the
        # confidence scores are not comparable to any threshold.
        self.temperature = float(metrics["test"].get("temperature", 1.0))

        params = dict(config["model"].get("params") or {})
        params.pop("cache_tag", None)
        self.model = build(config["model"]["name"], **params)
        self.model.load(self.run_dir / "model")

    @property
    def labels(self) -> list[str]:
        # Members may hand back numpy string arrays; coerce so callers get
        # plain str and JSON serialisation works.
        return [str(x) for x in self.model.labels]

    def predict(self, prompt: str) -> DomainPrediction:
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str]) -> list[DomainPrediction]:
        proba = self._calibrated(self.model.predict_proba(list(prompts)))
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
        from router.taxonomy import apply_domain_merges

        groups = [apply_domain_merges(x) for x in labels]
        merged = sorted(set(groups))
        index = {g: i for i, g in enumerate(merged)}
        out = np.zeros((len(proba), len(merged)))
        for j, g in enumerate(groups):
            out[:, index[g]] += proba[:, j]
        return out, merged

    def _calibrated(self, proba: np.ndarray) -> np.ndarray:
        if self.temperature == 1.0:
            return proba
        logits = np.log(np.clip(proba, 1e-12, None)) / max(self.temperature, 1e-12)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)
