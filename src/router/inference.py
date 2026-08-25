"""Serving-side wrapper: the seam between this training harness and the router.

Training produces a run directory. The router consumes a :class:`DomainHead`,
which loads that directory and returns a calibrated distribution plus an
explicit ``should_escalate`` signal. Keeping the escalation policy here rather
than inside the router means the threshold is chosen from the same
risk/coverage curve the analysis reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from router.models import build as build_model
from router.training.calibration import apply_temperature


@dataclass(slots=True)
class DomainPrediction:
    """One routing decision's worth of domain evidence."""

    domain: str
    confidence: float
    #: Full calibrated distribution, highest first.
    distribution: dict[str, float] = field(default_factory=dict)
    #: True when confidence is below threshold: the router should widen the
    #: candidate pool, escalate to a stronger model, or ask a follow-up.
    should_escalate: bool = False

    @property
    def runner_up(self) -> tuple[str, float] | None:
        ranked = sorted(self.distribution.items(), key=lambda kv: -kv[1])
        return ranked[1] if len(ranked) > 1 else None


class DomainHead:
    """Loads a trained run and serves calibrated domain predictions."""

    def __init__(self, run_dir: str | Path, *, escalate_below: float = 0.6) -> None:
        self.run_dir = Path(run_dir)
        self.escalate_below = escalate_below

        config = yaml.safe_load((self.run_dir / "config.yaml").read_text())
        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        self.temperature = float(metrics["test"].get("temperature", 1.0))

        params = dict(config["model"].get("params") or {})
        params.pop("cache_tag", None)
        self.model = build_model(config["model"]["name"], **params)
        self.model.load(self.run_dir / "model")

    def predict(self, prompt: str) -> DomainPrediction:
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str]) -> list[DomainPrediction]:
        proba = apply_temperature(self.model.predict_proba(prompts), self.temperature)
        labels = self.model.labels
        results: list[DomainPrediction] = []
        for row in proba:
            best = int(np.argmax(row))
            confidence = float(row[best])
            results.append(
                DomainPrediction(
                    domain=labels[best],
                    confidence=confidence,
                    distribution={label: float(p) for label, p in zip(labels, row, strict=True)},
                    should_escalate=confidence < self.escalate_below,
                )
            )
        return results


def as_domain_fn(head: DomainHead):
    """Adapt a :class:`DomainHead` to the pipeline's ``DomainFn`` signature."""

    def domain_fn(prompt: str) -> tuple[str, float, dict[str, float]]:
        prediction = head.predict(prompt)
        return prediction.domain, prediction.confidence, prediction.distribution

    return domain_fn
