"""Task head: what kind of work the prompt asks for.

0.844 top-1 / 0.967 top-2 over six classes, cross-validated on 1,000 real
prompts. Trained only on real traffic -- a head trained on 14,776 Dolly rows
scores 0.700 on real prompts, below the 0.729 of always predicting `answer`.

Deliberately has no shortlist. ``class_weight="balanced"`` is what lifts the
rare classes and it flattens the probabilities enough that sizing by mass asks
for half the label space. Callers wanting a second candidate should read
``distribution`` and choose their own threshold -- and should, because `create`
reaches precision 0.033 on held-out data where `answer` reaches 0.992.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from router.heads.base import CalibratedHead


@dataclass(slots=True)
class TaskPrediction:
    """What kind of work one prompt asks for."""

    task: str
    confidence: float
    distribution: dict[str, float] = field(default_factory=dict)
    should_defer: bool = False


class TaskHead(CalibratedHead):
    """Serves the task-type head trained on 1,000 hand-labelled real prompts.

    Deliberately has no shortlist. `class_weight="balanced"` is what lifts the
    rare classes and it flattens the probabilities enough that sizing a
    shortlist by mass asks for 3.15 of 6 labels -- the trick that works on the
    domain head does not transfer here. Callers that want a second candidate
    should read `distribution` and decide for themselves.

    `distribution["media"]` is worth reading directly whatever the argmax says.
    A media request is the one task that leaves the language models entirely, so
    it is usually a gate rather than a ranking, and the threshold belongs to the
    caller: 0.5 flags 6.0% of traffic at precision 0.617 and recall 0.841, 0.7
    flags 2.4% at precision 0.875 and recall 0.477.
    """

    def predict(self, prompt: str) -> TaskPrediction:
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str]) -> list[TaskPrediction]:
        # An empty batch is a legitimate call on a serving path -- a filter
        # upstream removed everything -- and sklearn raises on it.
        if not prompts:
            return []
        proba = self._calibrated(self.model.predict_proba(list(prompts)))
        labels = self.labels
        out = []
        for row in proba:
            best = int(np.argmax(row))
            out.append(TaskPrediction(
                task=labels[best],
                confidence=float(row[best]),
                distribution={n: float(p) for n, p in zip(labels, row, strict=True)},
                should_defer=float(row[best]) < self.defer_below,
            ))
        return out
