"""Both heads over one prompt, plus the feature vector the next stage consumes.

The axes are independent by construction -- "summarise this contract" and
"summarise this paper" share a task and differ in domain; "explain contract law"
and "draft me a contract" share a domain and differ in task -- so neither is
recoverable from the other.

``vectorise`` is the handoff. It returns the matrix *and* its column names
together, because a feature vector whose column meanings live somewhere else is
one refactor away from being silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prompt_decomposition.domain_classifier.head import DomainHead, DomainPrediction
from prompt_decomposition.task_classifier.head import TaskHead, TaskPrediction


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats, 0 when certain and log(k) when uniform."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


@dataclass(slots=True)
class RouterPrediction:
    """Both axes for one prompt."""

    domain: DomainPrediction
    task: TaskPrediction

    @property
    def key(self) -> str:
        """``"<domain>/<task>"`` -- the cell a routing table would look up."""
        return f"{self.domain.domain}/{self.task.task}"

    def _distributions(self) -> tuple[np.ndarray, np.ndarray]:
        """Both calibrated distributions as arrays, alphabetical by label."""
        dom = np.array([self.domain.distribution[k] for k in sorted(self.domain.distribution)])
        tsk = np.array([self.task.distribution[k] for k in sorted(self.task.distribution)])
        return dom, tsk

    def _scalars(self, prompt: str | None) -> dict[str, float]:
        """The named scalar tail of the vector, in column order.

        Names and values are defined together, once. They used to be written
        twice -- an ordered list of names in one method and an ordered list of
        floats in the other -- which is exactly the drift :meth:`feature_names`
        promises cannot happen. Now a column cannot be added, moved or removed
        in one without the other following.
        """
        dom, tsk = self._distributions()
        d_sorted = np.sort(dom)[::-1]
        t_sorted = np.sort(tsk)[::-1]
        # Length is the one prompt property a difficulty model almost always
        # wants and neither head exposes. Zero when the prompt is not passed,
        # so the vector keeps its width either way.
        n_chars = len(prompt or "")
        n_words = len((prompt or "").split())
        return {
            "domain.confidence": float(d_sorted[0]),
            "domain.margin": float(d_sorted[0] - d_sorted[1]) if len(d_sorted) > 1 else 1.0,
            "domain.entropy": _entropy(dom),
            # How many domains the head could not separate for this prompt.
            "domain.shortlist_size": float(len(self.domain.shortlist)),
            "task.confidence": float(t_sorted[0]),
            "task.margin": float(t_sorted[0] - t_sorted[1]) if len(t_sorted) > 1 else 1.0,
            "task.entropy": _entropy(tsk),
            "prompt.log_chars": float(np.log1p(n_chars)),
            "prompt.log_words": float(np.log1p(n_words)),
        }

    def feature_names(self) -> list[str]:
        """Names for :meth:`vector`, same order, same length.

        Held next to the vector rather than written down elsewhere, because a
        feature vector whose column meanings live in a different file is one
        refactor away from being silently wrong.
        """
        names = [f"domain.{d}" for d in sorted(self.domain.distribution)]
        names += [f"task.{t}" for t in sorted(self.task.distribution)]
        return names + list(self._scalars(None))

    def vector(self, prompt: str | None = None) -> np.ndarray:
        """Fixed-length float vector for a downstream difficulty model.

        Layout, in order:

          * the full calibrated domain distribution, alphabetical by label
          * the full calibrated task distribution, alphabetical by label
          * domain confidence, margin (top-1 minus top-2) and entropy
          * the adaptive shortlist length, which is how many domains the head
            could not separate for this prompt
          * task confidence, margin and entropy
          * log character and word count, when ``prompt`` is supplied

        The distributions are included whole rather than reduced to an argmax
        because the shape carries information the label does not: a prompt
        split 0.45/0.44 between two domains is a different routing problem from
        one at 0.89, and both have the same argmax. Entropy and margin are
        derivable from the distribution, and are included anyway -- they are
        the two summaries a difficulty model is most likely to want, and
        computing them here means every consumer computes them the same way.

        Length is ``len(self.feature_names())``: 10 domains (or 8 merged) + 7
        tasks + 9 scalars. Stable for a given head configuration; changing
        ``merge_domains`` changes it, which is why the names travel alongside.
        """
        dom, tsk = self._distributions()
        scalars = np.array(list(self._scalars(prompt).values()), dtype=float)
        return np.concatenate([dom, tsk, scalars])

    @property
    def should_defer(self) -> bool:
        """True if *either* axis is unsure. Deliberately pessimistic: a
        confident domain paired with an unsure task is not a confident route."""
        return self.domain.should_defer or self.task.should_defer


class RouterHead:
    """Both classifiers over one prompt, encoded once per head.

    The two axes are independent by construction -- "summarise this contract"
    and "summarise this paper" share a task and differ in domain; "explain
    contract law" and "draft me a contract" share a domain and differ in task --
    so neither head can be derived from the other, and the pair carries more
    routing signal than either alone.
    """

    def __init__(self, domain_run: str | Path, task_run: str | Path, *,
                 task_defer_below: float = 0.0, **domain_kwargs) -> None:
        self.domain = DomainHead(domain_run, **domain_kwargs)
        self.task = TaskHead(task_run, defer_below=task_defer_below)

    def predict(self, prompt: str) -> RouterPrediction:
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str]) -> list[RouterPrediction]:
        prompts = list(prompts)
        if not prompts:
            return []
        return [RouterPrediction(domain=d, task=t) for d, t in
                zip(self.domain.predict_batch(prompts),
                    self.task.predict_batch(prompts), strict=True)]

    def feature_names(self) -> list[str]:
        """Column names for :meth:`vectorise`, resolved from one prediction."""
        return self.predict("x").feature_names()

    def vectorise(self, prompts: list[str]) -> tuple[np.ndarray, list[str]]:
        """``(n_prompts, n_features)`` matrix plus its column names.

        The handoff to a downstream difficulty or ability model: one row per
        prompt, columns fixed and named. Returning the names with the matrix
        rather than expecting the caller to fetch them separately is deliberate
        -- these two must not be able to drift apart.
        """
        preds = self.predict_batch(prompts)
        if not preds:
            names = self.feature_names()
            return np.zeros((0, len(names))), names
        rows = np.vstack([p.vector(t) for p, t in zip(preds, prompts, strict=True)])
        return rows, preds[0].feature_names()
