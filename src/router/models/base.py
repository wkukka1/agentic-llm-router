"""Common interface every domain classifier implements.

The training harness only ever sees this surface, which is what lets a TF-IDF
baseline and a fine-tuned transformer be compared by the same runner and, later,
be swapped behind the router without touching the router.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class DomainClassifier(ABC):
    """Predicts a distribution over :data:`~router.data.taxonomy.DOMAIN_LABELS`.

    Probabilities, not argmax, are the contract: the router needs calibrated
    scores to decide when a prompt is ambiguous enough to warrant a stronger
    model or a clarifying follow-up.
    """

    #: Populated by :meth:`fit`; index position defines the column order of
    #: :meth:`predict_proba`.
    labels: list[str]

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.labels = []

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def fit(self, train_texts: list[str], train_labels: list[str],
            val_texts: list[str] | None = None, val_labels: list[str] | None = None) -> None:
        """Train in place. ``val_*`` is for early stopping/model selection only."""

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n_texts, n_labels)`` array of probabilities."""

    def predict(self, texts: list[str]) -> list[str]:
        proba = self.predict_proba(texts)
        return [self.labels[i] for i in proba.argmax(axis=1)]

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist to a directory."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore from a directory written by :meth:`save`."""

    def size_bytes(self) -> int:
        """Approximate on-disk footprint; reported alongside accuracy."""
        return 0
