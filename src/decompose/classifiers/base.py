"""The :class:`Classifier` interface: a request in, some :class:`Signal`\\ s out."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..conversation import Conversation
from ..signals import Signal


@dataclass
class ClassificationInput:
    prompt: str
    conversation: Conversation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Classifier(abc.ABC):
    """A named, versioned signal producer. Subclasses implement :meth:`classify`
    only; :class:`~decompose.decomposer.PromptDecomposer` calls it for
    every :class:`ClassificationInput` and merges the results into
    :class:`~decompose.signals.PromptSignals`."""

    name: str = "classifier"
    version: str = "0"

    @abc.abstractmethod
    def classify(self, input: ClassificationInput) -> list[Signal]:
        raise NotImplementedError


class SimpleClassifier(Classifier):
    """Base for a classifier with no sub-classifiers of its own -- concrete
    classifiers (e.g. a language detector, a complexity heuristic) subclass
    this and implement :meth:`classify` directly. No such classifiers exist
    yet; this is scaffolding."""


class CompositeClassifier(Classifier):
    """Runs several sub-classifiers and concatenates their signals -- lets a
    :class:`~decompose.decomposer.PromptDecomposer` treat a bundle of
    related classifiers as one plugin."""

    def __init__(self, name: str, sub_classifiers: list[Classifier], *, version: str = "0"):
        self.name = name
        self.version = version
        self.sub_classifiers = list(sub_classifiers)

    def classify(self, input: ClassificationInput) -> list[Signal]:
        out: list[Signal] = []
        for c in self.sub_classifiers:
            out.extend(c.classify(input))
        return out
