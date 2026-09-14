"""``PromptDecomposer``: run every registered :class:`Classifier` over a
request and merge the results into one :class:`PromptSignals`.

Wired into :class:`~router.router.RoutingPipeline` (real, working glue) but
not into :mod:`router.agentic` -- the live agentic flow decomposes *compound
requests into sub-prompts* (:mod:`router.agentic.decompose`, a different job:
splitting one request into several) rather than *extracting signals from one
prompt*, which is this class's job. No concrete :class:`Classifier` exists
yet, so an empty :class:`PromptDecomposer` produces an empty
:class:`PromptSignals`.
"""

from __future__ import annotations

from typing import Optional

from .classifiers.base import ClassificationInput, Classifier
from .embedders.base import Embedder
from .signals import PromptSignals


class PromptDecomposer:
    name: str = "decomposer"
    version: str = "0"

    def __init__(self, classifiers: Optional[list[Classifier]] = None, *,
                 embedder: Optional[Embedder] = None):
        self.classifiers: list[Classifier] = list(classifiers or [])
        self.embedder = embedder

    def add_classifier(self, classifier: Classifier) -> None:
        self.classifiers.append(classifier)

    def remove_classifier(self, name: str) -> None:
        self.classifiers = [c for c in self.classifiers if c.name != name]

    def extract_signals(self, input: ClassificationInput) -> PromptSignals:
        out = PromptSignals()
        for classifier in self.classifiers:
            for signal in classifier.classify(input):
                out.add(signal)
        if self.embedder is not None:
            out.add_embedding(self.embedder.name, self.embedder.embed(input.prompt))
        return out
