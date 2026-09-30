"""``PromptDecomposer``: run every registered :class:`Classifier` over a
request and merge the results into one :class:`PromptSignals`.

Wired into :class:`~router.router.Router` (real, working glue) but
not into :mod:`router.agentic` -- the live agentic flow decomposes *compound
requests into sub-prompts* (:mod:`router.agentic.decompose`, a different job:
splitting one request into several) rather than *extracting signals from one
prompt*, which is this class's job.

Five real :class:`Classifier` implementations exist --
:mod:`decompose.classifiers.prompt_heads` -- and :func:`default_prompt_decomposer`
below builds a live decomposer from whichever of them have a trained artifact
on hand. ``PromptDecomposer()`` with no arguments still constructs an empty,
always-safe decomposer (no artifact paths to fail on) -- that remains the
default a caller gets without opting into the factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

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


def default_prompt_decomposer(
    artifact_paths: Optional[Mapping[str, str | Path]] = None,
    *,
    embedder: Optional[Embedder] = None,
) -> PromptDecomposer:
    """Build a :class:`PromptDecomposer` from the real prompt-heads classifiers.

    ``SurfaceClassifier`` needs no artifact and is always included.
    ``artifact_paths`` may supply any of ``"domain"``, ``"task"``, ``"length"``,
    ``"attributes"`` -- each maps to a trained run directory and constructs the
    matching classifier from :mod:`decompose.classifiers.prompt_heads`; a key
    left out simply means that signal isn't produced. This lets a caller with
    only some of the five artifacts on hand still get a working decomposer.
    """
    from .classifiers.prompt_heads import (
        AttributeClassifier,
        DomainClassifier,
        LengthClassifier,
        SurfaceClassifier,
        TaskClassifier,
    )

    paths = dict(artifact_paths or {})
    unknown = set(paths) - {"domain", "task", "length", "attributes"}
    if unknown:
        raise ValueError(
            f"default_prompt_decomposer: unknown artifact_paths key(s) {sorted(unknown)}; "
            'expected any of "domain", "task", "length", "attributes"'
        )
    classifiers: list[Classifier] = [SurfaceClassifier()]
    if "domain" in paths:
        classifiers.append(DomainClassifier(paths["domain"], merge_domains=True))
    if "task" in paths:
        classifiers.append(TaskClassifier(paths["task"]))
    if "length" in paths:
        classifiers.append(LengthClassifier(paths["length"]))
    if "attributes" in paths:
        classifiers.append(AttributeClassifier(paths["attributes"]))
    return PromptDecomposer(classifiers, embedder=embedder)
