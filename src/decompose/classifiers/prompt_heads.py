"""The prompt-decomposition heads, as concrete :class:`Classifier` implementations.

``PromptDecomposer``'s docstring notes that no concrete ``Classifier`` exists
yet, so an empty decomposer produces empty ``PromptSignals``. These are those
classifiers: five trained heads, each emitting the signals it is good for and
nothing it is not.

Wiring them up is ordinary::

    decomposer = PromptDecomposer([
        SurfaceClassifier(),
        DomainClassifier("artifacts/v7/PROD_ensemble", merge_domains=True),
        TaskClassifier("artifacts/v11/PROD_task"),
        LengthClassifier("artifacts/length/multi_turn__bge-small-en-v1.5__surface_embedding"),
        AttributeClassifier("artifacts/attributes/bge-small-en-v1.5__surface_embedding"),
    ])
    signals = decomposer.extract_signals(ClassificationInput(prompt=prompt))

Two things these deliberately do not emit:

* **A difficulty signal.** Four independent feature families failed to predict
  it from prompt text, and the arena's own hardness rubric predicts the actual
  routing outcome at chance. It is a product of these signals and of model
  behaviour, computed downstream -- not a head.
* **A weak/strong recommendation.** Expected length, known *perfectly*, scores
  AUC 0.498 on whether the stronger model was needed. These signals describe a
  prompt; they do not pick a model. See ``docs/prompt_signals.md``.

Every classifier is versioned, and the version is the head's artifact
configuration rather than a hand-bumped integer, so a signal can be traced to
the run that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..signals import Signal
from .base import ClassificationInput, Classifier


class _HeadClassifier(Classifier):
    """Shared plumbing: load a run directory once, name signals consistently."""

    def __init__(self, run_dir: str | Path, *, name: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        if name:
            self.name = name
        self.version = self.run_dir.name

    def _signal(self, name: str, value: Any, confidence: float = 1.0,
                **metadata: Any) -> Signal:
        return Signal(name=name, value=value, confidence=confidence,
                      produced_by=f"{self.name}:{self.version}", metadata=metadata)


class SurfaceClassifier(Classifier):
    """23 deterministic features read straight off the prompt string.

    No model and no artifact, so this one is always available and cannot drift:
    "contains a code fence" means the same thing next year. Emitted as a single
    signal holding the whole dict rather than 23 separate ones, because they
    are consumed together as a vector.
    """

    name = "surface"
    version = "1"

    def classify(self, input: ClassificationInput) -> list[Signal]:
        from decompose.classifiers.prompt_decomposition.signals import extract

        features = extract(input.prompt).as_dict()
        return [Signal(name="surface", value=features, produced_by=f"{self.name}:{self.version}")]


class DomainClassifier(_HeadClassifier):
    """What the prompt is about: 8 domains, 0.923 top-1 on external labels.

    Emits the argmax, the calibrated distribution, and the adaptive shortlist --
    the shortlist is the product here, not the argmax: sized by probability
    mass it holds the true domain 0.985 of the time using 1.40 labels.
    """

    name = "domain"

    def __init__(self, run_dir: str | Path, **kwargs: Any) -> None:
        super().__init__(run_dir)
        from decompose.classifiers.prompt_decomposition import DomainHead

        self.head = DomainHead(run_dir, **kwargs)

    def classify(self, input: ClassificationInput) -> list[Signal]:
        p = self.head.predict(input.prompt)
        return [
            self._signal("domain", p.domain, p.confidence,
                         distribution=p.distribution, should_defer=p.should_defer),
            self._signal("domain_shortlist", p.shortlist, p.confidence),
        ]


class TaskClassifier(_HeadClassifier):
    """What the prompt asks to be done: 6 classes, 0.844 top-1.

    `distribution["media"]` is worth reading whatever the argmax says -- a media
    request is the one task that leaves the language models entirely, so it is a
    gate rather than a ranking, and the threshold belongs to the caller.
    """

    name = "task"

    def __init__(self, run_dir: str | Path, **kwargs: Any) -> None:
        super().__init__(run_dir)
        from decompose.classifiers.prompt_decomposition import TaskHead

        self.head = TaskHead(run_dir, **kwargs)

    def classify(self, input: ClassificationInput) -> list[Signal]:
        p = self.head.predict(input.prompt)
        return [self._signal("task", p.task, p.confidence,
                             distribution=p.distribution, should_defer=p.should_defer)]


class LengthClassifier(_HeadClassifier):
    """How much output the prompt implies.

    Emits a *bucket* and a rank-friendly estimate, plus the spread. Read it as
    an ordering, not a budget: a single estimate is within 2x of the truth about
    60% of the time, and the head's target is fixed at training time -- a
    single-turn head under-predicts a multi-turn total by 2.8x. The `variant`
    metadata says which target this artifact was fitted on.
    """

    name = "length"

    def __init__(self, run_dir: str | Path) -> None:
        super().__init__(run_dir)
        import json

        from decompose.classifiers.prompt_decomposition import LengthHead

        self.head = LengthHead(run_dir)
        meta = json.loads((self.run_dir / "length.json").read_text(encoding="utf-8"))
        self.variant = meta.get("variant", "single_turn")

    def classify(self, input: ClassificationInput) -> list[Signal]:
        p = self.head.predict(input.prompt)
        return [
            self._signal("expected_tokens", p.expected_tokens,
                         # A point estimate is worth roughly a coin flip at 2x;
                         # the confidence field should not imply otherwise.
                         confidence=0.6, bucket=p.bucket, sigma=p.sigma,
                         variant=self.variant, log_tokens=p.log_tokens),
            self._signal("length_bucket", p.bucket, confidence=0.6, variant=self.variant),
        ]


class AttributeClassifier(_HeadClassifier):
    """Eleven calibrated attributes: four gates and seven judged criteria.

    The gates are preconditions worth acting on -- `code` reaches 0.905
    precision at half recall. The criteria are the arena's LLM-judged difficulty
    dimensions, emitted as inputs to whatever combines them and never as a
    difficulty score: the rubric composed from them predicts the routing
    decision at chance.
    """

    name = "attributes"

    def __init__(self, run_dir: str | Path) -> None:
        super().__init__(run_dir)
        from decompose.classifiers.prompt_decomposition import AttributeHead

        self.head = AttributeHead(run_dir)

    def classify(self, input: ClassificationInput) -> list[Signal]:
        p = self.head.predict(input.prompt)
        return [
            self._signal("gates", p.gates, kind="precondition"),
            self._signal("criteria", p.criteria, kind="judged",
                         caveat="LLM-judged labels; chance-level for model choice"),
        ]
