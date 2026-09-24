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

import re
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


#: Verbs that name a deliverable. Two distinct ones joined by a sequence marker
#: is the signature of a compound request ("summarise X, then implement Y").
_TASK_VERBS = (
    "summarise", "summarize", "implement", "analyse", "analyze", "write", "compare",
    "explain", "translate", "list", "review", "design", "compute", "calculate",
    "draft", "refactor", "debug", "plan", "outline", "rank", "classify", "extract",
    "generate", "build", "create", "evaluate", "prove", "derive", "benchmark",
)
#: Sequence markers, matched after a clause boundary so ", then" counts as well
#: as " and then". The hand-written list in `router.agentic.triage` missed the
#: comma form, and routed "Summarise X, then implement Y" as a single shot.
_SEQUENCE_RE = re.compile(
    r"(?:^|[,;.]\s*|\band\s+|\bthen\s+)(?:then|next|after\s+that|afterwards|"
    r"followed\s+by|finally|lastly|as\s+well\s+as|also)\b"
    r"|\bstep[-\s]by[-\s]step\b|\bfirst,", re.I)
_VERB_RE = re.compile(r"\b(" + "|".join(_TASK_VERBS) + r")\b", re.I)
_ENUMERATED_RE = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s", re.MULTILINE)


class StructureClassifier(Classifier):
    """Is this one request or several?

    The decision `router.agentic.triage` makes before anything else: answer in
    one shot, or decompose and route each part. It is a *structural* property
    of the text -- how many deliverables are asked for -- not a semantic one,
    which is why a classifier can do it when nothing we tried could predict
    difficulty from a prompt.

    Emits ``multi_part`` (bool) with the evidence attached, and ``part_count``,
    a lower bound on the number of deliverables. Deliberately conservative:
    each piece of evidence is a thing a reader would point at.
    """

    name = "structure"
    version = "1"

    def __init__(self, *, max_chars_single: int = 600, min_questions: int = 3) -> None:
        self.max_chars_single = int(max_chars_single)
        self.min_questions = int(min_questions)

    def evidence(self, prompt: str) -> dict:
        """Every reason to think this is compound, named."""
        verbs = {m.group(1).lower() for m in _VERB_RE.finditer(prompt)}
        return {
            "distinct_task_verbs": sorted(verbs),
            "sequence_markers": [m.group(0).strip() for m in _SEQUENCE_RE.finditer(prompt)],
            "enumerated_items": len(_ENUMERATED_RE.findall(prompt)),
            "question_marks": prompt.count("?"),
            "chars": len(prompt),
        }

    def classify(self, input: ClassificationInput) -> list[Signal]:
        e = self.evidence(input.prompt)
        reasons = []
        if e["enumerated_items"] >= 2:
            reasons.append(f"{e['enumerated_items']} enumerated items")
        if e["sequence_markers"]:
            # A marker alone is enough. Requiring two known task verbs as well
            # made this *narrower* than the list it replaces -- and the verb
            # list is exactly the brittleness being removed. Triage pays little
            # for a false "decompose" and loses the whole point on a false
            # "single", so the asymmetry favours firing.
            verbs = e["distinct_task_verbs"]
            detail = f" joining {len(verbs)} task verbs" if len(verbs) >= 2 else ""
            reasons.append(f"sequence marker {e['sequence_markers'][0]!r}{detail}")
        if e["question_marks"] >= self.min_questions:
            reasons.append(f"{e['question_marks']} questions")
        if e["chars"] > self.max_chars_single:
            reasons.append(f"{e['chars']} characters")
        parts = max(e["enumerated_items"], len(e["distinct_task_verbs"]) if reasons else 1, 1)
        produced_by = f"{self.name}:{self.version}"
        return [
            Signal(name="multi_part", value=bool(reasons), confidence=1.0,
                   produced_by=produced_by,
                   metadata={"reasons": reasons, **e}),
            Signal(name="part_count", value=int(parts), confidence=1.0,
                   produced_by=produced_by),
        ]
