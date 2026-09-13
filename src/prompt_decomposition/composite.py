"""Every head over one prompt, and the single vector the next stage consumes.

Domain and task are independent by construction -- "summarise this contract" and
"summarise this paper" share a task and differ in domain; "explain contract law"
and "draft me a contract" share a domain and differ in task -- so neither is
recoverable from the other. Length and the attribute heads answer different
questions again: how big the job is, and what it needs.

``vectorise`` is the handoff. It returns the matrix *and* its column names
together, because a feature vector whose column meanings live somewhere else is
one refactor away from being silently wrong.

**Difficulty is not here and will not be.** Four feature families failed to
predict it from prompt text, and the arena's own hardness rubric predicts the
actual routing outcome at chance. It is a product of these signals and of
model behaviour, computed downstream of this vector rather than inside it.

The optional heads are optional on purpose: a deployment that has not trained
them still gets the 24-column core, and the column names say which of them are
present rather than leaving a caller to infer it from the width.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prompt_decomposition.attribute_classifier.head import AttributeHead, AttributePrediction
from prompt_decomposition.domain_classifier.head import DomainHead, DomainPrediction
from prompt_decomposition.length_estimator.head import LengthHead, LengthPrediction
from prompt_decomposition.signals import SURFACE_FEATURES, extract
from prompt_decomposition.task_classifier.head import TaskHead, TaskPrediction

#: Surface features already carried by the core block, so the surface block
#: leaves them out rather than shipping two columns of identical numbers.
_SURFACE_DUPLICATES = ("log_chars", "log_words")
SURFACE_BLOCK: tuple[str, ...] = tuple(
    f for f in SURFACE_FEATURES if f not in _SURFACE_DUPLICATES)


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats, 0 when certain and log(k) when uniform."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


@dataclass(slots=True)
class RouterPrediction:
    """Both axes for one prompt."""

    domain: DomainPrediction
    task: TaskPrediction
    #: Present only when the router was built with the corresponding run
    #: directory. Absent heads drop their columns rather than filling them with
    #: zeros, which a downstream model cannot distinguish from a real zero.
    length: LengthPrediction | None = None
    attributes: AttributePrediction | None = None
    #: The free deterministic features, carried so the whole handoff is one
    #: call. ``None`` when the router was built with ``surface=False``.
    surface: dict[str, float] | None = None

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

    def _extras(self) -> dict[str, float]:
        """The optional blocks, in a fixed order: length, attributes, surface.

        Each block is present only if its head was loaded. Appending rather
        than interleaving means adding a head extends the vector on the right
        and leaves every existing column index where it was.
        """
        out: dict[str, float] = {}
        if self.length is not None:
            # The log estimate, not the token count: the count spans four
            # orders of magnitude and would dominate any unscaled model.
            out["length.log_tokens"] = float(self.length.log_tokens)
        if self.attributes is not None:
            out.update({f"attr.{k}": float(v)
                        for k, v in sorted(self.attributes.probabilities.items())})
        if self.surface is not None:
            out.update({f"surface.{k}": float(self.surface[k]) for k in SURFACE_BLOCK})
        return out

    def feature_names(self) -> list[str]:
        """Names for :meth:`vector`, same order, same length.

        Held next to the vector rather than written down elsewhere, because a
        feature vector whose column meanings live in a different file is one
        refactor away from being silently wrong.
        """
        names = [f"domain.{d}" for d in sorted(self.domain.distribution)]
        names += [f"task.{t}" for t in sorted(self.task.distribution)]
        return names + list(self._scalars(None)) + list(self._extras())

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

        Width is ``len(self.feature_names())`` and depends on which heads were
        loaded: 8 domains (10 unmerged) + 6 tasks + 9 scalars is the core, then
        +1 for length, +11 for the attributes, +21 for the surface block (23
        features less the two the core already carries). Stable for a given
        configuration; changing ``merge_domains`` or loading another head
        changes it, which is why the names travel alongside.

        Blocks are appended, never interleaved, so adding a head extends the
        vector on the right and leaves every existing column index alone.
        """
        dom, tsk = self._distributions()
        scalars = np.array(list(self._scalars(prompt).values()), dtype=float)
        extras = np.array(list(self._extras().values()), dtype=float)
        return np.concatenate([dom, tsk, scalars, extras])

    @property
    def should_defer(self) -> bool:
        """True if *either* axis is unsure. Deliberately pessimistic: a
        confident domain paired with an unsure task is not a confident route."""
        return self.domain.should_defer or self.task.should_defer


class RouterHead:
    """Every head over one prompt, and one vector out.

    Domain and task are always present; length and the attribute heads are
    loaded only if their run directories are given, so a deployment that has
    not trained them still gets the 24-column core. Heads that share an encoder
    share the pass.
    """

    def __init__(self, domain_run: str | Path, task_run: str | Path, *,
                 length_run: str | Path | None = None,
                 attribute_run: str | Path | None = None,
                 surface: bool = True,
                 task_defer_below: float = 0.0, **domain_kwargs) -> None:
        self.domain = DomainHead(domain_run, **domain_kwargs)
        self.task = TaskHead(task_run, defer_below=task_defer_below)
        self.length = LengthHead(length_run) if length_run is not None else None
        self.attributes = AttributeHead(attribute_run) if attribute_run is not None else None
        self.surface = surface

    def predict(self, prompt: str) -> RouterPrediction:
        return self.predict_batch([prompt])[0]

    def _shared_embeddings(self, prompts: list[str]) -> dict[str, np.ndarray]:
        """Encode once per distinct encoder, not once per head.

        The length and attribute heads are linear probes; when they sit on the
        same encoder, running it twice doubles the only expensive part of the
        call for nothing. Keyed by encoder *name* because two heads fitted on
        different encoders cannot share vectors -- serving one the other's
        output is silent and wrong.
        """
        wanted = {h.encoder_model for h in (self.length, self.attributes)
                  if h is not None and h.encoder_model and "embedding" in h.feature_set}
        if len(wanted) < 1:
            return {}
        pool = {}
        for head in (self.length, self.attributes):
            if head is None or head.encoder_model not in wanted:
                continue
            if head.encoder_model not in pool:
                pool[head.encoder_model] = head.encoder.encode(prompts)
        return pool

    def predict_batch(self, prompts: list[str]) -> list[RouterPrediction]:
        prompts = list(prompts)
        if not prompts:
            return []
        pool = self._shared_embeddings(prompts)
        blank: list[None] = [None] * len(prompts)
        lengths = (self.length.predict_batch(prompts, pool.get(self.length.encoder_model))
                   if self.length is not None else blank)
        attributes = (self.attributes.predict_batch(
            prompts, pool.get(self.attributes.encoder_model))
            if self.attributes is not None else blank)
        surfaces = ([extract(p).as_dict() for p in prompts] if self.surface else blank)
        return [RouterPrediction(domain=d, task=t, length=le, attributes=at, surface=su)
                for d, t, le, at, su in zip(
                    self.domain.predict_batch(prompts), self.task.predict_batch(prompts),
                    lengths, attributes, surfaces, strict=True)]

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
