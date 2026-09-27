"""``TrainingHarness``: formalizes what's today an ad hoc script per trainer.

``RouterModelTrainer`` + ``TrainingHarness`` give every trainer the same
shape (``train(dataset, config) -> RouterModelArtifact``, wrapped into a
``TrainingRun``), so a new router kind doesn't need a new pipeline script
(``docs/architecture.md``).

**Metrics translation is necessarily best-effort.** Each trainer's
``last_result`` (see :mod:`training.trainers.base`) has a different shape --
``RunResult.val_metrics`` / ``BaselineRun.val_metrics`` are dicts keyed by
``training.nirt.metrics.prediction_metrics``'s vocabulary (``bce``,
``accuracy``, ``auc``, ``spearman_r``); ``MLPRouterTrainer.last_result`` is a
bare ``(model, model_ids)`` tuple with no metrics at all. ``TrainingMetrics``/
``ValidationMetrics`` below are populated from whatever keys are present and
left ``None`` otherwise -- not every trainer has an equivalent of every field
(none compute ``regret`` today; that lives in ``evaluation.routing.oracle``,
a separate, label-needing concern). ``None`` means "not reported by this
trainer", never "reported as zero" (TRAININGHARNESS-003).

**No separate save step here.** The diagram shows ``TrainingHarness ..>
ArtifactStore : saves to``, but every current trainer already persists
internally -- ``NIRTTrainer``/``MLPRouterTrainer`` via
:class:`~router.models.store.LocalArtifactStore`, ``BaselineNIRTTrainer`` via
its own checkpoint writer (which ``router``-side code must not import, see
that trainer's docstring). Adding a harness-level fallback save would need to
know which artifact/payload shapes are safe to persist again, which the
harness has no way to determine generically -- left undone rather than
built untested.

**Provenance: ``created_at`` is harness-stamped, ``training_dataset_version``
is not populated by anything today.** No trainer sets either field on the
``RouterModelArtifact`` it returns (``NIRTTrainer`` round-trips through
``fit()``'s own ``run.json``, which never carries them; ``BaselineNIRTTrainer``/
``MLPRouterTrainer`` construct the artifact directly with neither kwarg). Of
the two, ``created_at`` -- "training happened now" -- is generically knowable
regardless of trainer, so :meth:`TrainingHarness.train` stamps it on the
artifact whenever the trainer left it blank, once, for all three trainers.
``training_dataset_version`` is not: it names a ``DatasetRef`` this codebase
doesn't have yet (see :mod:`training.trainers.base`'s docstring), and
``dataset`` here is deliberately untyped (``Any``), so the harness has
nothing to derive a version from. It stays ``""`` until a trainer is given a
real dataset identity to report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from router.models.artifacts import RouterModelArtifact

from .trainers.base import RouterModelTrainer, TrainingConfig


@dataclass
class TrainingMetrics:
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    bce: Optional[float] = None
    auc: Optional[float] = None
    spearman: Optional[float] = None
    regret: Optional[float] = None
    training_time: float = 0.0


@dataclass
class ValidationMetrics:
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    bce: Optional[float] = None
    auc: Optional[float] = None
    spearman: Optional[float] = None
    regret: Optional[float] = None
    selected_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    run_id: str
    trainer: RouterModelTrainer
    dataset: Any
    config: TrainingConfig
    output_path: str = ""


@dataclass
class TrainingRun:
    run_id: str
    trainer_name: str
    trainer_version: str
    artifact: RouterModelArtifact
    training_metrics: TrainingMetrics
    validation_metrics: ValidationMetrics
    config: TrainingConfig
    dataset_version: str = ""
    created_at: str = ""


def _metrics_kwargs(d: dict[str, Any]) -> dict[str, float]:
    kw: dict[str, float] = {}
    for target, sources in (
        ("loss", ("loss", "train_loss")),
        ("accuracy", ("accuracy",)),
        ("bce", ("bce",)),
        ("auc", ("auc",)),
        ("spearman", ("spearman_r", "spearman")),
        ("regret", ("regret",)),
    ):
        for src in sources:
            if src in d:
                kw[target] = float(d[src])
                break
    return kw


def _extract_metrics(last_result: Any) -> tuple[TrainingMetrics, ValidationMetrics]:
    """Best-effort: works for anything exposing ``val_metrics``/``history``
    dicts (``RunResult``, ``BaselineRun``); leaves every field ``None``
    ("not reported") for anything else (e.g. ``MLPRouterTrainer``'s bare tuple).

    ``history`` rows only ever carry ``train_loss`` (matched onto
    ``TrainingMetrics.loss``) -- neither wrapped trainer's per-epoch loop
    computes train-split accuracy/bce/auc/spearman, so those four fields are
    left ``None`` ("not reported") regardless of trainer; only the validation split
    (``val_metrics``) has them."""
    val_metrics = getattr(last_result, "val_metrics", None) or {}
    validation = ValidationMetrics(**_metrics_kwargs(val_metrics))
    history = getattr(last_result, "history", None) or []
    train_dict = history[-1] if history else {}
    training = TrainingMetrics(**_metrics_kwargs(train_dict))
    return training, validation


class TrainingHarness:
    def run(self, job: TrainingJob) -> TrainingRun:
        return self.train(job.trainer, job.dataset, job.config)

    def train(self, trainer: RouterModelTrainer, dataset: Any, config: TrainingConfig) -> TrainingRun:
        t0 = time.time()
        artifact = trainer.train(dataset, config)
        elapsed = time.time() - t0
        if not artifact.created_at:
            artifact.created_at = datetime.now(timezone.utc).isoformat()
        training, validation = _extract_metrics(trainer.last_result)
        training.training_time = elapsed
        return TrainingRun(
            run_id=config.run_name,
            trainer_name=trainer.name,
            trainer_version=trainer.version,
            artifact=artifact,
            training_metrics=training,
            validation_metrics=validation,
            config=config,
            dataset_version=artifact.training_dataset_version,
            created_at=artifact.created_at,
        )
