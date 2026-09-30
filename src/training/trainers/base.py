"""``RouterModelTrainer``: the target's per-trainer contract
(``train(dataset, config) -> RouterModelArtifact``), and ``TrainingConfig``,
the normalized front door each trainer's ``train()`` translates into
whatever config shape its wrapped ``fit``/``fit_mlp_router`` function
actually needs -- not a replacement for those shapes.

One implementation per router model, registered under the same
``router_model_name`` a matching ``RouterModel``/``RouterModelFactory`` entry
uses (docs/architecture.md's own note on ``RouterModelTrainer`` -- concrete
variants aren't modeled in the diagram). ``NIRTTrainer``/``BaselineNIRTTrainer``/
``MLPRouterTrainer`` live next to the ``fit``/``fit_mlp_router`` functions
they wrap, under :mod:`training.nirt`/:mod:`training.trainers`.

Two known shape gaps between this and the target diagram, deliberately
worked around rather than resolved:

* **``dataset`` isn't a ``DatasetRef``.** The diagram's ``train(dataset:
  DatasetRef, ...)`` takes a versioned reference; every existing ``fit``
  function needs the actual loaded data object to run, and this codebase has
  no ``DatasetRef`` resolution step. ``dataset`` here is deliberately
  untyped: pass whatever the wrapped ``fit`` function already accepts (a
  ``(train_ds, val_ds)`` tuple, a ``TrainingData`` facade, or ``None`` to let
  it load from config).
* **``train()`` returns only an artifact, but metrics live elsewhere.** The
  diagram's ``RouterModelTrainer.train()`` returns just a
  ``RouterModelArtifact``; :class:`~training.pipeline.TrainingHarness` needs
  metrics too, to build a ``TrainingRun``. Each wrapper stashes the wrapped
  ``fit`` call's own untouched result object (``RunResult`` / ``BaselineRun``
  / the raw ``fit_mlp_router`` tuple) on ``self.last_result`` after
  ``train()`` returns -- the side-channel the harness reads from. The exact
  shape of ``last_result`` differs per trainer; it is not part of this base
  class's contract.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from router.models.artifacts import RouterModelArtifact


@dataclass
class TrainingConfig:
    run_name: str
    seed: int = 42
    max_epochs: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)


def translate_training_config(config: TrainingConfig) -> dict:
    """``config.hyperparameters`` is expected to already be shaped like
    ``training.nirt.train.fit``/``training.nirt.baseline.train.fit``'s own
    config (``{"model": {...}, "train": {...}, "data": {...}}``); the typed
    fields on ``config`` override/fill it in. Shared by ``NIRTTrainer`` and
    ``BaselineNIRTTrainer`` (previously two byte-identical private copies,
    TN-02) so a fix only needs to be made once.

    ``"train"``/``"data"`` may already be present as an explicit ``None``
    (a real YAML shape: ``train:`` with no children) -- ``dict.setdefault``
    does not replace an already-present ``None``, so the sub-section is
    normalized with ``... or {}`` before being written into, matching the
    defensive unwrap both wrapped ``fit()`` functions already do on their own
    config reads (TN-01)."""
    cfg = json.loads(json.dumps(config.hyperparameters))
    cfg["seed"] = config.seed
    if config.max_epochs is not None:
        cfg["train"] = cfg.get("train") or {}
        cfg["train"]["epochs"] = config.max_epochs
    if config.early_stopping_patience is not None:
        cfg["train"] = cfg.get("train") or {}
        cfg["train"]["patience"] = config.early_stopping_patience
    if config.preprocessing:
        cfg["data"] = cfg.get("data") or {}
        cfg["data"].update(config.preprocessing)
    return cfg


class RouterModelTrainer(abc.ABC):
    name: str = "trainer"
    version: str = "0"

    #: the wrapped fit() call's own result object from the most recent
    #: train() -- see the module docstring's second gap. None before train()
    #: is ever called.
    last_result: Optional[Any] = None

    @abc.abstractmethod
    def train(self, dataset: Any, config: TrainingConfig) -> RouterModelArtifact:
        raise NotImplementedError
