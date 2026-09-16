"""``NIRTTrainer``: a thin :class:`~training.trainers.base.RouterModelTrainer`
wrapper around :func:`training.nirt.train.fit`.

Lowest-risk of the three trainer wraps: ``fit(..., save=True)`` already
writes exactly what :class:`~router.models.store.LocalArtifactStore` expects
(``docs/architecture.md``'s ``model.pt``/``run.json`` convention), so this
wrapper needs no new persistence code -- it calls the store right after
``fit()`` returns. ``fit()``'s own return contract (:class:`RunResult`) is
completely unchanged; this wrapper is an *additional* entry point for
callers who want the typed ``RouterModelArtifact`` shape, not a replacement.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from router.models.artifacts import RouterModelArtifact
from router.models.store import LocalArtifactStore

from ..trainers.base import RouterModelTrainer, TrainingConfig
from .train import RunResult, fit


def _nirt_cfg_from_config(config: TrainingConfig) -> dict:
    """``config.hyperparameters`` is expected to already be shaped like
    ``training.nirt.train.fit``'s ``nirt_cfg`` (``{"model": {...}, "train": {...},
    "data": {...}}``); the typed fields on ``config`` override/fill it in."""
    cfg = json.loads(json.dumps(config.hyperparameters))
    cfg["seed"] = config.seed
    if config.max_epochs is not None:
        cfg.setdefault("train", {})["epochs"] = config.max_epochs
    if config.early_stopping_patience is not None:
        cfg.setdefault("train", {})["patience"] = config.early_stopping_patience
    if config.preprocessing:
        cfg.setdefault("data", {}).update(config.preprocessing)
    return cfg


class NIRTTrainer(RouterModelTrainer):
    name = "nirt"
    version = "0"

    def __init__(self, runs_dir: Optional[os.PathLike] = None):
        self.runs_dir = runs_dir
        self.last_result: Optional[RunResult] = None

    def train(self, dataset: Any, config: TrainingConfig) -> RouterModelArtifact:
        """``dataset`` is a ``(train_ds, val_ds)`` tuple (``fit``'s ``datasets=``),
        or ``None`` to let ``fit`` load from the phase-0 config."""
        nirt_cfg = _nirt_cfg_from_config(config)
        self.last_result = fit(
            nirt_cfg, datasets=dataset, name=config.run_name,
            runs_dir=self.runs_dir, save=True, verbose=False,
        )
        # LocalArtifactStore needs a concrete path; resolve runs_dir=None the
        # exact same way fit() itself just did (_resolve_runs_dir ->
        # resolve_path), including honoring a nirt_cfg["runs_dir"] override.
        runs_dir = self.runs_dir
        if runs_dir is None:
            from router.config import DEFAULT_NIRT_RUNS_DIR, resolve_path

            runs_dir = resolve_path(nirt_cfg.get("runs_dir", DEFAULT_NIRT_RUNS_DIR))
        return LocalArtifactStore(runs_dir).load(config.run_name)
