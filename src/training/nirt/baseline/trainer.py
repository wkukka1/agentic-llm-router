"""``BaselineNIRTTrainer``: a thin :class:`~training.trainers.base.RouterModelTrainer`
wrapper around :func:`training.nirt.baseline.train.fit`.

Unlike :class:`~training.nirt.trainer.NIRTTrainer`, this does **not**
round-trip through :class:`~router.models.store.ArtifactStore`: there is no
serving-side loader for the baseline checkpoint format --
``training.nirt.baseline.checkpoint`` lives under ``training``, which
``router`` must not import (the "serving never imports training" contract) --
so :class:`~router.models.artifacts.BaselineArtifactPayload` is built
directly from the in-memory ``fit()`` result instead of being read back off
disk. ``fit()``'s own return contract (:class:`BaselineRun`) is completely
unchanged; this wrapper is an *additional* entry point for the typed
``RouterModelArtifact`` shape.

``BaselineRun.train_imbalance``/``dims`` have no home in a typed
``TrainingMetrics``/``ValidationMetrics`` view -- they stay reachable via
``self.last_result`` rather than being forced into a lossy typed mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from router.models.artifacts import ArtifactFormat, BaselineArtifactPayload, RouterModelArtifact

from ...trainers.base import RouterModelTrainer, TrainingConfig, translate_training_config
from .train import BaselineRun, fit


class BaselineNIRTTrainer(RouterModelTrainer):
    name = "baseline"
    version = "0"

    def __init__(self, runs_dir: Optional[str] = None):
        self.runs_dir = runs_dir
        self.last_result: Optional[BaselineRun] = None

    def train(self, dataset: Any, config: TrainingConfig) -> RouterModelArtifact:
        """``dataset`` is a ``(train_arrays, val_arrays)`` tuple (``fit``'s
        ``arrays=``), or ``None`` to let ``fit`` build them from config."""
        phase1_cfg = translate_training_config(config)
        out_dir = str(Path(self.runs_dir) / config.run_name) if self.runs_dir else None
        run = fit(phase1_cfg, arrays=dataset, out_dir=out_dir, save=self.runs_dir is not None,
                  verbose=False)
        self.last_result = run

        payload = BaselineArtifactPayload(
            state_dict=run.model.state_dict(),
            model_cfg=run.config.get("model", {}),
            model_index=run.model_index,
            query_dim=run.dims["query_dim"],
            profile_dim=run.dims["profile_dim"] or run.dims["query_dim"],
            relevance_dim=run.dims["relevance_dim"],
        )
        return RouterModelArtifact(
            artifact_id=config.run_name,
            payload=payload,
            content_hash="",
            format=ArtifactFormat.PICKLE,
            router_model_name="baseline",
            supported_model_ids=list(run.model_index),
            hyperparameters=run.config,
        )
