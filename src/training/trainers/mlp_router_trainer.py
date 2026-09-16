"""``MLPRouterTrainer``: a thin :class:`~training.trainers.base.RouterModelTrainer`
wrapper around :func:`training.trainers.mlp_router.fit_mlp_router`.

**Not a pure wrap like the other two.** ``fit_mlp_router`` persists nothing
today -- no ``model.pt``/``run.json`` writer exists, and it had zero prior
test coverage. This is the one genuinely new-capability step among the three
trainer wraps: it adds persistence (via
:class:`~router.models.store.LocalArtifactStore`, using the new ``"mlp"``
format on :class:`~router.models.artifacts.MLPArtifactPayload`) rather than
adapting an existing convention.

``fit_mlp_router``'s own return contract (``(model, model_ids)``) is
unchanged; ``self.last_result`` holds it after ``train()`` returns.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from router.models.artifacts import ArtifactFormat, MLPArtifactPayload, RouterModelArtifact
from router.models.store import LocalArtifactStore

from .base import RouterModelTrainer, TrainingConfig
from .mlp_router import fit_mlp_router


class MLPRouterTrainer(RouterModelTrainer):
    name = "mlp"
    version = "0"

    def __init__(self, runs_dir: os.PathLike):
        #: required, unlike NIRTTrainer/BaselineNIRTTrainer's optional
        #: runs_dir -- persistence is the point of this wrapper existing.
        self.runs_dir = runs_dir
        self.last_result: Optional[tuple] = None

    def train(self, dataset: Any, config: TrainingConfig) -> RouterModelArtifact:
        """``dataset`` is the ``data`` object ``fit_mlp_router`` needs (a
        ``TrainingData`` facade or equivalent)."""
        kw = dict(config.hyperparameters)
        kw.setdefault("seed", config.seed)
        if config.max_epochs is not None:
            kw["epochs"] = config.max_epochs
        if config.early_stopping_patience is not None:
            kw["patience"] = config.early_stopping_patience

        model, model_ids = fit_mlp_router(dataset, **kw)
        self.last_result = (model, model_ids)

        payload = MLPArtifactPayload(
            state_dict=model.state_dict(),
            model_ids=list(model_ids),
            in_dim=int(model[0].in_features),
            hidden=int(model[0].out_features),
            dropout=float(model[2].p),
        )
        store = LocalArtifactStore(self.runs_dir)
        store.save(RouterModelArtifact(
            artifact_id=config.run_name,
            payload=payload,
            content_hash="",
            format=ArtifactFormat.PICKLE,
            router_model_name="mlp",
            supported_model_ids=list(model_ids),
        ))
        return store.load(config.run_name)
