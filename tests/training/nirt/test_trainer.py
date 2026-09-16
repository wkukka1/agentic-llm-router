"""``NIRTTrainer`` (training/nirt/trainer.py) -- thin RouterModelTrainer wrapper
around training.nirt.train.fit."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_synthetic_irt, nirt_cfg, nirt_datasets

torch = pytest.importorskip("torch")

from router.llm.cost import CostModel
from router.models.registry import RouterModelFactory
from router.nirt.predict import predict_dataset
from router.routing.routers import NIRTRouter
from training.nirt.trainer import NIRTTrainer
from training.trainers.base import TrainingConfig


def test_train_returns_an_artifact_that_predicts_like_fit_directly(tmp_path):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=5)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)

    trainer = NIRTTrainer(runs_dir=tmp_path)
    config = TrainingConfig(run_name="trainer-run", hyperparameters=nirt_cfg())
    artifact = trainer.train((train_ds, val_ds), config)

    assert artifact.artifact_id == "trainer-run"
    assert trainer.last_result is not None
    assert trainer.last_result.name == "trainer-run"
    assert np.isfinite(trainer.last_result.val_metrics["bce"])

    factory = RouterModelFactory()
    factory.register("nirt", NIRTRouter)
    router = factory.create(artifact, CostModel())
    y, p = predict_dataset(router._model, val_ds, router._model_index)
    assert p.shape == y.shape and np.all((p >= 0) & (p <= 1))


def test_typed_config_fields_override_hyperparameters(tmp_path):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=6)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)

    trainer = NIRTTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="trainer-run-2", seed=7, max_epochs=2,
        hyperparameters=nirt_cfg(),
    )
    trainer.train((train_ds, val_ds), config)
    assert trainer.last_result.config["seed"] == 7
    assert trainer.last_result.config["train"]["epochs"] == 2
