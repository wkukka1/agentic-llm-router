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


def test_train_returns_an_artifact_that_loads_through_the_factory_and_predicts(tmp_path):
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
    _, p_loaded = predict_dataset(router._model, val_ds, router._model_index)
    _, p_fitted = predict_dataset(trainer.last_result.model, val_ds, trainer.last_result.model_index)
    np.testing.assert_allclose(p_loaded, p_fitted, atol=1e-6)


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


def test_runs_dir_none_falls_back_to_the_default_runs_dir(tmp_path, monkeypatch):
    """A bare LocalArtifactStore(None) raises (Path(None) is invalid) -- train()
    must resolve the same default fit()/load_run already fall back to,
    rather than crashing whenever no runs_dir is given."""
    import router.config as router_config

    monkeypatch.setattr(router_config, "REPO_ROOT", tmp_path)

    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=15)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)

    trainer = NIRTTrainer(runs_dir=None)
    config = TrainingConfig(run_name="trainer-default-dir", hyperparameters=nirt_cfg())
    artifact = trainer.train((train_ds, val_ds), config)

    expected = tmp_path / router_config.DEFAULT_NIRT_RUNS_DIR / "trainer-default-dir"
    assert expected.exists()
    assert artifact.artifact_id == "trainer-default-dir"
