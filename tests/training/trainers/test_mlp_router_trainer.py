"""``MLPRouterTrainer`` (training/trainers/mlp_router_trainer.py).

``fit_mlp_router`` had zero persistence and zero prior test coverage before
this trainer -- this is a from-scratch test, not adapted from an existing one.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import FakeTrainingData, make_split_obs

torch = pytest.importorskip("torch")

from router.models.artifacts import MLPArtifactPayload
from router.models.registry import RouterModelFactory
from router.nirt.baselines_infer import mlp_router_matrix
from router.routing.routers import MLPRouter
from training.trainers.base import TrainingConfig
from training.trainers.mlp_router_trainer import MLPRouterTrainer


def test_train_persists_and_reloads_a_working_router(tmp_path):
    d = FakeTrainingData(make_split_obs(seed=13), query_dim=16)

    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run", hyperparameters={"hidden": 16, "epochs": 3, "pathway": "irt"},
    )
    artifact = trainer.train(d, config)

    assert artifact.artifact_id == "mlp-run"
    assert artifact.router_model_name == "mlp"
    assert isinstance(artifact.payload, MLPArtifactPayload)
    assert (tmp_path / "mlp-run" / "model.pt").exists()

    model, model_ids = trainer.last_result
    assert artifact.payload.model_ids == list(model_ids)

    # MLPRouter.from_artifact (TT-01): the real reload path, not a hand-rebuild
    # that dodges the class's own constructor.
    router = MLPRouter.from_artifact(artifact, data=d)
    eval_ids = sorted(d.observations.query_id.unique())[:20]
    m_a = mlp_router_matrix(model, model_ids, d, eval_ids, pathway="irt")
    m_b = router.predict_scores(eval_ids)
    assert np.allclose(m_a.to_numpy(), m_b.to_numpy())

    # and the documented artifact-driven RouterModelFactory path this trainer
    # exists to enable -- must not raise TypeError("no from_artifact")
    factory = RouterModelFactory()
    factory.register("mlp", MLPRouter)
    router2 = factory.create(artifact, None)
    assert isinstance(router2, MLPRouter)


def test_train_persists_the_pathway_hyperparameters(tmp_path):
    """MLPROUTERTRAINER-001: pathway/query_pathway/query_features must survive
    into the saved artifact, not just fit_mlp_router's in-memory call."""
    d = FakeTrainingData(make_split_obs(seed=15), query_dim=16)
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run-pathway",
        hyperparameters={"hidden": 16, "epochs": 2, "pathway": "irt", "query_features": None},
    )
    artifact = trainer.train(d, config)
    assert artifact.payload.pathway == "irt"


def test_from_artifact_restores_the_trained_pathway(tmp_path):
    """MLPROUTER-001: from_artifact must not silently fall back to the
    constructor's pathway="irt" default when the artifact recorded a
    different one."""
    d = FakeTrainingData(make_split_obs(seed=16), query_dim=16)
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run-pathway-2",
        hyperparameters={"hidden": 16, "epochs": 2, "pathway": "bert",
                         "query_pathway": "bert-query"},
    )
    artifact = trainer.train(d, config)
    assert artifact.payload.pathway == "bert"

    router = MLPRouter.from_artifact(artifact, data=d)
    assert router._pathway == "bert"
    assert router._query_pathway == "bert-query"
    assert router._query_features is None


def test_typed_config_fields_override_hyperparameters(tmp_path, monkeypatch):
    import training.trainers.mlp_router_trainer as mod

    captured = {}
    real_fit = mod.fit_mlp_router

    def spy(data, **kw):
        captured.update(kw)
        return real_fit(data, **kw)

    monkeypatch.setattr(mod, "fit_mlp_router", spy)

    d = FakeTrainingData(make_split_obs(seed=14), query_dim=16)
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run-2", seed=3, max_epochs=2,
        hyperparameters={"hidden": 16, "epochs": 40, "pathway": "irt"},
    )
    trainer.train(d, config)
    assert captured["seed"] == 3
    assert captured["epochs"] == 2
    assert captured["hidden"] == 16


def test_config_seed_always_wins_over_a_conflicting_hyperparameters_seed(tmp_path, monkeypatch):
    """MLPROUTERTRAINER-003: seed precedence must match the sibling trainers
    (NIRTTrainer/BaselineNIRTTrainer) -- TrainingConfig.seed wins even when
    hyperparameters also sets "seed" to something else."""
    import training.trainers.mlp_router_trainer as mod

    captured = {}
    real_fit = mod.fit_mlp_router

    def spy(data, **kw):
        captured.update(kw)
        return real_fit(data, **kw)

    monkeypatch.setattr(mod, "fit_mlp_router", spy)

    d = FakeTrainingData(make_split_obs(seed=18), query_dim=16)
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run-seed", seed=123,
        hyperparameters={"hidden": 16, "epochs": 2, "seed": 999},
    )
    trainer.train(d, config)

    assert captured["seed"] == 123
