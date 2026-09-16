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
from router.nirt.baselines_infer import mlp_router_matrix
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

    from router.nirt.baselines_infer import build_mlp_router

    rebuilt = build_mlp_router(artifact.payload.in_dim, len(artifact.payload.model_ids),
                               artifact.payload.hidden, artifact.payload.dropout)
    rebuilt.load_state_dict(artifact.payload.state_dict)
    rebuilt.eval()

    eval_ids = sorted(d.observations.query_id.unique())[:20]
    m_a = mlp_router_matrix(model, model_ids, d, eval_ids, pathway="irt")
    m_b = mlp_router_matrix(rebuilt, artifact.payload.model_ids, d, eval_ids, pathway="irt")
    assert np.allclose(m_a.to_numpy(), m_b.to_numpy())


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
