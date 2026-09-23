"""``BaselineNIRTTrainer`` (training/nirt/baseline/trainer.py)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import baseline_cfg

torch = pytest.importorskip("torch")

from router.models.artifacts import BaselineArtifactPayload
from training.nirt.baseline.model import build_baseline_model
from training.nirt.baseline.synthetic import make_synthetic, to_arrays
from training.nirt.baseline.trainer import BaselineNIRTTrainer
from training.trainers.base import TrainingConfig


def _tiny_arrays():
    syn = make_synthetic(n_queries=300, n_models=6, K=2, seed=1)
    return to_arrays(syn, seed=1)


def test_train_returns_an_artifact_matching_the_in_memory_fit_result():
    tr, va = _tiny_arrays()
    trainer = BaselineNIRTTrainer()
    config = TrainingConfig(run_name="baseline-run", hyperparameters=baseline_cfg(k=2, epochs=12))
    artifact = trainer.train((tr, va), config)

    assert artifact.artifact_id == "baseline-run"
    assert artifact.router_model_name == "baseline"
    assert isinstance(artifact.payload, BaselineArtifactPayload)
    assert trainer.last_result is not None
    assert trainer.last_result.model is not None

    rebuilt = build_baseline_model(
        artifact.payload.model_cfg,
        n_models=len(artifact.payload.model_index),
        query_dim=artifact.payload.query_dim,
        profile_dim=artifact.payload.profile_dim,
        relevance_dim=artifact.payload.relevance_dim,
    )
    rebuilt.load_state_dict(artifact.payload.state_dict)
    rebuilt.eval()
    for k, v in rebuilt.state_dict().items():
        assert torch.equal(v, trainer.last_result.model.state_dict()[k])


def test_typed_config_fields_override_hyperparameters():
    tr, va = _tiny_arrays()
    trainer = BaselineNIRTTrainer()
    config = TrainingConfig(
        run_name="baseline-run-2", seed=9, max_epochs=5,
        hyperparameters=baseline_cfg(k=2, epochs=12),
    )
    trainer.train((tr, va), config)
    assert trainer.last_result.config["seed"] == 9
    assert trainer.last_result.config["train"]["epochs"] == 5


def test_train_does_not_write_to_disk_without_a_runs_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr, va = _tiny_arrays()
    trainer = BaselineNIRTTrainer()  # no runs_dir
    config = TrainingConfig(run_name="baseline-run-3", hyperparameters=baseline_cfg(k=2, epochs=8))
    trainer.train((tr, va), config)
    assert trainer.last_result.path is None
    assert not any(tmp_path.rglob("*.pt"))


def test_train_writes_to_disk_when_a_runs_dir_is_given(tmp_path):
    tr, va = _tiny_arrays()
    trainer = BaselineNIRTTrainer(runs_dir=tmp_path)
    config = TrainingConfig(run_name="baseline-run-4", hyperparameters=baseline_cfg(k=2, epochs=8))
    trainer.train((tr, va), config)
    assert (tmp_path / "baseline-run-4" / "model.pt").exists()
