"""``RouterModelFactory`` (router/models/registry.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from helpers import make_synthetic_irt, nirt_cfg, nirt_datasets

torch = pytest.importorskip("torch")

from router.llm.cost import CostModel
from router.models.registry import RouterModelFactory
from router.models.store import LocalArtifactStore
from router.nirt.predict import predict_dataset
from router.routing.routers import NIRTRouter
from training.nirt.train import fit


def _fitted_artifact(tmp_path, name="factory-run"):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=4)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    fit(nirt_cfg(), datasets=(train_ds, val_ds), name=name, runs_dir=tmp_path, verbose=False)
    artifact = LocalArtifactStore(tmp_path).load(name)
    return artifact, val_ds


def test_factory_create_matches_from_run(tmp_path):
    artifact, val_ds = _fitted_artifact(tmp_path, "factory-run-a")

    factory = RouterModelFactory()
    factory.register("nirt", NIRTRouter)
    cost_model = CostModel()
    router = factory.create(artifact, cost_model)

    assert isinstance(router, NIRTRouter)
    assert router.cost_model is cost_model
    assert router.artifact is artifact

    expected = NIRTRouter.from_run("factory-run-a", runs_dir=tmp_path)
    y_a, p_a = predict_dataset(router._model, val_ds, router._model_index)
    y_b, p_b = predict_dataset(expected._model, val_ds, expected._model_index)
    assert np.allclose(p_a, p_b)


def test_factory_create_from_id_and_store(tmp_path):
    artifact, _ = _fitted_artifact(tmp_path, "factory-run-b")

    factory = RouterModelFactory()
    factory.register("nirt", NIRTRouter)
    store = LocalArtifactStore(tmp_path)
    router = factory.create("factory-run-b", store, CostModel())
    assert isinstance(router, NIRTRouter)
    assert router.artifact.artifact_id == artifact.artifact_id


def test_factory_create_forwards_the_cost_model_keyword():
    """ROUTERMODELFACTORY-001: ``create(artifact, cost_model=cm)`` overwrote the
    keyword with the (None) positional slot, so ``from_artifact`` got no cost model."""
    class Recorder:
        @classmethod
        def from_artifact(cls, artifact, cost_model=None):
            return cost_model

    factory = RouterModelFactory()
    factory.register("recorder", Recorder)
    artifact = SimpleNamespace(router_model_name="recorder")
    cost_model = CostModel()
    assert factory.create(artifact, cost_model=cost_model) is cost_model
    assert factory.create(artifact, cost_model) is cost_model    # positional still works


def test_factory_raises_a_clear_error_for_an_unregistered_kind(tmp_path):
    artifact, _ = _fitted_artifact(tmp_path, "factory-run-c")
    factory = RouterModelFactory()
    with pytest.raises(ValueError, match="no RouterModel registered"):
        factory.create(artifact, CostModel())


def test_factory_raises_a_clear_error_when_the_class_has_no_from_artifact(tmp_path):
    artifact, _ = _fitted_artifact(tmp_path, "factory-run-d")

    class NoFromArtifact:
        pass

    factory = RouterModelFactory()
    factory.register("nirt", NoFromArtifact)
    with pytest.raises(TypeError, match="from_artifact"):
        factory.create(artifact, CostModel())
