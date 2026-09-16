"""``LocalArtifactStore`` (router/models/store.py).

Pins the load-bearing invariant: a checkpoint written by
``training.nirt.train.fit`` must keep loading identically through both the
old ``router.nirt.checkpoint.load_run`` and the new store, and a checkpoint
written by the store must round-trip back through itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_synthetic_irt, nirt_cfg, nirt_datasets

torch = pytest.importorskip("torch")

from router.models.artifacts import ArtifactFormat, NIRTArtifactPayload, RouterModelArtifact
from router.models.store import LocalArtifactStore
from router.nirt.checkpoint import load_run
from router.nirt.predict import predict_dataset
from training.nirt.train import fit


def _fitted_run(tmp_path, name="store-run"):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=3)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)
    fit(nirt_cfg(), datasets=(train_ds, val_ds), name=name, runs_dir=tmp_path, verbose=False)
    return train_ds, val_ds


def test_store_load_matches_load_run_directly(tmp_path):
    train_ds, val_ds = _fitted_run(tmp_path, "direct-run")

    model_a, _, model_index_a = load_run("direct-run", runs_dir=tmp_path)
    y_a, p_a = predict_dataset(model_a, val_ds, model_index_a)

    artifact = LocalArtifactStore(tmp_path).load("direct-run")
    assert isinstance(artifact.payload, NIRTArtifactPayload)
    from router.nirt.model import build_model

    model_b = build_model(
        artifact.payload.config.get("model", {}),
        n_models=artifact.payload.n_models,
        query_dim=artifact.payload.query_dim,
        profile_dim=artifact.payload.profile_dim,
    )
    model_b.load_state_dict(artifact.payload.state_dict)
    model_b.eval()
    y_b, p_b = predict_dataset(model_b, val_ds, artifact.payload.model_index)

    assert np.allclose(p_a, p_b)
    assert artifact.supported_model_ids == sorted(model_index_a, key=model_index_a.get)


def test_store_round_trips_save_then_load(tmp_path):
    _fitted_run(tmp_path, "src-run")
    source = LocalArtifactStore(tmp_path).load("src-run")

    saved_artifact = RouterModelArtifact(
        artifact_id="saved-run",
        payload=source.payload,
        content_hash="abc123",
        format=ArtifactFormat.PICKLE,
        router_model_name="nirt",
    )
    store = LocalArtifactStore(tmp_path)
    store.save(saved_artifact)
    assert store.exists("saved-run")

    reloaded = store.load("saved-run")
    assert reloaded.payload.config == source.payload.config
    assert reloaded.payload.model_index == source.payload.model_index
    for k, v in reloaded.payload.state_dict.items():
        assert torch.equal(v, source.payload.state_dict[k])


def test_store_exists_is_false_for_a_missing_artifact(tmp_path):
    assert LocalArtifactStore(tmp_path).exists("nope") is False


def test_store_save_rejects_a_non_nirt_payload(tmp_path):
    class _OtherPayload:
        pass

    artifact = RouterModelArtifact(
        artifact_id="x", payload=_OtherPayload(), content_hash="", format=ArtifactFormat.PICKLE,
    )
    with pytest.raises(NotImplementedError):
        LocalArtifactStore(tmp_path).save(artifact)
