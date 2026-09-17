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

from router.models.artifacts import (
    ArtifactFormat,
    MLPArtifactPayload,
    NIRTArtifactPayload,
    RouterModelArtifact,
)
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


def _mlp_artifact(artifact_id: str, fill: float) -> RouterModelArtifact:
    return RouterModelArtifact(
        artifact_id=artifact_id,
        payload=MLPArtifactPayload(
            state_dict={"w": torch.full((2,), fill)},
            model_ids=["a", "b"], in_dim=2, hidden=2, dropout=0.0,
        ),
        content_hash=str(fill), format=ArtifactFormat.PICKLE, router_model_name="mlp",
    )


def test_save_writes_model_pt_and_run_json_via_atomic_rename_not_in_place(tmp_path, monkeypatch):
    """RM-01: a killed write must never leave a directory with a complete
    model.pt and a missing/stale run.json (or vice versa) -- save() should
    write to a same-directory temp path and os.replace() into place, not
    write straight to the final path."""
    import os as os_mod

    store = LocalArtifactStore(tmp_path)
    replaced: list[str] = []
    real_replace = os_mod.replace

    def spy_replace(src, dst):
        replaced.append(str(src))
        assert str(src) != str(dst), "save() must write to a temp path, not the final path directly"
        return real_replace(src, dst)

    monkeypatch.setattr(os_mod, "replace", spy_replace)
    store.save(_mlp_artifact("atomic-run", 1.0))

    assert len(replaced) == 2  # model.pt and run.json each renamed into place
    # no leftover temp files after a successful save
    assert sorted(p.name for p in (tmp_path / "atomic-run").iterdir()) == ["model.pt", "run.json"]


def test_save_overwrite_failure_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """RM-01: torch.save writing straight to model.pt's final path means a
    crash mid-write on an overwrite clobbers the previously-good checkpoint
    in place. Writing to a temp file first means a failure here must leave
    the old, still-loadable checkpoint untouched."""
    store = LocalArtifactStore(tmp_path)
    store.save(_mlp_artifact("run1", 1.0))
    model_pt = tmp_path / "run1" / "model.pt"
    run_json = tmp_path / "run1" / "run.json"
    original_model_bytes = model_pt.read_bytes()
    original_run_json = run_json.read_text(encoding="utf-8")

    def boom(obj, f, *args, **kwargs):
        # simulate a real partial write: some bytes land at the path torch.save
        # was given, then the process dies -- corrupts whichever path save()
        # actually writes to (the final path if non-atomic, a temp path if not)
        from pathlib import Path

        Path(f).write_bytes(b"corrupt-partial-write")
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError):
        store.save(_mlp_artifact("run1", 2.0))

    assert model_pt.read_bytes() == original_model_bytes
    assert run_json.read_text(encoding="utf-8") == original_run_json
    # no stray temp files left behind in the artifact directory
    assert sorted(p.name for p in (tmp_path / "run1").iterdir()) == ["model.pt", "run.json"]


def test_store_save_rejects_a_non_nirt_payload(tmp_path):
    class _OtherPayload:
        pass

    artifact = RouterModelArtifact(
        artifact_id="x", payload=_OtherPayload(), content_hash="", format=ArtifactFormat.PICKLE,
    )
    with pytest.raises(NotImplementedError):
        LocalArtifactStore(tmp_path).save(artifact)
