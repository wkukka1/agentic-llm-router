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


def _mlp_artifact_versioned(artifact_id: str) -> RouterModelArtifact:
    return RouterModelArtifact(
        artifact_id=artifact_id,
        payload=MLPArtifactPayload(
            state_dict={"w": torch.ones(2)},
            model_ids=["a", "b"], in_dim=2, hidden=2, dropout=0.0,
        ),
        content_hash="h1", format=ArtifactFormat.PICKLE, router_model_name="mlp",
        router_model_version="v2", training_dataset_version="ds-7",
    )


def test_mlp_roundtrip_keeps_router_model_version_and_training_dataset_version(tmp_path):
    """LOCALARTIFACTSTORE-001: these two provenance fields must survive an
    MLP save/load round trip, not come back as ''."""
    store = LocalArtifactStore(tmp_path)
    store.save(_mlp_artifact_versioned("mlp-run"))
    reloaded = store.load("mlp-run")
    assert reloaded.router_model_version == "v2"
    assert reloaded.training_dataset_version == "ds-7"


def test_save_merges_onto_an_existing_fit_style_run_json_instead_of_replacing_it(tmp_path):
    """LOCALARTIFACTSTORE-002: fit()'s run.json (history/val_metrics/git_sha)
    must survive a later store.save() for the same artifact id."""
    import json

    run_dir = tmp_path / "fit-run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "git_sha": "abc123", "history": [{"epoch": 0, "loss": 1.0}], "best_epoch": 0,
    }))

    store = LocalArtifactStore(tmp_path)
    store.save(_mlp_artifact_versioned("fit-run"))

    on_disk = json.loads((run_dir / "run.json").read_text())
    assert on_disk["git_sha"] == "abc123"
    assert on_disk["history"] == [{"epoch": 0, "loss": 1.0}]
    assert on_disk["router_model_version"] == "v2"  # store's own fields still land


@pytest.mark.parametrize("bad_id", ["../escaped", "/abs/escaped", "sub/../../escaped"])
def test_save_rejects_an_artifact_id_that_would_escape_the_root(tmp_path, bad_id):
    """LOCALARTIFACTSTORE-003"""
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.save(_mlp_artifact_versioned(bad_id))
    # nothing was written anywhere outside (or inside) the root
    assert not any(tmp_path.rglob("model.pt"))


def test_load_reads_model_pt_exactly_once(tmp_path, monkeypatch):
    """LOCALARTIFACTSTORE-004: load() must not deserialize the same model.pt
    more than once."""
    store = LocalArtifactStore(tmp_path)
    store.save(_mlp_artifact_versioned("once-run"))

    calls = []
    real_load = torch.load

    def counting_load(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    store.load("once-run")
    assert len(calls) == 1


def test_reload_after_resave_never_sees_new_weights_with_stale_metadata(tmp_path):
    """LOCALARTIFACTSTORE-005: metadata rides inside the same atomic model.pt
    write as the weights, so a load() right after a re-save must reflect the
    new save's metadata even if run.json is deleted/stale."""
    store = LocalArtifactStore(tmp_path)
    store.save(_mlp_artifact_versioned("resave-run"))

    resaved = RouterModelArtifact(
        artifact_id="resave-run",
        payload=MLPArtifactPayload(
            state_dict={"w": torch.zeros(2)}, model_ids=["a", "b"], in_dim=2, hidden=2, dropout=0.0,
        ),
        content_hash="h2", format=ArtifactFormat.PICKLE, router_model_name="mlp",
        router_model_version="v3", training_dataset_version="ds-8",
    )
    store.save(resaved)
    (tmp_path / "resave-run" / "run.json").unlink()  # remove the mirror entirely

    reloaded = store.load("resave-run")
    assert reloaded.content_hash == "h2"
    assert reloaded.router_model_version == "v3"
    assert torch.equal(reloaded.payload.state_dict["w"], torch.zeros(2))


def test_corrupt_run_json_logs_a_warning_instead_of_reading_as_clean(tmp_path, caplog):
    """LOCALARTIFACTSTORE-006"""
    run_dir = tmp_path / "corrupt-run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{not valid json")

    store = LocalArtifactStore(tmp_path)
    with caplog.at_level("WARNING"):
        meta = store._read_run_json("corrupt-run")
    assert meta == {}
    assert any("corrupt-run" in r.message for r in caplog.records)
