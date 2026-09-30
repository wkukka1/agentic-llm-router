"""``RouterModelArtifact``/``ArtifactPayload`` (router/models/artifacts.py)."""

from __future__ import annotations

from router.models.artifacts import (
    ArtifactFormat,
    BaselineArtifactPayload,
    MLPArtifactPayload,
    NIRTArtifactPayload,
    RouterModelArtifact,
)


def _payload() -> NIRTArtifactPayload:
    return NIRTArtifactPayload(
        state_dict={"w": [1.0, 2.0]},
        config={"model": {"orientation": "query_latent"}},
        n_models=3,
        model_index={"a": 0, "b": 1, "c": 2},
        query_dim=8,
        profile_dim=4,
    )


def test_nirt_payload_round_trips_through_to_state_and_from_state():
    payload = _payload()
    state = payload.to_state()
    assert state["format"] == "nirt"
    restored = NIRTArtifactPayload.from_state(state)
    assert restored == payload


def test_baseline_payload_round_trips_through_to_state_and_from_state():
    payload = BaselineArtifactPayload(
        state_dict={"w": [3.0]},
        model_cfg={"theta_dim": 2},
        model_index={"a": 0, "b": 1},
        query_dim=8,
        profile_dim=8,
        relevance_dim=0,
    )
    state = payload.to_state()
    assert state["format"] == "baseline"
    assert BaselineArtifactPayload.from_state(state) == payload


def test_mlp_payload_round_trips_through_to_state_and_from_state():
    payload = MLPArtifactPayload(
        state_dict={"w": [4.0]},
        model_ids=["a", "b"],
        in_dim=16,
        hidden=32,
        dropout=0.1,
    )
    state = payload.to_state()
    assert state["format"] == "mlp"
    assert MLPArtifactPayload.from_state(state) == payload


def test_mlp_artifact_payload_round_trips_pathway_fields():
    """MLPROUTERTRAINER-001: pathway/query_pathway/query_features must survive
    a to_state/from_state round trip so the serving side can restore them."""
    payload = MLPArtifactPayload(
        state_dict={}, model_ids=["a", "b"], in_dim=8, hidden=16, dropout=0.1,
        pathway="bert", query_pathway="bert-query", query_features="len",
    )
    restored = MLPArtifactPayload.from_state(payload.to_state())
    assert restored.pathway == "bert"
    assert restored.query_pathway == "bert-query"
    assert restored.query_features == "len"


def test_mlp_artifact_payload_from_state_defaults_pathway_for_old_artifacts():
    """Backward compatibility: an artifact saved before this field existed
    has no pathway/query_pathway/query_features keys at all."""
    old_state = {
        "format": "mlp", "state_dict": {}, "model_ids": ["a"],
        "in_dim": 4, "hidden": 8, "dropout": 0.1,
    }
    restored = MLPArtifactPayload.from_state(old_state)
    assert restored.pathway == "irt"
    assert restored.query_pathway is None
    assert restored.query_features is None


def test_router_model_artifact_holds_a_payload_not_a_path():
    artifact = RouterModelArtifact(
        artifact_id="run-1",
        payload=_payload(),
        content_hash="deadbeef",
        format=ArtifactFormat.PICKLE,
    )
    assert isinstance(artifact.payload, NIRTArtifactPayload)
    assert artifact.schema_version == ""
    assert artifact.training_run_id == ""
    assert artifact.supported_model_ids == []
    assert artifact.hyperparameters == {}
