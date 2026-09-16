"""``RouterModelArtifact``/``ArtifactPayload`` (router/models/artifacts.py)."""

from __future__ import annotations

from router.models.artifacts import ArtifactFormat, NIRTArtifactPayload, RouterModelArtifact


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
