"""The kNN router as one small file: saved and loaded without the training
data, scoring exactly like KNNRouter, loaded by the gateway's config, and
installed by `aar-gateway setup`."""

from __future__ import annotations

import numpy as np
import pytest

from router.gateway import cli
from router.gateway.routing import load_router_model
from router.models.artifacts import ArtifactFormat, KNNArtifactPayload, RouterModelArtifact
from router.routing.routers import KNNRouter


class FakeEncoder:
    """Maps a prompt to a fixed unit vector by its first word."""
    VECTORS = {"easy": [1.0, 0.0], "hard": [0.0, 1.0]}

    def encode(self, texts):
        return np.array([self.VECTORS[t.split()[0]] for t in texts], dtype=np.float32)


def payload(k=2):
    # four training prompts: two "easy" (cheap model fine), two "hard" (only strong)
    return KNNArtifactPayload(
        embeddings=np.array([[1, 0], [0.99, 0.14], [0, 1], [0.14, 0.99]], dtype=np.float32),
        scores=np.array([[1, 1], [1, 0.9], [1, 0], [0.9, 0.1]], dtype=np.float32),
        model_ids=["strong", "cheap"], k=k,
        encoder={"backend": "sentence_transformer", "model_name": "x"}, model_costs=[10.0, 1.0])


def test_the_file_round_trips(tmp_path):
    path = tmp_path / "router.npz"
    payload().save(path, meta={"n_queries": 4, "dataset": "toy"})
    back, header = KNNArtifactPayload.load(path)
    assert back.model_ids == ["strong", "cheap"] and back.k == 2 and header["dataset"] == "toy"
    assert np.allclose(back.scores, payload().scores) and np.allclose(back.embeddings, payload().embeddings, atol=1e-3)
    (tmp_path / "other.npz").write_bytes(b"not a router")
    with pytest.raises(ValueError):
        KNNArtifactPayload.load(tmp_path / "other.npz")


def test_a_router_from_the_file_averages_its_nearest_training_prompts():
    artifact = RouterModelArtifact(artifact_id="t", payload=payload(), content_hash="", format=ArtifactFormat.NPZ)
    router = KNNRouter.from_artifact(artifact, encoder=FakeEncoder())
    scores = router.predict_scores_text(["easy question", "hard question"])
    assert list(scores.columns) == ["strong", "cheap"]
    assert np.allclose(scores.loc[0], [1.0, 0.95]) and np.allclose(scores.loc[1], [0.95, 0.05])
    assert list(router.default_model_costs) == [10.0, 1.0]
    with pytest.raises(NotImplementedError):
        router.predict_scores(["q1"])


def test_the_gateway_loads_the_router_file_from_its_config(tmp_path, monkeypatch):
    path = tmp_path / "router.npz"
    payload().save(path)
    router = load_router_model({"artifact": str(path), "kind": "knn"})
    assert isinstance(router, KNNRouter) and router.model_ids == ["strong", "cheap"]


def test_setup_installs_the_router_and_writes_a_config(tmp_path, monkeypatch, capsys):
    source = tmp_path / "downloaded.npz"
    payload().save(source, meta={"n_queries": 4})
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    assert cli.main(["setup", "--router", str(source)]) == 0
    home = tmp_path  # AAR_GATEWAY_LOG lives in tmp_path (conftest)
    assert (home / "router.npz").exists()
    config = (home / "config.yaml").read_text()
    assert "enabled: true" in config and str(home / "router.npz") in config
    from router.gateway.config import load_settings
    assert load_settings(home / "config.yaml")["router"]["artifact"] == str(home / "router.npz")
    (home / "config.yaml").write_text("mine\n")
    cli.main(["setup", "--router", str(source)])
    assert (home / "config.yaml").read_text() == "mine\n"  # kept without --force


def test_setup_refuses_a_file_that_is_not_a_router(tmp_path, monkeypatch):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"nope")
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    with pytest.raises(SystemExit):
        cli.main(["setup", "--router", str(bad)])
    assert not (tmp_path / "router.npz").exists()


def mlp_payload():
    from router.models.artifacts import MLPArtifactPayload

    rng = np.random.default_rng(0)
    state = {"0.weight": rng.normal(size=(4, 2)).astype(np.float32), "0.bias": np.zeros(4, np.float32),
             "3.weight": rng.normal(size=(2, 4)).astype(np.float32), "3.bias": np.zeros(2, np.float32)}
    return MLPArtifactPayload(state_dict=state, model_ids=["strong", "cheap"], in_dim=2, hidden=4, dropout=0.1,
                              pathway="retrieval")


def test_a_trained_router_file_holds_weights_only_and_serves_with_numpy(tmp_path):
    from router.models.artifacts import load_router_file
    from router.routing.routers import MLPRouter

    path = tmp_path / "router.npz"
    mlp_payload().save(path, encoder={"backend": "sentence_transformer", "model_name": "x"},
                       model_costs=[10.0, 1.0], meta={"dataset": "toy"})
    with np.load(path) as z:
        assert sorted(z.files) == ["b1", "b2", "header", "w1", "w2"]  # no prompts, no scores
    payload, header = load_router_file(path)
    assert header["format"] == "mlp" and payload.model_ids == ["strong", "cheap"]
    router = load_router_model({"artifact": str(path)})
    assert isinstance(router, MLPRouter) and router.can_route_text
    router._file_encoder = (FakeEncoder(), {})
    got = router.predict_scores_text(["easy question"]).to_numpy()[0]
    sd = mlp_payload().state_dict
    x = np.array([1.0, 0.0], np.float32)
    want = 1 / (1 + np.exp(-(np.maximum(x @ sd["0.weight"].T, 0) @ sd["3.weight"].T)))
    assert np.allclose(got, want, atol=1e-5)
    assert list(router.default_model_costs) == [10.0, 1.0]


def test_setup_installs_a_trained_router_file(tmp_path, monkeypatch, capsys):
    source = tmp_path / "trained.npz"
    mlp_payload().save(source, encoder={"backend": "sentence_transformer", "model_name": "x"})
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    assert cli.main(["setup", "--router", str(source)]) == 0
    assert "trained network" in capsys.readouterr().out


def test_setup_with_no_arguments_installs_the_router_shipped_with_the_package(tmp_path, monkeypatch, capsys):
    from router.models.artifacts import load_router_file

    assert cli.BUNDLED_ROUTER.exists()
    payload, header = load_router_file(cli.BUNDLED_ROUTER)
    assert header["format"] == "mlp" and "gpt-5" in payload.model_ids  # the stand-ins the gateway config names
    monkeypatch.delenv("AAR_GATEWAY_ROUTER_URL", raising=False)
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    assert cli.main(["setup"]) == 0
    assert (tmp_path / "router.npz").read_bytes() == cli.BUNDLED_ROUTER.read_bytes()
