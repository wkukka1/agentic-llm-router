"""Folding a NIRT into the gateway's ``mlp`` router file
(:mod:`training.trainers.nirt_router_export`): exact for the architectures it
accepts, refused for the rest."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from router.models.artifacts import MLPArtifactPayload
from router.nirt.model import build_model
from router.routing.routers import NIRTRouter
from training.trainers.nirt_router_export import (
    check_foldable,
    fold_nirt,
    nirt_scores,
    parity_max_diff,
    parity_max_diff_file,
)

Q_DIM, P_DIM, M = 12, 7, 5
MODELS = [f"m{j}" for j in range(M)]


def make(**over):
    cfg = {"dim": 2, "model_params": "projected", "query_hidden": 16, "constrain_discrimination": "auto", **over}
    torch.manual_seed(0)
    model = build_model(cfg, n_models=M, query_dim=Q_DIM, profile_dim=P_DIM)
    model.eval()
    return model


def world(model, seed=0):
    rng = np.random.default_rng(seed)
    refs = rng.standard_normal((M, P_DIM)).astype(np.float32)
    e_q = rng.standard_normal((64, Q_DIM)).astype(np.float32)
    router = NIRTRouter(model, {m: j for j, m in enumerate(MODELS)})
    router._pool_ref_cache = (refs, refs)  # the profile matrix, without a data store
    return router, refs, e_q


@pytest.mark.parametrize("over", [{}, {"center_discrimination": True}, {"constrain_discrimination": False},
                                  {"dim": 1}])
def test_the_fold_matches_the_nirt_exactly(over):
    router, refs, e_q = world(make(**over))
    state = fold_nirt(router._model, refs, refs)
    assert parity_max_diff(state, nirt_scores(router, e_q), e_q) < 1e-6


def test_the_saved_file_scores_the_same_through_the_gateway_loader(tmp_path):
    router, refs, e_q = world(make())
    state = fold_nirt(router._model, refs, refs)
    payload = MLPArtifactPayload(state_dict={"0.weight": state["w1"], "0.bias": state["b1"],
                                             "3.weight": state["w2"], "3.bias": state["b2"]},
                                 model_ids=MODELS, in_dim=Q_DIM, hidden=16, dropout=0.0, pathway="retrieval")
    path = tmp_path / "nirt.npz"
    payload.save(path, encoder={"backend": "sentence_transformer", "model_name": "x"},
                 meta={"artifact_id": "nirt-toy"})
    assert parity_max_diff_file(path, MODELS, e_q, nirt_scores(router, e_q)) < 1e-6
    with pytest.raises(ValueError, match="model order"):
        parity_max_diff_file(path, MODELS[::-1], e_q, nirt_scores(router, e_q))

    from aar_gateway._core.scorers import load_router
    assert load_router(path).name == "nirt-toy"


def test_a_pool_in_the_wrong_order_fails_parity():
    router, refs, e_q = world(make())
    state = fold_nirt(router._model, refs[::-1].copy(), refs)
    with pytest.raises(ValueError, match="disagrees"):
        parity_max_diff(state, nirt_scores(router, e_q), e_q)


@pytest.mark.parametrize("over, why", [
    ({"interaction": True}, "interaction"),
    ({"query_difficulty": True}, "query_difficulty"),
    ({"difficulty": "vector"}, "difficulty"),
    ({"query_head_batchnorm": True}, "batchnorm"),
    ({"query_head_layers": 2}, "query head"),
    ({"query_hidden": None}, "query head"),
    ({"model_params": "free"}, "model_params"),
    ({"orientation": "model_latent"}, "orientation"),
])
def test_architectures_that_dont_fold_are_refused(over, why):
    with pytest.raises(ValueError, match=why):
        check_foldable(make(**over))


@pytest.mark.parametrize("data, why", [
    ({"pathway": "retrieval", "query_features": "cx"}, "query_features"),
    ({"pathway": "retrieval", "query_pathway": "irt"}, "query_pathway"),
])
def test_run_configs_that_dont_fold_are_refused(data, why):
    with pytest.raises(ValueError, match=why):
        check_foldable(make(), {"data": data})


def test_non_finite_weights_are_refused():
    router, refs, _ = world(make())
    with torch.no_grad():
        router._model.query_head[2].weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        fold_nirt(router._model, refs, refs)


@pytest.mark.parametrize("over", [{}, {"profile_latent_dim": 32, "profile_encoder_hidden": 16}])
def test_an_ad_hoc_pool_folds_exactly(over):
    """A pool the run never trained on (the Claude tiers): rows from their own
    profiles, self-centred -- the profile manifold folds into A and B too."""
    model = make(**over)
    check_foldable(model)
    router, _, e_q = world(model)
    tiers = np.random.default_rng(1).standard_normal((3, P_DIM)).astype(np.float32)
    state = fold_nirt(model, tiers, tiers)
    assert parity_max_diff(state, router._forward_pool(e_q, tiers, tiers), e_q) < 1e-6


@pytest.mark.parametrize("encoder, ok", [
    ({"backend": "sentence_transformer", "model_name": "sentence-transformers/all-MiniLM-L6-v2"}, True),
    ({"backend": "bert", "model_name": "bert-base-uncased"}, False),
    ({"backend": "sentence_transformer", "model_name": "bert-base-uncased"}, False),
])
def test_only_encoders_the_gateway_can_load_are_exported(encoder, ok):
    pytest.importorskip("fastembed")
    from training.trainers.nirt_router_export import check_servable_encoder

    if ok:
        check_servable_encoder(encoder)
    else:
        with pytest.raises(ValueError, match="gateway|fastembed"):
            check_servable_encoder(encoder)


def test_a_bert_pathway_is_named_as_a_local_onnx_encoder(tmp_path):
    from types import SimpleNamespace

    from training.trainers.nirt_router_export import check_servable_encoder, gateway_encoder_spec

    cfg = SimpleNamespace(backend="bert", model_name="bert-base-uncased", normalize=False,
                          max_seq_length=256, pooling="mean")
    spec = gateway_encoder_spec(cfg, dim=768)
    assert spec == {"backend": "fastembed", "model_name": "aar/bert-base-uncased-mean", "normalize": False,
                    "max_seq_length": 256, "model_path": "bert-base-uncased-mean", "pooling": "mean", "dim": 768}
    with pytest.raises(ValueError, match="export_bert_encoder"):
        check_servable_encoder(spec, encoders_dir=tmp_path)
    (tmp_path / "bert-base-uncased-mean").mkdir()
    (tmp_path / "bert-base-uncased-mean" / "model.onnx").write_bytes(b"")
    check_servable_encoder(spec, encoders_dir=tmp_path)


def test_a_router_file_resolves_its_local_encoder_under_encoders_dir(tmp_path):
    from aar_gateway._core.scorers import _artifact_encoder

    spec = {"backend": "fastembed", "model_name": "aar/x", "model_path": "x", "pooling": "mean", "dim": 8,
            "normalize": False, "max_seq_length": 256}
    assert _artifact_encoder(spec, tmp_path)["model_path"] == str(tmp_path / "x")
    assert _artifact_encoder({"backend": "sentence_transformer", "model_name": "m"}, tmp_path).get("model_path") is None


def test_tier_offsets_fold_exactly():
    from training.nirt.judge_finetune import predict_judged

    model = make()
    _, _, e_q = world(model)
    tiers = np.random.default_rng(2).standard_normal((3, P_DIM)).astype(np.float32)
    da = np.random.default_rng(3).standard_normal((3, model.dim)).astype(np.float32) * 0.3
    db = np.array([0.2, -0.1, 0.4], np.float32)
    state = fold_nirt(model, tiers, None, offsets=(da, db))
    assert parity_max_diff(state, predict_judged((model, da, db), e_q, tiers), e_q) < 1e-6


def test_an_ensemble_of_folds_is_the_mean_logit_exactly():
    from training.nirt.judge_finetune import predict_judged
    from training.trainers.nirt_router_export import ensemble_fold

    tiers = np.random.default_rng(4).standard_normal((3, P_DIM)).astype(np.float32)
    models = []
    for seed in range(3):
        torch.manual_seed(seed)
        m = build_model({"dim": 2, "model_params": "projected", "query_hidden": 16}, n_models=M,
                        query_dim=Q_DIM, profile_dim=P_DIM).eval()
        models.append((m, np.zeros((3, 2), np.float32), np.zeros(3, np.float32)))
    _, _, e_q = world(models[0][0])
    state = ensemble_fold([fold_nirt(m, tiers, None) for m, _, _ in models])
    mean_logit = np.mean([predict_judged(f, e_q, tiers, logits=True) for f in models], axis=0)
    assert state["w1"].shape == (48, Q_DIM)
    assert parity_max_diff(state, 1 / (1 + np.exp(-mean_logit)), e_q) < 1e-6
    with pytest.raises(ValueError):
        ensemble_fold([])
