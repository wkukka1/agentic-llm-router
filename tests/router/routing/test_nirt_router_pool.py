"""``NIRTRouter``'s broadened, call-time pool-scoring path
(``resolve_pool_profile_matrix`` / ``predict_pool_scores_text``): cold-start
scoring for a model outside the trained checkpoint's ``model_index``, via an
already-built profile row, an ad hoc rendered/embedded entry, or that same
entry kNN-blended toward the trained pool -- plus the free-mode guard and a
check that the ``_score_embeddings``/``_forward_pool`` refactor left the
existing narrow-pool path unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import FakeTrainingData, make_synthetic_irt, nirt_cfg, nirt_datasets
from helpers.stores import make_store

torch = pytest.importorskip("torch")
pytest.importorskip("faiss")

from router.routing import model_profiles
from router.routing.routers import NIRTRouter
from router.nirt.model import NIRTModel
from training.nirt.train import fit


class _StubEncoder:
    """Deterministic, hash-seeded ``encode`` -- no real model load."""

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts, show_progress: bool = False) -> np.ndarray:
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(rng.standard_normal(self.dim).astype(np.float32))
        return np.stack(out)


def _fit_projected(seed=5, n_models=8, held=("m6", "m7"), epochs=40):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(
        n_queries=800, n_models=n_models, seed=seed, informative_profiles=True
    )
    held = set(held)
    tr = train_obs[~train_obs.model_id.isin(held)].reset_index(drop=True)
    train_ds, val_ds = nirt_datasets(tr, val_obs, q_store, m_store)
    cfg = nirt_cfg(model_params="projected", dim=2)
    cfg["train"].update(epochs=epochs, patience=10, lr=1e-2)
    res = fit(cfg, datasets=(train_ds, val_ds), save=False, verbose=False)
    # fit()'s model_index enumerates every id seen across BOTH train_ds and
    # val_ds (held-out models still appear in val_obs so cold-start BCE can be
    # measured) -- drop them here so the router's own pool genuinely excludes
    # them, the way a model with zero observations anywhere (model_index's
    # real-world meaning) would be excluded. Projected mode never looks model
    # params up by this index (it comes from the profile embedding instead),
    # so truncating it doesn't change what the trained model computes.
    model_index = {m: i for m, i in res.model_index.items() if m not in held}
    data = FakeTrainingData(profile_store=m_store, query_store=q_store)
    router = NIRTRouter(res.model, model_index, data=data, pathway="irt")
    return router, q_store, m_store, held


# --------------------------------------------------------------------------- #
# free mode: unchanged hard guard, generalized to the pool API
# --------------------------------------------------------------------------- #
def test_free_mode_rejects_pool_scoring():
    model = NIRTModel(query_dim=4, dim=1, n_models=2, model_params="free", profile_dim=4)
    router = NIRTRouter(model, {"m0": 0, "m1": 1}, data=FakeTrainingData())
    with pytest.raises(ValueError, match="projected"):
        router.resolve_pool_profile_matrix(["m0"])


# --------------------------------------------------------------------------- #
# source 1: a pool id already in the profile store, outside model_index
# --------------------------------------------------------------------------- #
def test_held_out_profiled_model_uses_store_row_directly():
    router, q_store, m_store, held = _fit_projected()
    cold = next(iter(held))
    assert cold not in router.model_ids  # held out of training
    mat = router.resolve_pool_profile_matrix([cold])
    np.testing.assert_allclose(mat[0], m_store.get(cold))


# --------------------------------------------------------------------------- #
# source 2: an ad hoc entry for a model with NO store row at all
# --------------------------------------------------------------------------- #
def test_ad_hoc_entry_renders_and_embeds_on_the_fly():
    router, q_store, m_store, held = _fit_projected()
    encoder = _StubEncoder(m_store.dim)
    entry = {"feature": "A hypothetical new coding assistant.",
             "strengths": ["code"], "weaknesses": ["math"]}

    mat = router.resolve_pool_profile_matrix(
        ["brand-new-model"], profiles={"brand-new-model": entry}, profile_encoder=encoder,
    )
    expect = model_profiles.embed_ad_hoc_profile("brand-new-model", entry, encoder)
    np.testing.assert_allclose(mat[0], expect)
    # genuinely novel: not a zeroed/garbage row
    assert np.count_nonzero(mat[0]) > 0


def test_ad_hoc_entry_without_schema_still_renders_a_generic_profile():
    router, q_store, m_store, held = _fit_projected()
    encoder = _StubEncoder(m_store.dim)
    mat = router.resolve_pool_profile_matrix(["totally-bare-id"], profile_encoder=encoder)
    expect = model_profiles.embed_ad_hoc_profile("totally-bare-id", None, encoder)
    np.testing.assert_allclose(mat[0], expect)


# --------------------------------------------------------------------------- #
# source 3: the same ad hoc entry, additionally kNN-blended toward the
# trained pool (impute_unknown=True)
# --------------------------------------------------------------------------- #
def test_impute_unknown_blends_toward_trained_pool():
    from training.retrieval.model_bank import build_model_bank, knn_impute_model

    router, q_store, m_store, held = _fit_projected()
    encoder = _StubEncoder(m_store.dim)
    entry = {"feature": "A hypothetical new coding assistant.", "strengths": ["code"]}

    mat = router.resolve_pool_profile_matrix(
        ["brand-new-model"], profiles={"brand-new-model": entry},
        impute_unknown=True, knn_k=3, profile_encoder=encoder,
    )
    raw = model_profiles.embed_ad_hoc_profile("brand-new-model", entry, encoder)
    bank = build_model_bank(m_store, router.model_ids)
    expect, found = knn_impute_model(bank, raw, m_store, k=3, weighting="similarity")
    assert found > 0
    np.testing.assert_allclose(mat[0], expect)
    assert not np.allclose(mat[0], raw)  # the blend actually moved it


def test_impute_unknown_never_touches_the_trained_pools_own_rows():
    router, q_store, m_store, held = _fit_projected()
    trained_id = router.model_ids[0]
    mat = router.resolve_pool_profile_matrix(
        [trained_id], impute_unknown=True, knn_k=3,
    )
    np.testing.assert_allclose(mat[0], m_store.get(trained_id))


# --------------------------------------------------------------------------- #
# the _score_embeddings / _forward_pool refactor is behaviour-preserving
# --------------------------------------------------------------------------- #
def test_predict_scores_text_matches_manual_pool_refs_forward(monkeypatch):
    router, q_store, m_store, held = _fit_projected()
    query_encoder = _StubEncoder(q_store.dim)
    scores = router.predict_scores_text(["explain quicksort"], encoder=query_encoder)

    e_q = np.asarray(query_encoder.encode(["explain quicksort"]), dtype=np.float32)
    refs, pool_ref = router._pool_refs()
    expect = router._forward_pool(e_q, refs, pool_ref)
    np.testing.assert_allclose(scores.to_numpy(), expect)
    assert list(scores.columns) == router.model_ids


def test_predict_pool_scores_text_broadens_beyond_model_index():
    router, q_store, m_store, held = _fit_projected()
    query_encoder = _StubEncoder(q_store.dim)
    cold = next(iter(held))
    pool_ids = list(router.model_ids) + [cold]

    scores = router.predict_pool_scores_text(["explain quicksort"], pool_ids, encoder=query_encoder)
    assert list(scores.columns) == pool_ids
    assert np.isfinite(scores.to_numpy()).all()
