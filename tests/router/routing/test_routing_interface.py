"""The modular routing interface (router.routing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from router.routing import (
    REGISTRY,
    MatrixRouter,
    RandomRouter,
    RouterModel,
    RoutingResult,
    build_router,
    register,
)

M = ["GPT-A", "GPT-B", "GPT-C"]


def _mats():
    #                A     B     C
    true = np.array([[0.72, 0.94, 0.81],   # oracle = B
                     [0.50, 0.50, 0.20],   # oracle tie A & B
                     [0.10, 0.20, 0.20]])  # oracle tie B & C
    cost = np.array([[0.01, 0.03, 0.02],
                     [0.05, 0.02, 0.01],
                     [0.01, 0.09, 0.04]])
    q = ["q001", "q002", "q003"]
    return pd.DataFrame(true, index=q, columns=M), pd.DataFrame(cost, index=q, columns=M)


# --------------------------------------------------------------------------- #
# MatrixRouter / RoutingResult                                                #
# --------------------------------------------------------------------------- #
def test_route_returns_argmax_and_result_shape():
    true_df, _ = _mats()
    r = MatrixRouter(true_df, name="truth")
    res = r.route(["q001", "q003"])
    assert isinstance(res, RoutingResult)
    assert res.query_ids == ["q001", "q003"]
    assert res.model_ids == M
    assert res.selected_model_ids == ["GPT-B", "GPT-B"]
    assert res.scores.shape == (2, 3)
    np.testing.assert_allclose(res.selected_scores, [0.94, 0.20])


def test_result_frames_and_mix():
    true_df, _ = _mats()
    res = MatrixRouter(true_df).route(list(true_df.index))
    frame = res.to_frame()
    assert list(frame.columns) == ["query_id", "selected_model_id", "selected_score"]
    assert len(frame) == 3
    # q001,q003 -> B ; q002 [0.5,0.5,0.2] -> A (first of the tie)
    assert res.model_mix() == {"GPT-A": 1, "GPT-B": 2}
    assert sum(res.model_mix().values()) == 3
    pd.testing.assert_frame_equal(res.score_frame(), true_df)


def test_call_dispatches_to_a_subclass_route_override():
    """ROUTERMODEL-001: ``__call__ = route`` froze the base function, so
    ``router(...)`` bypassed a subclass's own ``route()``."""
    true_df, _ = _mats()

    class Overriding(MatrixRouter):
        def route(self, *args, **kwargs):
            raise RuntimeError("subclass route() called")

    with pytest.raises(RuntimeError, match="subclass route"):
        Overriding(true_df)(["q001"])


def test_series_model_costs_are_aligned_by_label_not_position():
    """ROUTERMODEL-003: a labelled ``pd.Series`` in a different order than
    ``model_ids`` must be aligned by index, like a dict."""
    true_df, _ = _mats()
    shuffled = pd.Series({"GPT-C": 0.03, "GPT-A": 0.01, "GPT-B": 0.02})
    res = MatrixRouter(true_df).route(["q001"], lam=1.0, model_costs=shuffled)
    np.testing.assert_allclose(res.model_costs, [0.01, 0.02, 0.03])


def test_series_model_costs_missing_a_model_raises():
    true_df, _ = _mats()
    with pytest.raises(KeyError, match="GPT-C"):
        MatrixRouter(true_df).route(
            ["q001"], lam=1.0, model_costs=pd.Series({"GPT-A": 0.01, "GPT-B": 0.02}))


def test_query_order_and_missing_rows_preserved():
    true_df, _ = _mats()
    r = MatrixRouter(true_df)
    res = r.route(["q003", "nope", "q001"])
    assert res.query_ids == ["q003", "nope", "q001"]
    # the unknown query has all-NaN scores -> routing_decision falls back to col 0
    assert res.selected_model_ids == ["GPT-B", "GPT-A", "GPT-B"]


# --------------------------------------------------------------------------- #
# cost-aware routing                                                          #
# --------------------------------------------------------------------------- #
def test_cost_aware_selection_changes_with_lambda():
    scores = pd.DataFrame([[0.90, 0.88]], index=["q"], columns=["exp", "cheap"])
    r = MatrixRouter(scores)
    assert r.route(["q"], lam=0.0, model_costs={"exp": 1.0, "cheap": 0.0}).selected_model_ids == ["exp"]
    assert r.route(["q"], lam=0.1, model_costs={"exp": 1.0, "cheap": 0.0}).selected_model_ids == ["cheap"]


def test_default_model_costs_used_when_lambda_positive():
    scores = pd.DataFrame([[0.90, 0.88]], index=["q"], columns=["exp", "cheap"])
    r = MatrixRouter(scores, model_costs=[1.0, 0.0])
    assert r.route(["q"], lam=0.1).selected_model_ids == ["cheap"]
    assert r.route(["q"], lam=0.0).selected_model_ids == ["exp"]


def test_model_costs_dict_missing_entry_raises():
    true_df, _ = _mats()
    with pytest.raises(KeyError):
        MatrixRouter(true_df).route(["q001"], lam=0.5, model_costs={"GPT-A": 0.1})


def test_eligible_mask_blocks_models():
    true_df, _ = _mats()
    elig = pd.DataFrame(True, index=true_df.index, columns=M)
    elig.loc["q001", "GPT-B"] = False       # force the second-best on q001
    res = MatrixRouter(true_df).route(list(true_df.index), eligible=elig)
    assert res.selected_model_ids[0] == "GPT-C"


# --------------------------------------------------------------------------- #
# IRT-Router-style alpha routing                                              #
# --------------------------------------------------------------------------- #
def test_alpha_extremes_are_quality_only_and_cost_averse():
    scores = pd.DataFrame([[0.90, 0.88]], index=["q"], columns=["exp", "cheap"])
    r = MatrixRouter(scores, model_costs=[1.0, 0.0])
    assert r.route(["q"], alpha=1.0).selected_model_ids == ["exp"]
    assert r.route(["q"], alpha=0.05).selected_model_ids == ["cheap"]


def test_alpha_matches_manually_normalized_lam_call():
    """alpha-routing is argmax[alpha*pred - (1-alpha)*linear_cost(C)], which is
    the same ordering as lam-routing on the linear_cost-normalized costs with
    lam = alpha_to_lam(alpha) -- the equivalence the design relies on."""
    from router.nirt.routing_decision import alpha_to_lam, linear_cost

    true_df, _ = _mats()
    raw_costs = [1.0, 2.0, 3.0]
    r = MatrixRouter(true_df, model_costs=raw_costs)
    alpha = 0.7
    via_alpha = r.route(list(true_df.index), alpha=alpha)
    via_lam = r.route(list(true_df.index), lam=alpha_to_lam(alpha),
                      model_costs=linear_cost(raw_costs, raw_costs))
    assert via_alpha.selected_model_ids == via_lam.selected_model_ids
    assert via_alpha.alpha == pytest.approx(alpha)


def test_alpha_and_lam_both_given_raises():
    true_df, _ = _mats()
    r = MatrixRouter(true_df, model_costs=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="alpha.*lam|lam.*alpha"):
        r.route(list(true_df.index), lam=0.5, alpha=0.5)


def test_route_text_can_use_a_per_cell_predicted_cost():
    """route_text() used to always resolve costs via _resolve_costs(model_costs)
    or default_model_costs, never predict_costs() -- so a router with a real
    per-cell cost model (decision_costs()'s whole point) couldn't use it for
    live-text routing, only for route(). route_text() must now consult it too,
    keyed by the same positional ids ("0", "1", ...) it scores with."""

    class TextRouterWithPerCellCost(RouterModel):
        kind = "_text_per_cell_cost_test"
        can_route_text = True

        def __init__(self):
            super().__init__(["exp", "cheap"], name="t")

        @property
        def default_model_costs(self):
            return np.array([1.0, 1.0])          # flat: would pick "exp" (tie -> col 0)

        def predict_scores(self, query_ids):
            raise AssertionError("route_text must not call predict_scores")

        def predict_scores_text(self, texts, *, encoder=None):
            return pd.DataFrame([[0.90, 0.88]] * len(texts), columns=["exp", "cheap"])

        def predict_costs(self, query_ids):
            # per-cell: "exp" really is expensive, "cheap" really is cheap --
            # the opposite of the flat default_model_costs above
            return pd.DataFrame([[1.0, 0.0]] * len(query_ids), index=query_ids,
                                columns=["exp", "cheap"])

    r = TextRouterWithPerCellCost()
    assert r.route_text(["hello"], lam=0.1).selected_model_ids == ["cheap"]
    assert r.route_text(["hello"]).selected_model_ids == ["exp"]  # lam=0: cost-blind, as before


# --------------------------------------------------------------------------- #
# evaluate_router()                                                           #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# registry                                                                    #
# --------------------------------------------------------------------------- #
def test_build_router_by_kind():
    r = build_router("random", model_ids=M, seed=1)
    assert isinstance(r, RandomRouter)
    assert r.model_ids == M


def test_build_router_unknown_kind():
    with pytest.raises(ValueError, match="unknown router kind"):
        build_router("does-not-exist", model_ids=M)


def test_register_rejects_missing_kind():
    with pytest.raises(ValueError, match="distinct `kind`"):
        @register
        class _NoKind(RouterModel):
            def predict_scores(self, query_ids):
                ...


def test_register_custom_router_roundtrips():
    @register
    class _ConstRouter(RouterModel):
        kind = "_const_test"

        def predict_scores(self, query_ids):
            ids = [str(q) for q in query_ids]
            return pd.DataFrame(
                np.tile([0.1, 0.9, 0.2], (len(ids), 1)), index=ids, columns=self._model_ids
            )

    try:
        r = build_router("_const_test", model_ids=M)
        assert r.route(["q1", "q2"]).selected_model_ids == ["GPT-B", "GPT-B"]
    finally:
        REGISTRY.pop("_const_test", None)


# --------------------------------------------------------------------------- #
# RandomRouter                                                               #
# --------------------------------------------------------------------------- #
def test_random_router_is_deterministic():
    a = RandomRouter(M, seed=7).predict_scores(["q1", "q2"])
    b = RandomRouter(M, seed=7).predict_scores(["q1", "q2"])
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# XA-01: cost-aware evaluation must match the served router's own cost vector #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# NIRTRouter (needs torch)                                                    #
# --------------------------------------------------------------------------- #
Q_IDS = [f"q{i}" for i in range(6)]
M_IDS = ["m0", "m1", "m2"]
MODEL_INDEX = {"m0": 0, "m1": 1, "m2": 2}


class _Data:
    """The three ``TrainingData`` accessors NIRTRouter touches."""

    def __init__(self, q_store, m_store):
        self._q, self._m = q_store, m_store

    def query_embeddings(self, pw):
        return self._q

    def profile_embeddings(self, pw):
        return self._m

    def nirt_observations(self):
        return pd.DataFrame({"split": ["train"] * 3, "model_id": M_IDS, "cost": [0.1, 0.2, 0.3]})


@pytest.fixture
def nirt_router():
    pytest.importorskip("torch")
    from helpers.stores import make_store
    from router.nirt.model import build_model
    from router.routing import NIRTRouter

    data = _Data(make_store(Q_IDS, 8, "query_id"), make_store(M_IDS, 4, "model_id", seed=1))
    model = build_model({"model_params": "projected", "dim": 2}, n_models=3, query_dim=8, profile_dim=4)
    return NIRTRouter(model, MODEL_INDEX, data=data, name="nirt-smoke"), model, data


def test_nirt_router_default_costs_come_from_the_observation_table(nirt_router):
    router, _, _ = nirt_router
    np.testing.assert_allclose(router.default_model_costs, [0.1, 0.2, 0.3])
    assert router.route(Q_IDS).selected.shape == (6,)
    assert router.route(Q_IDS, lam=5.0).model_costs is not None


def test_nirt_router_scores_raw_query_embeddings(nirt_router):
    from router.routing import NIRTRouter

    router, _, _ = nirt_router
    assert NIRTRouter.can_route_text is True
    text_scores = router._score_embeddings(np.random.default_rng(0).standard_normal((4, 8)))
    assert text_scores.shape == (4, 3)
    assert list(text_scores.columns) == M_IDS
    # a short embedding with no feature variant to explain it is a wrong encoder
    with pytest.raises(ValueError, match="does not match model query_dim"):
        router._score_embeddings(np.zeros((2, 6)))


def test_nirt_router_zero_pads_exactly_the_structured_feature_block(nirt_router):
    from router.routing import NIRTRouter

    _, model, data = nirt_router

    class _Feat:
        dim = 2

    class _FeatData(_Data):
        def query_features(self, name):
            return _Feat()

    rng = np.random.default_rng(0)
    router = NIRTRouter(model, MODEL_INDEX, data=_FeatData(data._q, data._m),
                        query_features="default", name="nirt-feat")
    with pytest.warns(UserWarning, match="zero-fills"):
        padded = router._score_embeddings(rng.standard_normal((2, 6)))
    assert padded.shape == (2, 3)
    with pytest.raises(ValueError):
        router._score_embeddings(rng.standard_normal((2, 5)))


def test_nirt_router_lazy_data_is_the_serving_view(monkeypatch):
    """With no ``data=``, the first use loads ``TrainingData`` in serving mode
    (cfg + stores only) -- never the full training tables."""
    pytest.importorskip("torch")
    import training.data.facade as facade
    from router.nirt.model import build_model
    from router.routing import NIRTRouter

    seen = {}

    def fake_load(cfg, *, serving=False):
        seen["serving"] = serving
        return object()

    monkeypatch.setattr(facade, "load_training_data", fake_load)
    model = build_model({"model_params": "projected", "dim": 2}, n_models=3, query_dim=8, profile_dim=4)
    router = NIRTRouter(model, MODEL_INDEX)
    router._data
    assert seen == {"serving": True}


def _per_model_loop_scores(router, e_q):
    """The pre-batching reference: one forward pass per pool model."""
    import torch

    from router.nirt.predict import pool_centering

    model = router._model
    projected = model.model_params == "projected"
    m_store = router._store("profile", router._pathway) if projected else None
    q = torch.from_numpy(np.ascontiguousarray(e_q, dtype=np.float32))
    pool_ref = (np.stack([np.asarray(m_store.get(m), np.float32) for m in router.model_ids])
                if projected else None)
    cols = {}
    with torch.no_grad(), pool_centering(model, pool_ref):
        for m in router.model_ids:
            if projected:
                pv = np.asarray(m_store.get(m), np.float32)
                ref = torch.from_numpy(np.repeat(pv[None], len(e_q), axis=0))
            else:
                ref = torch.full((len(e_q),), int(router._model_index[m]), dtype=torch.long)
            cols[m] = torch.sigmoid(model(q, ref)).numpy().astype(np.float64)
    return pd.DataFrame(cols)[router.model_ids]


@pytest.mark.parametrize("model_cfg", [
    {"model_params": "projected", "dim": 2},
    {"model_params": "projected", "dim": 2, "center_discrimination": True},
    {"model_params": "free", "dim": 2},
])
def test_nirt_batched_scoring_matches_per_model_loop(model_cfg):
    pytest.importorskip("torch")
    import torch

    from helpers.stores import make_store
    from router.nirt.model import build_model
    from router.routing import NIRTRouter

    torch.manual_seed(0)
    data = _Data(make_store(Q_IDS, 8, "query_id"), make_store(M_IDS, 4, "model_id", seed=1))
    model = build_model(model_cfg, n_models=3, query_dim=8, profile_dim=4)
    router = NIRTRouter(model, MODEL_INDEX, data=data)
    e_q = np.random.default_rng(3).normal(size=(5, 8))
    got = router._score_embeddings(e_q)
    want = _per_model_loop_scores(router, e_q)
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=1e-5, atol=1e-6)
    assert list(got.columns) == router.model_ids
    # cached pool reference is reused across calls, and still agrees
    np.testing.assert_allclose(router._score_embeddings(e_q).to_numpy(), want.to_numpy(),
                               rtol=1e-5, atol=1e-6)


class _StubEncoder:
    def __init__(self, dim):
        self.dim, self.loads, self.calls = dim, 0, 0

    @property
    def model(self):
        self.loads += 1
        return self

    def encode(self, texts):
        self.calls += 1
        return np.ones((len(texts), self.dim), np.float32)


def test_nirt_warmup_pays_first_use_costs(nirt_router):
    router, _, _ = nirt_router
    enc = _StubEncoder(8)
    router._encoder_cache[router._query_pathway] = enc
    timings = router.warmup()
    assert {"data", "encoder", "pool_refs", "costs", "route_text"} <= set(timings)
    assert router._pool_ref_cache is not None
    assert router._cost_cache is not None and enc.calls == 1
    # a routed request after warm-up reuses everything
    router.route_text(["hello"])
    assert enc.calls == 2


def test_cost_by_model_keeps_only_finite_costs():
    class _NaNCost(MatrixRouter):                 # a data-backed router's cost vector can carry NaN
        @property
        def default_model_costs(self):
            return np.array([0.01, np.nan, 0.03])

    r = _NaNCost(pd.DataFrame({"a": [0.1], "b": [0.2], "c": [0.3]}, index=["q"]))
    assert r.cost_by_model() == {"a": 0.01, "c": 0.03}
    assert MatrixRouter(pd.DataFrame({"a": [0.1]}, index=["q"])).cost_by_model() == {}


def test_routing_result_row_scores():
    r = MatrixRouter(pd.DataFrame({"a": [0.1, 0.5], "b": [0.2, 0.4]}, index=["q1", "q2"]))
    rr = r.route(["q1", "q2"])
    assert rr.row_scores(1) == {"a": 0.5, "b": 0.4} and rr.row_scores() == {"a": 0.1, "b": 0.2}
