"""The modular routing interface (router.routing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.routing.oracle import compare_routers, evaluate_router
from router.routing import (
    REGISTRY,
    MatrixRouter,
    RandomRouter,
    Router,
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


def test_call_is_route():
    true_df, _ = _mats()
    r = MatrixRouter(true_df)
    assert r(list(true_df.index)).selected_model_ids == r.route(list(true_df.index)).selected_model_ids


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
# evaluate_router()                                                           #
# --------------------------------------------------------------------------- #
def test_evaluate_perfect_router_zero_regret():
    true_df, cost_df = _mats()
    out = evaluate_router(MatrixRouter(true_df, name="truth"), true_df, cost_df)
    assert out["strategy"] == "truth"
    assert out["mean_regret"] == pytest.approx(0.0)
    assert out["oracle_hit_rate_any_best"] == pytest.approx(1.0)


def test_evaluate_bad_router_positive_regret():
    true_df, cost_df = _mats()
    out = evaluate_router(MatrixRouter(-true_df, name="worst"), true_df, cost_df)
    assert out["mean_regret"] > 0.0


def test_evaluate_aligns_pool_subset():
    true_df, cost_df = _mats()
    sub = true_df[["GPT-A", "GPT-C"]]
    out = evaluate_router(MatrixRouter(sub), sub, cost_df[["GPT-A", "GPT-C"]])
    assert out["n_queries"] == 3
    assert set(out["selected_model_mix"]) <= {"GPT-A", "GPT-C"}


def test_evaluate_raises_when_outcomes_lack_a_pool_model():
    true_df, cost_df = _mats()
    router = MatrixRouter(true_df)                      # pool = A, B, C
    with pytest.raises(ValueError, match="GPT-B"):
        evaluate_router(router, true_df[["GPT-A", "GPT-C"]], cost_df)
    with pytest.raises(ValueError, match="cost_df.*GPT-C"):
        evaluate_router(router, true_df, cost_df[["GPT-A", "GPT-B"]])


# --------------------------------------------------------------------------- #
# registry                                                                    #
# --------------------------------------------------------------------------- #
def test_builtin_kinds_registered():
    assert {"matrix", "nirt", "knn", "mlp", "random"} <= set(REGISTRY)


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
        class _NoKind(Router):
            def predict_scores(self, query_ids):
                ...


def test_register_custom_router_roundtrips():
    @register
    class _ConstRouter(Router):
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
# RandomRouter + compare_routers                                              #
# --------------------------------------------------------------------------- #
def test_random_router_is_deterministic():
    a = RandomRouter(M, seed=7).predict_scores(["q1", "q2"])
    b = RandomRouter(M, seed=7).predict_scores(["q1", "q2"])
    pd.testing.assert_frame_equal(a, b)


def test_compare_routers_adds_oracle_and_floor():
    true_df, cost_df = _mats()
    routers = [MatrixRouter(true_df, name="truth"), MatrixRouter(-true_df, name="worst")]
    summary, detail = compare_routers(routers, true_df, cost_df, lam=0.5)
    assert (summary["strategy"] == "hard oracle (upper bound)").any()
    assert (summary["strategy"] == "random").any()
    truth = summary[summary.strategy == "truth (quality)"].iloc[0]
    worst = summary[summary.strategy == "worst (quality)"].iloc[0]
    assert truth["mean_regret"] <= worst["mean_regret"]
    assert "truth (quality)" in detail


# --------------------------------------------------------------------------- #
# NIRTRouter smoke test (needs torch)                                         #
# --------------------------------------------------------------------------- #
def test_nirt_router_scores_and_routes():
    torch = pytest.importorskip("torch")
    from router.nirt.model import build_model
    from router.routing import NIRTRouter

    class _Store:
        def __init__(self, ids, mat, field):
            self._index = {str(i): r for r, i in enumerate(ids)}
            self.matrix = np.asarray(mat, np.float32)
            self.id_field = field

        @property
        def dim(self):
            return self.matrix.shape[1]

        def __contains__(self, i):
            return str(i) in self._index

        def rows_of(self, ids):
            return np.fromiter((self._index[str(i)] for i in ids), dtype=np.int64, count=len(ids))

        def get(self, i):
            return np.asarray(self.matrix[self._index[str(i)]])

    rng = np.random.default_rng(0)
    q_ids = [f"q{i}" for i in range(6)]
    m_ids = ["m0", "m1", "m2"]
    q_store = _Store(q_ids, rng.standard_normal((6, 8)), "query_id")
    m_store = _Store(m_ids, rng.standard_normal((3, 4)), "model_id")

    class _Data:
        def query_embeddings(self, pw):
            return q_store

        def profile_embeddings(self, pw):
            return m_store

        def nirt_observations(self):
            return pd.DataFrame({"split": ["train"] * 3, "model_id": m_ids, "cost": [0.1, 0.2, 0.3]})

    model = build_model({"model_params": "projected", "dim": 2}, n_models=3, query_dim=8, profile_dim=4)
    r = NIRTRouter(model, {"m0": 0, "m1": 1, "m2": 2}, data=_Data(), name="nirt-smoke")

    scores = r.predict_scores(q_ids)
    assert scores.shape == (6, 3)
    assert list(scores.columns) == m_ids
    assert ((scores.to_numpy() >= 0) & (scores.to_numpy() <= 1)).all()

    res = r.route(q_ids)
    assert res.selected.shape == (6,)
    # default cost vector comes from the (stub) observation table
    np.testing.assert_allclose(r.default_model_costs, [0.1, 0.2, 0.3])
    res_cost = r.route(q_ids, lam=5.0)
    assert res_cost.model_costs is not None

    # text scoring path: _score_embeddings forwards raw [N, d] query embeddings
    assert NIRTRouter.can_route_text is True
    e_q = rng.standard_normal((4, 8))
    text_scores = r._score_embeddings(e_q)
    assert text_scores.shape == (4, 3)
    assert list(text_scores.columns) == m_ids
    assert ((text_scores.to_numpy() >= 0) & (text_scores.to_numpy() <= 1)).all()
    # a short embedding with no feature variant to explain it is a wrong encoder
    with pytest.raises(ValueError, match="does not match model query_dim"):
        r._score_embeddings(rng.standard_normal((2, 6)))

    # a run trained with structured query features -> exactly the feature block is zero-padded
    class _Feat:
        dim = 2

    class _FeatData(_Data):
        def query_features(self, name):
            return _Feat()

    rf = NIRTRouter(model, {"m0": 0, "m1": 1, "m2": 2}, data=_FeatData(),
                    query_features="default", name="nirt-feat")
    with pytest.warns(UserWarning, match="zero-fills"):
        padded = rf._score_embeddings(rng.standard_normal((2, 6)))
    assert padded.shape == (2, 3)
    with pytest.raises(ValueError):
        rf._score_embeddings(rng.standard_normal((2, 5)))
