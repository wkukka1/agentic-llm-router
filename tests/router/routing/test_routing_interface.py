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


def test_call_is_route():
    true_df, _ = _mats()
    r = MatrixRouter(true_df)
    assert r(list(true_df.index)).selected_model_ids == r.route(list(true_df.index)).selected_model_ids


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
# XA-01: cost-aware evaluation must match the served router's own cost vector #
# --------------------------------------------------------------------------- #
def test_evaluate_router_uses_router_default_model_costs():
    """The cost-aware selection reported by evaluate_router must be the one
    router.route(lam=...) would actually make (router.default_model_costs),
    not one recomputed from the eval-split cost matrix -- which can point the
    opposite way entirely, as it does here."""
    scores = pd.DataFrame([[0.90, 0.88]], index=["q"], columns=["exp", "cheap"])
    r = MatrixRouter(scores, name="r", model_costs=[1.0, 0.0])  # exp is expensive
    true_df = pd.DataFrame([[1.0, 1.0]], index=["q"], columns=["exp", "cheap"])
    cost_df = pd.DataFrame([[0.0, 1.0]], index=["q"], columns=["exp", "cheap"])  # eval says the opposite

    assert r.route(["q"], lam=0.1).selected_model_ids == ["cheap"]
    res = evaluate_router(r, true_df, cost_df, lam=0.1)
    assert res["_per_query"]["selected_model_id"].iloc[0] == "cheap"


def test_compare_routers_uses_router_default_model_costs():
    scores = pd.DataFrame([[0.90, 0.88]], index=["q"], columns=["exp", "cheap"])
    r = MatrixRouter(scores, name="r", model_costs=[1.0, 0.0])
    true_df = pd.DataFrame([[1.0, 1.0]], index=["q"], columns=["exp", "cheap"])
    cost_df = pd.DataFrame([[0.0, 1.0]], index=["q"], columns=["exp", "cheap"])
    summary, _ = compare_routers([r], true_df, cost_df, lam=0.1,
                                 include_hard_oracle=False, include_random=False)
    row = summary[summary.strategy == "r (cost-aware lam=0.1)"].iloc[0]
    # picked "cheap" (per router.route's own cost vector) -- its eval-matrix
    # cost (1.0) shows up, not "exp"'s (0.0), which the eval-split-mean cost
    # vector would have picked instead
    assert row["mean_selected_cost_per_1k"] == pytest.approx(1000.0)


def test_compare_routers_rejects_duplicate_names():
    true_df, cost_df = _mats()
    routers = [MatrixRouter(true_df, name="dup"), MatrixRouter(-true_df, name="dup")]
    with pytest.raises(ValueError, match="duplicate"):
        compare_routers(routers, true_df, cost_df)


def test_compare_routers_handles_non_string_query_ids():
    """XD-08: compare_routers must align via RouterModel.aligned_scores (which
    stringifies ids the same way predict_scores does), not a raw reindex by
    the caller's original ids -- a non-string query id previously produced an
    all-NaN row that silently fell back to column 0."""
    true_df = pd.DataFrame([[0.1, 0.9]], index=[1], columns=["A", "B"])
    scores = pd.DataFrame([[0.1, 0.9]], index=["1"], columns=["A", "B"])
    r = MatrixRouter(scores, name="r")
    summary, _ = compare_routers([r], true_df, include_hard_oracle=False, include_random=False)
    row = summary[summary.strategy == "r (quality)"].iloc[0]
    assert row["mean_regret"] == pytest.approx(0.0)


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


def test_nirt_router_scores_one_column_per_model(nirt_router):
    router, _, _ = nirt_router
    scores = router.predict_scores(Q_IDS)
    assert scores.shape == (6, 3)
    assert list(scores.columns) == M_IDS


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
