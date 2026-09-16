"""Regressions for the router.* findings in code_review/ (IDs in each test name's
docstring). Each test fails on the pre-fix behaviour described there."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from router.agentic import AgenticRouter, CallableClient, ClientRegistry, HeuristicTriage, NaiveDecomposer
from router.agentic.llm_clients import LLMResponse, guess_provider, message_text
from router.agentic.result import AgenticResult
from router.agentic.triage import TriageDecision
from router.config import Config, coerce_auto_bool, coerce_hidden
from router.constraints import RoutingConstraints
from router.context import RoutingContext, RoutingRequest
from router.data.nirt import NIRTDataset
from router.decision import ModelScore
from router.embeddings.encoder import EmbeddingStore
from router.execution.budget import BudgetLedger
from router.execution.results import ExecutionResult
from router.execution.task import AgentTask, TaskStatus
from router.llm.client import LLMClient
from router.llm.profile import LLMProfile
from router.llm.registry import LLMRegistry
from router.nirt.routing_decision import routing_decision
from router.nirt.shrinkage import novelty_weight
from router.policy import DefaultRoutingPolicy
from router.router import RoutingPipeline
from router.routing import MatrixRouter, RandomRouter, Router
from router.routing.registry import REGISTRY, register

POOL = ["a", "b", "c"]


def _store(ids, dim, field="query_id", seed=0):
    mat = np.random.default_rng(seed).standard_normal((len(ids), dim)).astype(np.float32)
    return EmbeddingStore(ids, mat, {"id_field": field}, field)


class _TextRouter(Router):
    kind = "review_text"
    can_route_text = True

    def __init__(self, row, model_ids=POOL):
        super().__init__(model_ids)
        self._row = row

    def predict_scores(self, query_ids):
        return pd.DataFrame([self._row] * len(query_ids), index=list(query_ids), columns=self._model_ids)

    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame([self._row] * len(texts), columns=self._model_ids)


# --------------------------------------------------------------------------- #
# router (top level)                                                          #
# --------------------------------------------------------------------------- #
def test_empty_registry_means_no_candidates():
    """RT-01"""
    decision = RoutingPipeline(_TextRouter([0.1, 0.9, 0.5]), llm_registry=LLMRegistry()).route(
        RoutingRequest(request_id="r", prompt="x"))
    assert decision.ranked_models == []


def test_nan_score_never_ranks_first():
    """RT-02"""
    ctx = RoutingContext(request=RoutingRequest(request_id="r", prompt="x"), signals=None)
    scores = [ModelScore(model=m, score=q, expected_quality=q)
              for m, q in zip(POOL, [float("nan"), 0.2, 0.9])]
    ranked = DefaultRoutingPolicy().decide(ctx, scores).ranked_models
    assert [s.model for s in ranked][:2] == ["c", "b"]
    # and the pipeline drops non-finite scores altogether
    d = RoutingPipeline(_TextRouter([float("nan"), 0.2, 0.9])).route(RoutingRequest(request_id="r", prompt="x"))
    assert [s.model.model_id for s in d.ranked_models] == ["c", "b"]


def test_min_quality_is_enforced():
    """RR-05"""
    d = RoutingPipeline(_TextRouter([0.1, 0.9, 0.5])).route(
        RoutingRequest(request_id="r", prompt="x", constraints=RoutingConstraints(min_quality=0.4)))
    assert {s.model.model_id for s in d.ranked_models} == {"b", "c"}


def test_config_survives_copy_and_pickle():
    """RT-04"""
    cfg = Config({"a": {"b": 1}})
    assert copy.deepcopy(cfg).get("a.b") == 1
    assert pickle.loads(pickle.dumps(cfg)).get("a.b") == 1


def test_string_config_flags():
    """RT-05"""
    assert coerce_auto_bool("false", True) is False
    assert coerce_auto_bool("auto", True) is True
    assert coerce_hidden("0") is None and coerce_hidden("null") is None
    with pytest.raises(ValueError):
        coerce_auto_bool("maybe", True)


# --------------------------------------------------------------------------- #
# router.routing / routing_decision                                           #
# --------------------------------------------------------------------------- #
def _matrix():
    return MatrixRouter(pd.DataFrame([[0.9, 0.5, 0.1], [0.1, 0.2, 0.8]], index=["q1", "q2"], columns=POOL))


def test_partial_eligibility_mask_fails_closed():
    """RR-02"""
    mask = pd.DataFrame([[False, True, True]], index=["q1"], columns=POOL)
    res = _matrix().route(["q1", "q2"], eligible=mask)
    assert res.selected_model_ids[0] == "b"
    assert res.fallback.tolist() == [False, True]      # q2 has no mask row -> nothing eligible


def test_fallback_rows_are_flagged():
    """RR-03"""
    res = _matrix().route(["q1", "missing"])
    assert res.fallback.tolist() == [False, True] and res.any_fallback


def test_nan_cost_is_rejected_and_never_selected():
    """RR-04"""
    with pytest.raises(ValueError, match="finite"):
        _matrix().route(["q1"], lam=0.1, model_costs=[0.0, float("nan"), 0.0])
    sel = routing_decision(np.array([[0.9, 0.5, 0.1]]), lam=0.1, model_costs=np.array([0.0, np.nan, 0.0]))
    assert sel.tolist() == [0]


def test_lam_without_costs_warns_and_records_zero():
    """RR-14"""
    with pytest.warns(UserWarning, match="no model cost vector"):
        res = _matrix().route(["q1"], lam=0.5)
    assert res.lam == 0.0


def test_int_labelled_matrix_routes():
    """RR-07"""
    r = MatrixRouter(pd.DataFrame([[0.1, 0.9]], index=[101], columns=[1, 2]))
    res = r.route([101])
    assert res.selected_model_ids == ["2"] and not res.any_fallback


def test_iterator_pool_is_materialised():
    """RR-08"""
    assert RandomRouter(m for m in POOL).model_ids == POOL


def test_random_router_depends_on_query_not_position():
    """RR-13"""
    r = RandomRouter(POOL, seed=3)
    a = r.predict_scores(["x", "y"])
    b = r.predict_scores(["y", "x"])
    np.testing.assert_allclose(a.loc["x"], b.loc["x"])
    assert not np.allclose(r.predict_scores(["x"]).to_numpy(), r.predict_scores(["y"]).to_numpy())


def test_reregistering_same_qualname_is_allowed():
    """RR-16"""
    def make():
        class ReloadProbe(Router):
            kind = "reload_probe"

            def predict_scores(self, query_ids):  # pragma: no cover
                raise NotImplementedError
        return ReloadProbe

    try:
        register(make())
        register(make())          # a reload: same module + qualname
    finally:
        REGISTRY.pop("reload_probe", None)


# --------------------------------------------------------------------------- #
# router.nirt / router.data                                                   #
# --------------------------------------------------------------------------- #
def _obs(qids, mids):
    rows = [(q, m) for q in qids for m in mids]
    return pd.DataFrame({
        "query_id": [q for q, _ in rows], "model_id": [m for _, m in rows],
        "target": 1.0, "cost": np.nan, "metric_type": "accuracy", "source": "t",
    })


def test_projected_centering_is_batch_invariant():
    """RN-01"""
    torch = pytest.importorskip("torch")
    from router.nirt.model import NIRTModel
    from router.nirt.predict import predict_dataset

    torch.manual_seed(0)
    qids, mids = [f"q{i}" for i in range(5)], ["m0", "m1", "m2"]
    ds = NIRTDataset(_obs(qids, mids), _store(qids, 6), _store(mids, 4, "model_id", seed=1))
    model = NIRTModel(query_dim=6, dim=2, model_params="projected", profile_dim=4,
                      center_discrimination=True)
    _, p_big = predict_dataset(model, ds, {m: i for i, m in enumerate(mids)}, batch_size=1000)
    _, p_small = predict_dataset(model, ds, {m: i for i, m in enumerate(mids)}, batch_size=2)
    np.testing.assert_allclose(p_big, p_small, rtol=1e-5)
    assert model._a_ref is None           # reference restored after inference


def test_empty_bank_raises():
    """RN-02"""
    with pytest.raises(ValueError, match="non-empty"):
        novelty_weight(np.ones((2, 3)), np.zeros((0, 3)))


def test_fixed_warmup_alpha_is_validated():
    """RN-03"""
    pytest.importorskip("torch")
    from router.nirt.components import WarmupBlender

    with pytest.raises(ValueError):
        WarmupBlender(enabled=True, alpha=1.5)


def test_empty_frame_with_feature_store():
    """RD-01"""
    qids, mids = ["q0", "q1"], ["m0"]
    ds = NIRTDataset(_obs(qids, mids).iloc[:0], _store(qids, 3), _store(mids, 2, "model_id"),
                     feature_store=_store(qids, 2))
    assert len(ds) == 0 and ds.query_dim == 5


def test_int_ids_match_str_stores():
    """RD-02 / RE-07"""
    ds = NIRTDataset(_obs([1, 2], [7]), _store([1, 2], 3), _store([7], 2, "model_id"))
    assert len(ds) == 2 and ds.dropped == 0


# --------------------------------------------------------------------------- #
# router.agentic                                                              #
# --------------------------------------------------------------------------- #
def _costed(model_ids):
    return ClientRegistry({m: CallableClient(m, lambda p, m=m, **kw: LLMResponse(f"{m}:{p}", m, cost=1.0))
                           for m in model_ids})


def test_nested_orchestration_counts_each_leaf_once():
    """RA-01"""
    splits = {"root": ["sub one here", "sub two here"], "sub one here": ["leaf a here", "leaf b here"]}
    agent = AgenticRouter(
        _TextRouter([0.9, 0.9, 0.9]), _costed(POOL), max_depth=2,
        triage=lambda p, **kw: TriageDecision("orchestrate", kw["model_id"], 0.1, "forced", kw.get("scores", {})),
        decomposer=lambda p: splits.get(p, [p]),
    )
    res = agent.run("root")
    assert sum(1 for s in res.steps for _ in s.leaves()) == 3
    assert res.cost == pytest.approx(3.0)


def test_nan_quality_is_not_strong():
    """RA-02"""
    d = HeuristicTriage()("What is 2+2?", model_id="a", predicted_quality=float("nan"))
    assert d.orchestrate


def test_triage_uses_pool_best_not_cost_selected():
    """RA-06"""
    from router.agentic.triage import best_from_result

    r = MatrixRouter(pd.DataFrame([[0.9, 0.3]], index=["q"], columns=["big", "cheap"]), model_costs=[10.0, 0.0])
    mid, q, _ = best_from_result(r.route(["q"], lam=1.0))
    assert mid == "cheap" and q == pytest.approx(0.9)


def test_decimals_are_not_enumerations():
    """RA-12"""
    assert HeuristicTriage()("What is 1.5 plus 2.25?", model_id="m", predicted_quality=0.9).mode == "single"
    assert HeuristicTriage()("Do this:\n1. alpha\n2. beta", model_id="m", predicted_quality=0.9).orchestrate


def test_abbreviations_do_not_split():
    """RA-07"""
    p = "Use numpy, e.g. arrays, to compute the mean of a list of numbers."
    assert NaiveDecomposer()(p) == [p]
    assert len(NaiveDecomposer()("Explain quicksort. Then write it in Rust.")) == 2


def test_failed_subcall_keeps_paid_siblings():
    """RA-05"""
    def boom(p, **kw):
        if "second" in p:
            raise TimeoutError("provider timeout")
        return LLMResponse("ok", "a", cost=1.0)

    reg = ClientRegistry({m: CallableClient(m, boom) for m in POOL})
    res = AgenticRouter(_TextRouter([0.9, 0.9, 0.9]), reg).run("first part here and then second part here")
    assert res.mode == "orchestrated" and len(res.errors()) == 1
    assert res.cost == pytest.approx(1.0)


def test_max_calls_caps_fanout():
    """RA-17"""
    res = AgenticRouter(_TextRouter([0.9, 0.9, 0.9]), _costed(POOL), max_calls=1).run(
        "one thing here and then two thing here and then three thing here")
    assert len(res.errors()) == 2


def test_models_used_and_summary_edge_cases():
    """RA-16 / RA-20"""
    res = AgenticResult(prompt="p", mode="orchestrated", answer="", triage_reason="r", selected_model_id="z")
    assert res.models_used() == []
    assert "q~?" in AgenticResult(prompt="p", mode="single", answer="", triage_reason="r").summary()


def test_block_content_and_provider_boundaries():
    """RA-11 / RA-19"""
    class Msg:
        content = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "hi"}]

    assert message_text(Msg()) == "hi"
    assert guess_provider("o1-mini") == "openai" and guess_provider("yolo1") is None


# --------------------------------------------------------------------------- #
# router.execution / router.llm                                               #
# --------------------------------------------------------------------------- #
def test_ledger_rejects_nan_and_negative():
    """RX-02"""
    ledger = BudgetLedger("root", 5.0)
    for bad in (float("nan"), -1.0):
        with pytest.raises(ValueError):
            ledger.reserve("t", bad)
    assert ledger.remaining() == 5.0


def test_aggregate_children_reports_failures():
    """RX-03 / RX-09 / RX-14"""
    ok = AgentTask("c1", "p", "root", "x", status=TaskStatus.COMPLETED,
                   result=ExecutionResult(output="a", actual_latency=2.0))
    crashed = AgentTask("c2", "p", "root", "y", status=TaskStatus.FAILED)
    parent = AgentTask("p", None, "root", "z", children=[ok, crashed])
    agg = parent.aggregate_children()
    assert agg.success is False and agg.error_code == "child_failed"
    assert AgentTask("e", None, "root", "z").aggregate_children().success is False
    assert ExecutionResult(error_code="timeout").success is False


def test_duplicate_profile_raises_and_client_prices_tokens():
    """RL-06 / RL-01"""
    reg = LLMRegistry()
    prof = LLMProfile(name="m", provider="openai", model_id="m",
                      input_cost_per_token=0.001, output_cost_per_token=0.002)
    reg.register(prof)
    with pytest.raises(ValueError):
        reg.register(prof)
    assert LLMClient(adapter=None, profile=prof).price(100, 50) == pytest.approx(0.2)
