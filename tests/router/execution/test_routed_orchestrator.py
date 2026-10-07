"""RouterPlanForecaster, RoutedOrchestrator and the AgenticRouter bridge
(router.execution.forecast / router.execution.routed)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

from router.agentic import AgenticRouter, ClientRegistry, LLMClient, LLMResponse
from router.constraints import OptimizationObjective
from router.execution.budget import BudgetLedger, ExecutionLimits
from router.execution.forecast import RouterPlanForecaster
from router.execution.orchestrator import OrchestratorConfig
from router.execution.plan import Plan, PlanStep
from router.execution.routed import PlanOrchestratorBridge, RoutedOrchestrator, plan_orchestrator
from router.execution.task import AgentTask, TaskStatus
from router.llm.adapters.fakes import CallableAdapter
from router.llm.profile import LLMProfile
from router.routing import RouterModel

POOL = ["small", "big", "coder"]
COMPOUND = "Explain ownership, and then write a borrow example."


class TextRouter(RouterModel):
    """big 0.9 > coder 0.8 > small 0.7 for every prompt; counts route_text calls."""

    kind = "test_text"
    can_route_text = True

    def __init__(self, costs=None):
        super().__init__(POOL, name="t")
        self.text_calls: list[list[str]] = []
        self._costs = None if costs is None else np.asarray(costs, float)

    @property
    def default_model_costs(self):
        return self._costs

    def predict_scores(self, query_ids):
        return pd.DataFrame(0.5, index=[str(q) for q in query_ids], columns=POOL)

    def predict_scores_text(self, texts, *, encoder=None):
        self.text_calls.append(list(texts))
        return pd.DataFrame([[0.7, 0.9, 0.8]] * len(texts), index=range(len(texts)), columns=POOL)


def _clients(*, fail=(), delay=0.0, cost=0.01, sent=None):
    def make(m):
        def fn(prompt, **kw):
            if sent is not None:
                sent.append((m, prompt))
            if delay:
                time.sleep(delay)
            if m in fail:
                raise RuntimeError(f"{m} is down")
            return LLMResponse(f"{m} answers", cost=cost, input_tokens=5, output_tokens=2)
        return LLMClient(CallableAdapter(m, fn))
    return ClientRegistry({m: make(m) for m in POOL})


def _orchestrator(agent, *, budget=float("inf"), **kw):
    ledger = BudgetLedger("root", budget)
    return RoutedOrchestrator(
        OrchestratorConfig("plan", "0", llm=None, ledger=ledger, limits=ExecutionLimits(max_cost=budget)),
        agent=agent, plan_forecaster=kw.pop("forecaster", RouterPlanForecaster(agent.router)), **kw)


def _task(prompt=COMPOUND):
    return AgentTask("root", None, "root", prompt, depth=0)


def _decomposer(parts):
    return lambda prompt: list(parts)


# --------------------------------------------------------------------------- #
# RouterPlanForecaster                                                        #
# --------------------------------------------------------------------------- #
def _profile(mid, inp=None, out=None):
    return LLMProfile(name=mid, provider="x", model_id=mid, input_cost_per_token=inp, output_cost_per_token=out)


def test_forecast_cost_prefers_prices_then_router_costs():
    router = TextRouter(costs=[0.001, 0.01, 0.005])
    plan = Plan("p", "t", steps=[
        PlanStep("a", "x", _profile("small", 1e-6, 2e-6), estimated_input_tokens=100, estimated_output_tokens=50),
        PlanStep("b", "y", _profile("big"), estimated_input_tokens=100, estimated_output_tokens=50),
    ])
    fc = RouterPlanForecaster(router).forecast_cost(plan)
    assert fc.total_cost == pytest.approx(100e-6 + 100e-6 + 0.01)
    assert fc.confidence == 0.5                                    # one step fell back to router costs


def test_forecast_cost_unknown_model_has_zero_confidence():
    plan = Plan("p", "t", steps=[PlanStep("a", "x", _profile("mystery"))])
    assert RouterPlanForecaster(TextRouter()).forecast_cost(plan).confidence == 0.0


def test_forecast_quality_and_critical_path_latency():
    plan = Plan("p", "t", steps=[
        PlanStep("a", "x", _profile("small"), predicted_quality=0.6),
        PlanStep("b", "y", _profile("big"), predicted_quality=0.9),
        PlanStep("c", "z", _profile("small"), predicted_quality=float("nan"), dependencies=["a"]),
    ])
    f = RouterPlanForecaster(latency_table={"small": 1.0, "big": 3.0})
    assert f.forecast_quality(plan) == pytest.approx(0.75)
    assert f.forecast_latency(plan) == pytest.approx(3.0)          # max(a->c = 2, b = 3)
    score = f.score_plan(plan, OptimizationObjective())
    assert plan.estimated_latency == pytest.approx(3.0) and score.within_budget is None


# --------------------------------------------------------------------------- #
# plan()                                                                      #
# --------------------------------------------------------------------------- #
def test_plan_routes_all_subprompts_in_one_batch():
    router = TextRouter(costs=[0.001, 0.01, 0.005])
    agent = AgenticRouter(router, _clients(), decomposer=_decomposer(["a?", "b?", "c?"]))
    orch = _orchestrator(agent)
    orch.task = _task()
    plan = orch.plan(orch.task)
    assert router.text_calls == [["a?", "b?", "c?"]]               # ONE router call
    assert [s.model.model_id for s in plan.steps] == ["big"] * 3
    assert plan.steps[0].alternatives == ["coder", "small"]
    assert plan.steps[0].predicted_quality == pytest.approx(0.9)
    assert plan.estimated_cost == pytest.approx(0.03) and plan.status.value == "evaluated"


# --------------------------------------------------------------------------- #
# execute / run                                                               #
# --------------------------------------------------------------------------- #
def test_independent_steps_run_concurrently():
    agent = AgenticRouter(TextRouter(), _clients(delay=0.2), decomposer=_decomposer(["a", "b", "c"]))
    orch = _orchestrator(agent)
    t0 = time.perf_counter()
    res = orch.run(_task())
    wall = time.perf_counter() - t0
    assert res.success and len(orch.subcalls) == 3
    assert wall < 0.45                                            # ~0.2 s, not ~0.6 s sequential
    assert res.actual_cost == pytest.approx(0.03) and orch.ledger.spent == pytest.approx(0.03)


def test_max_calls_is_exact_across_worker_threads():
    """3 parallel steps under max_calls=2: exactly two calls go through; the
    third (and its replan retry) is refused by the shared, locked budget."""
    sent = []
    agent = AgenticRouter(TextRouter(), _clients(sent=sent), decomposer=_decomposer(["a", "b", "c"]),
                          orchestrator=plan_orchestrator(), max_calls=2)
    res = agent.run(COMPOUND)
    errors = res.errors()
    assert len(sent) == 2
    assert len(errors) == 1 and "max_calls" in errors[0].error


def test_sequential_steps_see_earlier_results():
    sent = []
    agent = AgenticRouter(TextRouter(), _clients(sent=sent), decomposer=_decomposer(["first", "second"]))
    orch = _orchestrator(agent, sequential=True)
    assert orch.run(_task()).success
    assert sent[0][1] == "first"
    assert "big answers" in sent[1][1] and sent[1][1].endswith("Now: second")


def test_failed_step_is_replanned_onto_its_next_best_model():
    sent = []
    agent = AgenticRouter(TextRouter(), _clients(fail={"big"}, sent=sent), decomposer=_decomposer(["a", "b"]))
    orch = _orchestrator(agent, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=1)
    res = orch.run(_task())
    assert res.success
    assert [sc.model_id for sc in orch.subcalls] == ["coder", "coder"]
    assert orch.task.plan.revision == 1


def test_replan_does_not_rerun_succeeded_steps():
    sent = []
    agent = AgenticRouter(TextRouter(), _clients(sent=sent), decomposer=_decomposer(["ok", "flaky"]))
    flaky = {"n": 0}
    orig = agent._call_model

    def call(prompt, model_id, depth, **kw):
        if prompt == "flaky" and flaky["n"] == 0:
            flaky["n"] += 1
            return orig(prompt, "missing-model-raises", depth, **kw)
        return orig(prompt, model_id, depth, **kw)

    agent._call_model = call
    def unknown(model_id):
        raise KeyError(model_id)

    agent.clients = ClientRegistry({m: agent.clients.get(m) for m in POOL}, default_factory=unknown)
    orch = _orchestrator(agent, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=1)
    assert orch.run(_task()).success
    assert [p for _, p in sent].count("ok") == 1


def test_over_budget_plan_is_refused_before_any_call():
    sent = []
    agent = AgenticRouter(TextRouter(costs=[0.001, 0.01, 0.005]), _clients(sent=sent),
                          decomposer=_decomposer(["a", "b"]))
    orch = _orchestrator(agent, budget=0.015)                       # plan needs 0.02
    res = orch.run(_task())
    assert res.error_code == "over_budget" and orch.task.status is TaskStatus.TERMINATED
    assert sent == [] and orch.ledger.reserved == 0.0


# --------------------------------------------------------------------------- #
# the AgenticRouter bridge                                                    #
# --------------------------------------------------------------------------- #
def test_bridge_end_to_end_through_agentic_router():
    agent = AgenticRouter(TextRouter(), _clients(cost=0.02), orchestrator=plan_orchestrator())
    res = agent.run(COMPOUND)
    assert isinstance(agent.orchestrator, PlanOrchestratorBridge)
    assert res.mode == "orchestrated" and res.orchestrator == "plan"
    assert len(res.steps) == 2 and res.cost == pytest.approx(0.04)
    assert all(s.meta["step_id"].startswith("plan-") for s in res.steps)


def test_bridge_alpha_override_reaches_every_plan_step():
    """Regression: _route_batch used to route every plan step with the
    agent's static lam, silently ignoring a run(prompt, alpha=...) override
    -- unlike RecursiveOrchestrator, whose _solve -> route_decision path
    already read it. big (q=0.9) is the cost-blind pick; a strongly
    cost-averse alpha must instead pick small (cheapest) for every step."""
    agent = AgenticRouter(TextRouter(costs=[0.001, 0.01, 0.005]), _clients(),
                          orchestrator=plan_orchestrator())
    assert agent.run(COMPOUND).models_used() == ["big"]          # default: quality-only
    res = agent.run(COMPOUND, alpha=0.01)                        # override: strongly cost-averse
    assert res.mode == "orchestrated"
    assert res.models_used() == ["small"]


def test_bridge_sends_conversation_history_to_every_step():
    sent = []
    msgs = [{"role": "user", "content": "I am learning Rust."},
            {"role": "assistant", "content": "Great."},
            {"role": "user", "content": COMPOUND}]
    agent = AgenticRouter(TextRouter(), _clients(sent=sent), orchestrator=plan_orchestrator())
    assert agent.run(msgs).mode == "orchestrated"
    assert len(sent) == 2
    assert all(isinstance(p, list) and p[0]["content"] == "I am learning Rust." for _, p in sent)


def test_bridge_single_subprompt_collapses_to_single_mode():
    agent = AgenticRouter(TextRouter(), _clients(), orchestrator=plan_orchestrator(),
                          decomposer=lambda p: [p],
                          triage=lambda p, **kw: __import__("router.agentic", fromlist=["TriageDecision"])
                          .TriageDecision("orchestrate", "big", 0.9, "forced", kw.get("scores") or {}))
    res = agent.run("What is 2+2?")
    assert res.mode == "single" and res.selected_model_id == "big"


def test_bridge_over_budget_returns_a_failed_subcall():
    agent = AgenticRouter(TextRouter(costs=[0.001, 0.01, 0.005]), _clients(),
                          orchestrator=plan_orchestrator(budget=0.001))
    res = agent.run(COMPOUND)
    assert res.mode == "orchestrated" and "over budget" in res.answer
    assert res.errors() and res.errors()[0].error == "over_budget"


# --------------------------------------------------------------------------- #
# adaptive plan orchestrator                                                  #
# --------------------------------------------------------------------------- #
class _PlanChat:
    """Planner stand-in: replies ``reply`` to every plan request."""

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def invoke(self, text):
        self.calls += 1
        return self.reply


def _planning_agent(reply, *, clients=None, router=None, **kw):
    from router.agentic import LLMDecomposer
    return AgenticRouter(router or TextRouter(), clients or _clients(),
                         decomposer=LLMDecomposer(_PlanChat(reply)), **kw)


DEP_PLAN = ("1. Implement quicksort in Rust.\n"
            "2. [after 1] Analyse the time complexity of the Rust quicksort from step 1.")


def test_planner_dependencies_become_plan_dependencies():
    orch = _orchestrator(_planning_agent(DEP_PLAN))
    orch.task = _task()
    plan = orch.plan(orch.task)
    assert [s.dependencies for s in plan.steps] == [[], ["root.0"]]


def test_malformed_planner_dependencies_still_give_a_valid_plan():
    orch = _orchestrator(_planning_agent(
        "1. [after 2] First task here.\n2. [after 1, 2] Second task here."))
    orch.task = _task()
    plan = orch.plan(orch.task)
    plan.validate()
    assert [s.dependencies for s in plan.steps] == [[], ["root.0"]]


def test_dependent_step_receives_its_dependency_output():
    sent = []
    agent = _planning_agent(DEP_PLAN, clients=_clients(sent=sent))
    assert _orchestrator(agent).run(_task()).success
    assert sent[0][1] == "Implement quicksort in Rust."
    assert "big answers" in sent[1][1]
    assert sent[1][1].endswith("Now: Analyse the time complexity of the Rust quicksort from step 1.")


def test_failed_dependency_output_is_not_passed_on():
    sent = []
    agent = _planning_agent(DEP_PLAN, clients=_clients(sent=sent, fail={"big"}))
    orch = _orchestrator(agent, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=0)
    orch.run(_task())
    step2 = [p for _, p in sent if "Analyse the time complexity" in p]
    assert step2 and all("[error" not in p and "Results of earlier steps" not in p for p in step2)
