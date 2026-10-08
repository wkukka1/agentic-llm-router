"""RouterPlanForecaster, RoutedOrchestrator and the AgenticRouter bridge
(router.execution.forecast / router.execution.routed)."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from router.agentic import AgenticRouter, ClientRegistry, LLMClient, LLMResponse
from router.agentic.decompose import SubTask
from router.constraints import OptimizationObjective
from router.execution.budget import BudgetLedger, ExecutionLimits
from router.execution.forecast import RouterPlanForecaster
from router.execution.orchestrator import OrchestratorConfig
from router.execution.plan import Plan, PlanStep
from router.execution.routed import PlanOrchestratorBridge, RoutedOrchestrator, plan_orchestrator
from router.execution.task import AgentTask, TaskStatus
from router.agentic.quality import QualityVerdict
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
    assert not any("Analyse the time complexity" in str(p) for _, p in sent)    # nothing is sent
    step2 = orch.subcalls[1]
    assert step2.error == "dependency_failed" and step2.cost is None
    assert step2.answer == "[error: dependency_failed]" and step2.prompt.startswith("Analyse the time")
    assert step2.depth == 1 and step2.meta["step_id"] == "root.1" and "tried_models" not in step2.meta
    assert not orch._tried.get("root.1")


def test_dependent_of_a_failed_step_reruns_once_with_its_input_and_is_not_escalated():
    sent, boom = [], {"n": 0}

    def make(m):
        def fn(prompt, **kw):
            sent.append((m, str(prompt)))
            if m == "big" and prompt == "Implement quicksort in Rust." and boom["n"] == 0:
                boom["n"] += 1
                raise RuntimeError("big is down once")
            return LLMResponse(f"{m} answers", cost=0.01, input_tokens=5, output_tokens=2)
        return LLMClient(CallableAdapter(m, fn))
    agent = _planning_agent(DEP_PLAN, clients=ClientRegistry({m: make(m) for m in POOL}))
    orch = _orchestrator(agent, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=1)
    orch._escalation_note["root.1"] = "escalated"                  # stale note from an earlier round
    assert orch.run(_task()).success
    step2 = [p for _, p in sent if "Analyse the time complexity" in p]
    assert len(step2) == 1 and "coder answers" in step2[0]
    assert orch.subcalls[0].meta["escalation"] == "escalated" and orch.subcalls[0].model_id == "coder"
    assert orch.subcalls[1].model_id == "big" and "escalation" not in orch.subcalls[1].meta


def test_a_rerun_dependent_does_not_report_an_earlier_rounds_escalation():
    checker = _ScriptedChecker(_fail(4, weak=[2]), _fail(4, weak=[1]), _pass())
    orch = _orchestrator(_planning_agent(THREE_STEP), checker=checker, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=2)
    assert orch.run(_task()).success
    assert orch.subcalls[0].meta["escalation"] == "escalated"
    assert "escalation" not in orch.subcalls[1].meta


class ContextRouter(TextRouter):
    """Like TextRouter, but 'coder' tops any text containing code ('fn ')."""

    def predict_scores_text(self, texts, *, encoder=None):
        self.text_calls.append(list(texts))
        rows = [[0.7, 0.9, 0.95] if "fn " in t else [0.7, 0.9, 0.8] for t in texts]
        return pd.DataFrame(rows, index=range(len(texts)), columns=POOL)


def _scripted_clients(answer, sent=None):
    """Every model replies ``answer(model_id, prompt)``."""
    def make(m):
        def fn(prompt, **kw):
            if sent is not None:
                sent.append((m, prompt))
            return LLMResponse(answer(m, str(prompt)), cost=0.01, input_tokens=5, output_tokens=2)
        return LLMClient(CallableAdapter(m, fn))
    return ClientRegistry({m: make(m) for m in POOL})


def _code_answers(m, p):
    return "fn quicksort() {}" if p.startswith("Implement") else f"{m} answers"


def _plan_settings(**kw):
    from router.agentic.schemas import AgenticRouterSettings, load_settings
    return AgenticRouterSettings.model_validate(
        {**load_settings().model_dump(), "orchestrator": "plan", "warmup": False, **kw})


def test_late_routing_routes_a_dependent_step_on_its_context():
    router = ContextRouter()
    agent = _planning_agent(DEP_PLAN, router=router, clients=_scripted_clients(_code_answers))
    orch = _orchestrator(agent, route_late=True)
    assert orch.run(_task()).success
    step1, step2 = orch.subcalls
    assert step1.meta["planned_model"] == step1.meta["routed_model"] == "big"
    assert step2.model_id == "coder"
    assert step2.meta["planned_model"] == "big" and step2.meta["routed_model"] == "coder"
    assert router.text_calls[-1][0].startswith("Analyse the time complexity")   # task first


def test_without_late_routing_steps_keep_their_plan_time_model():
    agent = _planning_agent(DEP_PLAN, router=ContextRouter(), clients=_scripted_clients(_code_answers))
    orch = _orchestrator(agent)
    assert orch.run(_task()).success
    assert orch.subcalls[1].model_id == "big"


def test_route_late_setting_reaches_the_orchestrator():
    agent = AgenticRouter.from_settings(_plan_settings(plan={"route_late": True}), router=TextRouter())
    assert agent.orchestrator.route_late is True


def test_escalation_never_retries_a_tried_model_and_takes_the_closest_step_up():
    orch = _orchestrator(AgenticRouter(TextRouter(), _clients(), decomposer=_decomposer(["a", "b"])))
    orch.task = _task()
    step = replace(orch.plan(orch.task).steps[0], model=orch._profile("small"))
    orch._scores[step.step_id] = {"small": 0.7, "big": 0.9, "coder": 0.8}
    assert orch._escalation_pick(step) == ("coder", "escalated")   # .8 is the closest step up
    orch._tried[step.step_id] = ["small", "coder"]
    assert orch._escalation_pick(step) == ("big", "escalated")
    orch._tried[step.step_id] = ["small", "coder", "big"]
    assert orch._escalation_pick(step) == ("small", "exhausted")


def test_escalation_prefers_the_cheapest_stronger_model():
    router = TextRouter(costs=[0.001, 0.001, 0.01])               # small, big, coder
    orch = _orchestrator(AgenticRouter(router, _clients(), decomposer=_decomposer(["a", "b"])))
    orch.task = _task()
    step = replace(orch.plan(orch.task).steps[0], model=orch._profile("small"))
    orch._scores[step.step_id] = {"small": 0.7, "big": 0.9, "coder": 0.8}
    assert orch._escalation_pick(step) == ("big", "escalated")


def test_step_failing_on_every_model_is_marked_exhausted():
    agent = AgenticRouter(TextRouter(), _clients(fail=set(POOL)), decomposer=_decomposer(["a", "b"]))
    orch = _orchestrator(agent, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=3)
    assert not orch.run(_task()).success
    assert orch.subcalls[0].meta["tried_models"] == ["big", "coder", "small", "small"]
    assert orch.subcalls[0].meta["escalation"] == "exhausted"


def _fail_dependent_on_big():
    """'big' fails only for steps that carry dependency context."""
    def make(m):
        def fn(prompt, **kw):
            if m == "big" and "Now:" in str(prompt):
                raise RuntimeError("big is down for dependent steps")
            return LLMResponse(f"{m} answers", cost=0.01, input_tokens=5, output_tokens=2)
        return LLMClient(CallableAdapter(m, fn))
    return ClientRegistry({m: make(m) for m in POOL})


def test_late_routing_does_not_undo_an_escalation():
    orch = _orchestrator(_planning_agent(DEP_PLAN, clients=_fail_dependent_on_big()),
                         route_late=True, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=1)
    assert orch.run(_task()).success
    assert orch.subcalls[1].model_id == "coder"
    assert orch.subcalls[1].meta["escalation"] == "escalated"


# --------------------------------------------------------------------------- #
# quality check rounds                                                        #
# --------------------------------------------------------------------------- #
class _ScriptedChecker:
    """Returns the next verdict per call (the last one repeats); records what it saw."""

    def __init__(self, *verdicts):
        self.verdicts, self.seen, self.on_call = list(verdicts), [], None

    def __call__(self, original, steps, answer):
        self.seen.append((original, list(steps), answer))
        return self.verdicts[min(len(self.seen), len(self.verdicts)) - 1]


def _fail(score, weak=(), follow_ups=()):
    return QualityVerdict(False, float(score), list(weak), "bad", list(follow_ups))


def _pass(score=9.0):
    return QualityVerdict(True, float(score))


THREE_STEP = ("1. Implement quicksort in Rust.\n"
              "2. [after 1] Analyse the Rust quicksort from step 1.\n"
              "3. Explain mergesort.")


def test_failed_check_escalates_the_weak_step_and_reruns_only_it_and_its_dependents():
    sent = []
    checker = _ScriptedChecker(_fail(4, weak=[1]), _pass())
    orch = _orchestrator(_planning_agent(THREE_STEP, clients=_clients(sent=sent)),
                         checker=checker, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=2)
    assert orch.run(_task()).success and len(orch.rounds) == 2
    prompts = [str(p) for _, p in sent]
    assert sum(p == "Implement quicksort in Rust." for p in prompts) == 2       # step 1 re-ran
    assert sum("Analyse the Rust quicksort" in p for p in prompts) == 2         # its dependent re-ran
    assert sum("Explain mergesort" in p for p in prompts) == 1                  # untouched
    assert orch.subcalls[0].model_id == "coder"
    assert orch.subcalls[0].meta["escalation"] == "escalated"


def test_failing_check_runs_max_replans_plus_one_rounds_and_returns_the_best():
    checker = _ScriptedChecker(_fail(6.1, weak=[3]), _fail(7.4, weak=[3]), _fail(6.8, weak=[3]))
    agent = _planning_agent(THREE_STEP, orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=2)))
    res = agent.run(COMPOUND)
    qc = res.quality_check
    assert qc["passed"] is False and qc["best_round"] == 1 and len(qc["rounds"]) == 3
    assert res.answer == checker.seen[1][2]


def test_a_passing_round_beats_a_higher_scoring_failing_round():
    checker = _ScriptedChecker(_fail(8, weak=[3]), _pass(7.5))
    agent = _planning_agent(THREE_STEP, orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=2)))
    res = agent.run(COMPOUND)
    assert res.quality_check["passed"] is True and res.quality_check["best_round"] == 1
    assert res.answer == checker.seen[1][2]


def test_unparsed_check_returns_the_answer_flagged_without_a_repair_round():
    checker = _ScriptedChecker(QualityVerdict(False, status="unparsed"))
    agent = _planning_agent(THREE_STEP, orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=2)))
    qc = agent.run(COMPOUND).quality_check
    assert len(qc["rounds"]) == 1 and qc["status"] == "unparsed" and qc["passed"] is False


def test_out_of_range_weak_steps_are_ignored():
    sent = []
    checker = _ScriptedChecker(_fail(4, weak=[0, 99]), _pass())
    agent = _planning_agent(THREE_STEP, clients=_clients(sent=sent), orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=2)))
    qc = agent.run(COMPOUND).quality_check
    assert qc["passed"] is False and len(qc["rounds"]) == 1     # nothing to repair: no second check
    assert len(sent) == 3                                       # nothing re-ran


def test_over_budget_repair_round_returns_the_best_answer_not_an_error():
    checker = _ScriptedChecker(_fail(5, weak=[1, 3]))
    agent = _planning_agent(
        THREE_STEP, router=TextRouter(costs=[0.001, 0.01, 0.005]),
        orchestrator=plan_orchestrator(budget=0.04, checker=checker,
                                       limits=ExecutionLimits(max_cost=0.04, max_replans=2)))
    res = agent.run(COMPOUND)                                   # round 1 spends 0.03; repair needs 0.02
    assert "over budget" not in res.answer and res.answer == checker.seen[0][2]
    assert res.quality_check["best_round"] == 0


def test_quality_check_setting_builds_a_routed_checker():
    from router.agentic.quality import QualityChecker
    agent = AgenticRouter.from_settings(
        _plan_settings(plan={"quality_check": {"enabled": True, "min_quality": 0.5}}), router=TextRouter())
    assert isinstance(agent.orchestrator.checker, QualityChecker)


def test_unknown_verify_prompt_is_rejected():
    with pytest.raises(ValueError, match="quality_check.prompt"):
        _plan_settings(plan={"quality_check": {"enabled": True, "prompt": "nope"}})


def test_best_round_answer_comes_with_that_rounds_steps_and_all_spend_is_counted():
    checker = _ScriptedChecker(_fail(5, weak=[1]))
    agent = _planning_agent(THREE_STEP, clients=_clients(fail={"coder"}), orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=1)))
    res = agent.run(COMPOUND)
    answer_steps = [s for s in res.steps if not s.meta.get("superseded")]
    assert res.answer == checker.seen[0][2]
    assert [s.model_id for s in answer_steps] == ["big", "big", "big"]
    assert not any(s.error for s in answer_steps)
    assert any(s.meta.get("superseded") and s.error for s in res.steps)   # the failed escalation is still reported
    assert res.cost == pytest.approx(sum(p["actual_cost"] for p in res.plans if p["actual_cost"] is not None))
    assert res.errors() == []


def test_checker_follow_up_becomes_a_step_with_its_dependency():
    sent = []
    follow = SubTask("Fix the compile error in the Rust quicksort from step 1.", (1,))
    checker = _ScriptedChecker(_fail(5, follow_ups=[follow]), _pass())
    orch = _orchestrator(_planning_agent(THREE_STEP, clients=_clients(sent=sent)),
                         checker=checker, max_workers=1)
    orch.limits = ExecutionLimits(max_replans=1)
    assert orch.run(_task()).success
    last = orch.task.plan.steps[-1]
    assert last.prompt == follow.text and last.dependencies == ["root.0"]
    fix = [str(p) for _, p in sent if "Fix the compile error" in str(p)]
    assert len(fix) == 1 and "Results of earlier steps" in fix[0]
    assert len(sent) == 4                                       # 3 original steps once + the follow-up
    assert len(checker.seen[1][1]) == 4                         # round 2 reviews the follow-up too


def test_follow_ups_respect_max_fanout():
    follows = [SubTask(f"Extra task number {i}.") for i in range(5)]
    checker = _ScriptedChecker(_fail(5, follow_ups=follows), _pass())
    orch = _orchestrator(_planning_agent(THREE_STEP), checker=checker)
    orch.limits = ExecutionLimits(max_replans=1, max_fanout=4)
    orch.run(_task())
    assert len(orch.task.plan.steps) == 4
    changes = orch._round_changes[orch.task.plan.plan_id]
    assert changes["follow_ups"] == ["Extra task number 0."]
    assert changes["dropped_follow_ups"] == [f"Extra task number {i}." for i in range(1, 5)]


class _UnscorableFollowUpRouter(TextRouter):
    """TextRouter, but any text containing 'UNSCORABLE' gets no usable score."""

    def predict_scores_text(self, texts, *, encoder=None):
        self.text_calls.append(list(texts))
        rows = [[float("nan")] * 3 if "UNSCORABLE" in t else [0.7, 0.9, 0.8] for t in texts]
        return pd.DataFrame(rows, index=range(len(texts)), columns=POOL)


def test_unscorable_follow_up_is_dropped_not_fatal():
    checker = _ScriptedChecker(_fail(5, follow_ups=[SubTask("UNSCORABLE follow-up task.")]), _pass())
    orch = _orchestrator(_planning_agent(THREE_STEP, router=_UnscorableFollowUpRouter()), checker=checker)
    orch.limits = ExecutionLimits(max_replans=1)
    assert not orch.run(_task()).success                        # nothing left to repair: no second check
    assert len(orch.task.plan.steps) == 3
    changes = orch._round_changes[orch.task.plan.plan_id]
    assert changes["follow_ups"] == [] and changes["dropped_follow_ups"] == ["UNSCORABLE follow-up task."]


# --------------------------------------------------------------------------- #
# final-review fixes                                                          #
# --------------------------------------------------------------------------- #
def _clients_failing(model, needle, sent=None, cost=0.01):
    """'model' fails on prompts containing 'needle'; everything else succeeds."""
    def make(m):
        def fn(prompt, **kw):
            if sent is not None:
                sent.append((m, prompt))
            if m == model and needle in str(prompt):
                raise RuntimeError(f"{m} is down")
            return LLMResponse(f"{m} answers", cost=cost, input_tokens=5, output_tokens=2)
        return LLMClient(CallableAdapter(m, fn))
    return ClientRegistry({m: make(m) for m in POOL})


def test_over_budget_repair_round_without_a_check_keeps_the_paid_work():
    agent = _planning_agent(
        THREE_STEP, clients=_clients_failing("big", "mergesort", cost=0.015),
        router=TextRouter(costs=[0.001, 0.01, 0.005]),
        orchestrator=plan_orchestrator(budget=0.031, limits=ExecutionLimits(max_cost=0.031, max_replans=1)))
    res = agent.run(COMPOUND)                         # round 0 spends 0.03; the escalation doesn't fit
    assert "over budget" not in res.answer
    assert len(res.steps) == 4
    refusal = res.steps[-1]
    assert refusal.error == "over_budget" and refusal.cost is None
    assert "budget remaining" in refusal.answer
    assert res.cost == pytest.approx(sum(p["actual_cost"] for p in res.plans if p["actual_cost"] is not None))
    assert res.cost == pytest.approx(0.03)


def test_over_budget_message_reports_the_remaining_budget():
    agent = AgenticRouter(TextRouter(costs=[0.001, 0.01, 0.005]), _clients(),
                          orchestrator=plan_orchestrator(budget=0.001))
    assert "budget remaining 0.001" in agent.run(COMPOUND).answer


def test_a_repair_round_with_nothing_to_repair_makes_no_calls():
    sent, synth = [], []
    checker = _ScriptedChecker(_fail(4))
    agent = _planning_agent(THREE_STEP, clients=_clients(sent=sent), orchestrator=plan_orchestrator(
        checker=checker, limits=ExecutionLimits(max_replans=2)))
    real = agent.synthesizer
    agent.synthesizer = lambda prompt, pairs: (synth.append(1), real(prompt, pairs))[1]
    res = agent.run(COMPOUND)
    assert len(sent) == 3 and len(checker.seen) == 1 and len(synth) == 1
    qc = res.quality_check
    assert len(qc["rounds"]) == 1 and qc["passed"] is False
    assert res.answer == checker.seen[0][2]
