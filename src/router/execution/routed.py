"""The concrete plan-based orchestrator, and the bridge that lets
:class:`~router.agentic.router.AgenticRouter` use it.

``RoutedOrchestrator`` (an :class:`~router.execution.orchestrator.Orchestrator`)
solves one compound request in four phases, under the inherited
:meth:`~router.execution.orchestrator.Orchestrator.run` template:

1. **plan** -- decompose the request, route *every* sub-prompt in one batched
   :meth:`~router.routing.base.RouterModel.route_text` call (one encoder pass,
   not one per sub-prompt), and record per step the chosen model, its
   predicted quality, the ranked fallback models and token estimates;
2. **forecast / admit** -- a :class:`~router.execution.forecast.PlanForecaster`
   prices the plan before anything runs, and ``run()`` reserves that estimate
   on the shared :class:`~router.execution.budget.BudgetLedger` (a plan that
   doesn't fit is refused with ``error_code="over_budget"``, nothing spent);
3. **execute** -- steps run level by level through the dependency DAG;
   steps in one level run concurrently on a thread pool, so the plan takes
   about as long as its slowest chain, not the sum of its steps;
4. **replan** -- a failed step moves to its next-best model; steps that
   already succeeded are reused, not re-run.

Every model call goes through the agent (``agent._call_model``), so the run's
``max_calls`` budget, conversation history and failure handling are exactly
:class:`AgenticRouter`'s; worker threads get a copy of the caller's context so
they see that run's state. Steps are leaves: a plan is fully known (and
priceable) up front, so steps are not re-triaged or recursively decomposed --
deeper recursion is what :meth:`Orchestrator.spawn` is for.

``PlanOrchestratorBridge`` adapts the two interfaces: ``AgenticRouter`` calls
``orchestrator.run(prompt, agent, depth) -> (answer, [SubCall])``, while an
``Orchestrator`` runs an ``AgentTask`` against a ledger. The bridge builds the
task, ledger and orchestrator per request and maps the result back::

    from router.execution.routed import plan_orchestrator

    agent = AgenticRouter(router, clients, orchestrator=plan_orchestrator(budget=0.05))
"""

from __future__ import annotations

import contextvars
import math
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from ..agentic.conversation import history_chars
from ..agentic.result import SubCall, total_cost
from ..llm.cost import estimate_tokens
from ..llm.profile import LLMProfile
from .budget import BudgetLedger, ExecutionLimits
from .forecast import PlanForecaster, RouterPlanForecaster
from .orchestrator import Orchestrator, OrchestratorConfig
from .plan import Plan, PlanStep
from .results import ExecutionResult
from .task import AgentTask, TaskStatus

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from ..agentic.router import AgenticRouter
    from ..constraints import OptimizationObjective
    from ..llm.registry import LLMRegistry

__all__ = ["RoutedOrchestrator", "PlanOrchestratorBridge", "plan_orchestrator"]

class RoutedOrchestrator(Orchestrator):
    """Plan -> forecast -> reserve -> execute (parallel) -> replan, for one
    task, with every model call made through ``agent``. See the module
    docstring. ``subcalls`` holds the latest attempt of every step after
    :meth:`run`, in plan order."""

    def __init__(self, config: OrchestratorConfig, *, agent: "AgenticRouter",
                 plan_forecaster: Optional[PlanForecaster] = None,
                 objective: Optional["OptimizationObjective"] = None,
                 registry: Optional["LLMRegistry"] = None,
                 max_workers: int = 4, sequential: bool = False,
                 route_late: bool = False, router_tool=None):
        super().__init__(config, router_tool=router_tool, plan_forecaster=plan_forecaster)
        from ..constraints import OptimizationObjective

        self.agent = agent
        self.objective = objective or OptimizationObjective()
        self.registry = registry
        self.max_workers = max(1, int(max_workers))
        self.sequential = bool(sequential)
        self.route_late = bool(route_late)
        self.subcalls: list[SubCall] = []
        #: every plan this orchestrator made (the first, then each replan)
        self.plan_history: list[Plan] = []
        #: ``(plan_id, ExecutionResult)`` per executed attempt
        self.attempts: list[tuple[str, ExecutionResult]] = []
        self._calls: dict[str, SubCall] = {}            # step_id -> latest attempt
        self._scores: dict[str, dict[str, float]] = {}  # step_id -> model_id -> quality
        self._planned: dict[str, str] = {}             # step_id -> plan-time model
        self._pinned: set[str] = set()                 # escalated this round: keep the model
        cost_model = getattr(plan_forecaster, "cost_model", None)
        if cost_model is None:
            from ..llm.cost import CostModel
            cost_model = CostModel()
        self._cost_model = cost_model

    # -- plan --------------------------------------------------------------
    def plan(self, task: AgentTask) -> Plan:
        agent = self.agent
        subs = agent.decompose_plan(task.prompt, task.depth, sequential=self.sequential)
        texts = [s.text for s in subs]
        routes = [self._route_known(texts[0])] if len(texts) == 1 else self._route_batch(texts)
        context_chars = history_chars(agent._run_state().history)
        steps = []
        for i, (sub, (model_id, scores)) in enumerate(zip(subs, routes)):
            step_id = f"{task.task_id}.{i}"
            self._scores[step_id] = scores
            self._planned[step_id] = model_id
            steps.append(self._new_step(step_id, sub.text, model_id, scores,
                                        [f"{task.task_id}.{d - 1}" for d in sub.depends_on],
                                        context_chars))
        plan = Plan(plan_id=f"{task.task_id}/plan0", task_id=task.task_id, steps=steps)
        self._score(plan)
        task.plan = plan
        self.plan_history.append(plan)
        return plan

    def _new_step(self, step_id: str, prompt: str, model_id: str, scores: dict[str, float],
                  dependencies: list[str], context_chars: int) -> PlanStep:
        profile = self._profile(model_id)
        in_tok = estimate_tokens(len(prompt) + context_chars)
        return PlanStep(
            step_id=step_id, prompt=prompt, model=profile,
            estimated_input_tokens=in_tok,
            estimated_output_tokens=self._cost_model.predict_output_tokens(profile, in_tok),
            dependencies=list(dependencies),
            predicted_quality=scores.get(model_id, float("nan")),
            alternatives=self._alternatives(scores, exclude=model_id),
        )

    def _route_known(self, prompt: str) -> tuple[str, dict[str, float]]:
        """A request that doesn't decompose: reuse the route triage already
        computed for it (as ``RecursiveOrchestrator`` does), else route it."""
        known = self.agent._run_state().decisions.get(prompt)
        if known is not None:
            return known.selected_model_id, dict(known.scores)
        return self._route_batch([prompt])[0]

    def _route_batch(self, prompts: list[str]) -> list[tuple[str, dict[str, float]]]:
        """``(selected model, {model: quality})`` per prompt from ONE router
        call. Like ``AgenticRouter.route_decision``, a prompt the router has no
        usable score for is an error, not a silent default-model dispatch."""
        agent = self.agent
        if agent._resolver is not None or not getattr(agent.router, "can_route_text", False):
            # no batched text path: one prompt at a time through the agent (its
            # query_resolver, or the documented TypeError for a text-less router)
            rrs = [agent.route_decision(p) for p in prompts]
            return [(rr.selected_model_ids[0], rr.row_scores()) for rr in rrs]
        # same alpha-override resolution as AgenticRouter.route_decision: a
        # run(prompt, alpha=...) override must hold for every plan step too,
        # not just the root (AGENTICROUTER-alpha-plan).
        alpha = agent._run_state().alpha
        lam = 0.0 if alpha is not None else agent.lam
        rr = agent.router.route_text(prompts, encoder=agent.encoder, lam=lam, alpha=alpha)
        fallback = getattr(rr, "fallback", None)
        out = []
        for i, p in enumerate(prompts):
            if fallback is not None and bool(np.asarray(fallback)[i]):
                raise LookupError(f"{agent.router.name}: no usable score for sub-task {p[:80]!r}")
            out.append((rr.selected_model_ids[i], rr.row_scores(i)))
        return out

    @staticmethod
    def _alternatives(scores: dict[str, float], *, exclude: str) -> list[str]:
        ranked = sorted(((q, m) for m, q in scores.items() if math.isfinite(q) and m != exclude),
                        reverse=True)
        return [m for _, m in ranked]

    def _profile(self, model_id: str) -> LLMProfile:
        """The step's model profile: the ``registry`` entry if there is one,
        else the pool id carrying the prices of the client that serves it (so
        a plan is priced like its calls are -- e.g. at the Claude tier's
        rates under ``ClientRegistry.anthropic``), else a bare, unpriced one."""
        if self.registry is not None:
            try:
                return self.registry.get(model_id)
            except KeyError:
                pass
        try:
            served = getattr(self.agent.clients.get(model_id), "profile", None)
        except KeyError:
            served = None
        return LLMProfile(
            name=model_id, provider=getattr(served, "provider", None) or "unknown", model_id=model_id,
            input_cost_per_token=getattr(served, "input_cost_per_token", None),
            output_cost_per_token=getattr(served, "output_cost_per_token", None),
        )

    def _score(self, plan: Plan) -> None:
        """Forecast only the steps still to run (a replan keeps finished
        steps, already paid for), and write the estimates onto ``plan``."""
        if self.plan_forecaster is None:
            return
        pending = [s for s in plan.steps if not self._succeeded(s.step_id)]
        view = Plan(plan_id=plan.plan_id, task_id=plan.task_id, steps=pending)
        self.plan_forecaster.score_plan(view, self.objective, ledger=self.ledger, limits=self.limits)
        plan.estimated_cost, plan.estimated_quality = view.estimated_cost, view.estimated_quality
        plan.estimated_latency, plan.utility, plan.status = view.estimated_latency, view.utility, view.status

    def _succeeded(self, step_id: str) -> bool:
        sc = self._calls.get(step_id)
        return sc is not None and sc.error is None

    # -- execute -----------------------------------------------------------
    def execute(self, plan: Plan) -> ExecutionResult:
        t0 = time.perf_counter()
        fresh: list[SubCall] = []
        for level in _levels(plan):
            todo = [s for s in level if not self._succeeded(s.step_id)]
            todo = self._assign_models(plan, todo)
            if len(todo) == 1:
                results = [self._run_step(todo[0])]
            elif todo:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(todo))) as pool:
                    futures = [pool.submit(contextvars.copy_context().run, self._run_step, s)
                               for s in todo]
                    results = [f.result() for f in futures]
            else:
                results = []
            for step, sc in zip(todo, results):
                self._calls[step.step_id] = sc
                fresh.append(sc)
        self.subcalls = [self._calls[s.step_id] for s in plan.steps if s.step_id in self._calls]
        failed = [sc for sc in self.subcalls if sc.error is not None]
        output = ""
        if not failed:
            if len(self.subcalls) == 1:
                output = self.subcalls[0].answer
            else:
                output = self.agent.synthesizer(self.task.prompt,
                                                [(sc.prompt, sc.answer) for sc in self.subcalls])
        spent = total_cost(fresh)
        if spent is None:
            # a successful call with neither an actual cost nor an estimate:
            # the ledger needs a number, so commit 0 -- but say so, never silently
            warnings.warn(
                f"plan {plan.plan_id!r}: a step's cost is unknown (no prices, no usage, no "
                "router cost data); committing 0 for it to the budget ledger",
                stacklevel=2,
            )
            spent = float(sum(sc.cost for sc in fresh if sc.cost is not None))
        result = ExecutionResult(
            output=output,
            actual_cost=spent,
            actual_latency=time.perf_counter() - t0,
            input_tokens=int(sum(sc.meta.get("input_tokens") or 0 for sc in fresh)),
            output_tokens=int(sum(sc.meta.get("output_tokens") or 0 for sc in fresh)),
            error_code="step_failed" if failed else None,
        )
        self.attempts.append((plan.plan_id, result))
        return result

    def _assign_models(self, plan: Plan, steps: list[PlanStep]) -> list[PlanStep]:
        """Late routing (``route_late``): re-route, in one batched router call,
        every step whose routing text now carries dependency results, and write
        the pick into ``plan``. A step pinned by escalation keeps its model; a
        step with no context yet keeps its plan-time route (same text -- no
        second encoder pass). A step the router can't score keeps its model."""
        if not self.route_late:
            return steps
        due = [s for s in steps if s.step_id not in self._pinned
               and self._contextual_prompt(s)[1] != s.prompt]
        if not due:
            return steps
        try:
            routes = self._route_batch([self._contextual_prompt(s)[1] for s in due])
        except LookupError:
            return steps
        changed = {}
        for s, (model_id, scores) in zip(due, routes):
            self._scores[s.step_id] = scores
            changed[s.step_id] = replace(s, model=self._profile(model_id),
                                         predicted_quality=scores.get(model_id, float("nan")))
        plan.steps = [changed.get(s.step_id, s) for s in plan.steps]
        return [changed.get(s.step_id, s) for s in steps]

    def _contextual_prompt(self, step: PlanStep) -> tuple[str, str]:
        """``(call prompt, routing text)`` for ``step``. Only dependencies that
        *succeeded* contribute -- a failed one's ``[error: ...]`` placeholder
        is not a result. The routing text puts the task first, so encoder
        truncation drops context, not the task."""
        earlier = [self._calls[d] for d in step.dependencies if self._succeeded(d)]
        if not earlier:
            return step.prompt, step.prompt
        context = "\n\n".join(f"[{sc.prompt}]\n{sc.answer}" for sc in earlier)
        return (f"Results of earlier steps:\n{context}\n\nNow: {step.prompt}",
                f"{step.prompt}\n\nContext from earlier steps:\n{context}")

    def _run_step(self, step: PlanStep) -> SubCall:
        prompt, _ = self._contextual_prompt(step)
        sc = self.agent._call_model(prompt, step.model.model_id, self.task.depth + 1,
                                    predicted_quality=step.predicted_quality)
        sc.prompt = step.prompt
        sc.meta = {**sc.meta, "step_id": step.step_id,
                   "planned_model": self._planned.get(step.step_id, step.model.model_id),
                   "routed_model": step.model.model_id}
        return sc

    # -- replan ------------------------------------------------------------
    def replan(self, plan: Plan) -> Plan:
        steps = []
        for s in plan.steps:
            if self._succeeded(s.step_id) or not s.alternatives:
                steps.append(s)                  # keep: done, or nothing left to try (retry as is)
                continue
            nxt, rest = s.alternatives[0], s.alternatives[1:]
            profile = self._profile(nxt)
            steps.append(replace(
                s, model=profile, alternatives=list(rest),
                predicted_quality=self._scores.get(s.step_id, {}).get(nxt, float("nan")),
                estimated_output_tokens=self._cost_model.predict_output_tokens(
                    profile, s.estimated_input_tokens),
            ))
        new = Plan(plan_id=f"{plan.task_id}/plan{plan.revision + 1}", task_id=plan.task_id,
                   steps=steps, revision=plan.revision + 1)
        self._score(new)
        self.task.plan = new
        self.plan_history.append(new)
        return new

    def _child_kwargs(self) -> dict:
        return {**super()._child_kwargs(), "agent": self.agent, "objective": self.objective,
                "registry": self.registry, "max_workers": self.max_workers,
                "sequential": self.sequential, "route_late": self.route_late}


def _levels(plan: Plan) -> list[list[PlanStep]]:
    """Steps grouped so every step's dependencies are in an earlier group
    (``plan.validate()`` has already rejected cycles / unknown ids)."""
    remaining = {s.step_id: s for s in plan.steps}
    done: set[str] = set()
    levels = []
    while remaining:
        ready = [s for s in remaining.values() if set(s.dependencies) <= done]
        levels.append(ready)
        for s in ready:
            done.add(s.step_id)
            del remaining[s.step_id]
    return levels


class PlanOrchestratorBridge:
    """Lets ``AgenticRouter(orchestrator=...)`` drive a
    :class:`RoutedOrchestrator`: implements the agentic orchestrator call
    ``run(prompt, agent, depth) -> (answer, [SubCall])`` by building, per
    request, an :class:`AgentTask`, a fresh :class:`BudgetLedger` of
    ``budget`` (USD, the units the forecaster prices in) and the orchestrator,
    then mapping the :class:`ExecutionResult` back.

    A plan refused as over budget comes back as one failed ``SubCall``; a plan
    that still has failed steps after its replans comes back with those
    failures recorded and the rest of the answers synthesized -- the same
    "keep what was paid for" rule ``AgenticRouter`` applies to sub-calls."""

    name = "plan"

    def __init__(self, *, budget: float = math.inf, limits: Optional[ExecutionLimits] = None,
                 forecaster: Optional[Callable[["AgenticRouter"], PlanForecaster]] = None,
                 objective: Optional["OptimizationObjective"] = None,
                 registry: Optional["LLMRegistry"] = None,
                 max_workers: int = 4, sequential: bool = False,
                 route_late: bool = False):
        self.budget = float(budget)
        self.limits = limits
        self.forecaster = forecaster or (lambda agent: RouterPlanForecaster(agent.router))
        self.objective = objective
        self.registry = registry
        self.max_workers = max_workers
        self.sequential = sequential
        self.route_late = route_late

    def run(self, prompt: str, agent: "AgenticRouter", depth: int) -> tuple[str, list[SubCall]]:
        root_id = f"plan-{uuid.uuid4().hex[:12]}"
        limits = self.limits or ExecutionLimits(max_cost=self.budget, max_depth=agent.max_depth)
        orch = RoutedOrchestrator(
            OrchestratorConfig(name=self.name, version="0", llm=None,
                               ledger=BudgetLedger(root_id, self.budget), limits=limits),
            agent=agent, plan_forecaster=self.forecaster(agent), objective=self.objective,
            registry=self.registry, max_workers=self.max_workers, sequential=self.sequential,
            route_late=self.route_late,
        )
        task = AgentTask(task_id=root_id, parent_task_id=None, root_task_id=root_id,
                         prompt=prompt, depth=depth)
        result = orch.run(task)
        agent._run_state().plans.extend(_plan_records(orch, self.budget))
        if task.status is TaskStatus.TERMINATED:
            est = task.plan.budget_estimate() if task.plan is not None else float("nan")
            msg = f"plan over budget: needs ~{est:.6g}, budget {self.budget:.6g}"
            return f"[error: {msg}]", [SubCall(
                prompt=prompt, model_id="(none)", predicted_quality=float("nan"),
                answer=f"[error: {msg}]", depth=depth + 1, error=result.error_code or "over_budget",
            )]
        steps = list(orch.subcalls)
        if result.success:
            return result.output, steps
        answer = agent.synthesizer(prompt, [(sc.prompt, sc.answer) for sc in steps])
        return answer, steps


def _plan_records(orch: RoutedOrchestrator, budget: float) -> list[dict]:
    """One summary per plan revision: what the forecaster expected the
    remaining steps to cost / score / take, and what executing it actually
    spent (``None`` for a plan that never ran, e.g. refused over budget)."""
    spent = dict(orch.attempts)
    rows = []
    for plan in orch.plan_history:
        res = spent.get(plan.plan_id)
        if res is None:
            outcome = "over_budget" if orch.task.status is TaskStatus.TERMINATED else "not_run"
        else:
            outcome = "completed" if res.success else (res.error_code or "failed")
        rows.append({
            "plan_id": plan.plan_id, "task_id": plan.task_id, "revision": plan.revision,
            "n_steps": len(plan.steps), "budget": budget,
            "estimated_cost": plan.estimated_cost, "estimated_quality": plan.estimated_quality,
            "estimated_latency": plan.estimated_latency, "utility": plan.utility,
            "actual_cost": None if res is None else res.actual_cost,
            "actual_latency": None if res is None else res.actual_latency,
            "outcome": outcome,
        })
    return rows


def plan_orchestrator(**kwargs) -> PlanOrchestratorBridge:
    """The plan orchestrator, ready to pass as ``AgenticRouter(orchestrator=...)``.
    Keyword arguments are :class:`PlanOrchestratorBridge`'s (``budget``,
    ``limits``, ``forecaster``, ``objective``, ``registry``, ``max_workers``,
    ``sequential``, ``route_late``)."""
    return PlanOrchestratorBridge(**kwargs)
