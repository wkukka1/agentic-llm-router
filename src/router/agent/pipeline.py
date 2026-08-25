"""The agentic router: one object, one entry point, the whole flow.

::

    prompt
      -> gate            underspecified? -> back to the user with questions
      -> signals         domain head + difficulty estimator + task type
      -> policy          weak / strong / orchestrate
           weak     -> one cheap model answers
           strong   -> one capable model answers
           orchestrate -> decompose -> route each sub-query -> interleave -> synthesise
      -> session         answer + follow-ups feed the next turn

Two execution modes:

* **plan** (default) -- produces the full routing plan and costs it, without
  generating any answer. Runs offline, and is what the evaluation harness scores.
* **execute** -- additionally calls the assigned models. Needs a real backend.

The mode is explicit because a plan is a genuinely useful artefact on its own,
and silently returning placeholder text in place of an answer would not be.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from router.agent.backends import HeuristicBackend, LLMBackend
from router.agent.contracts import (
    Action,
    RouteDecision,
    Session,
    Signals,
    SubQuery,
    Tier,
    TurnResult,
)
from router.agent.difficulty import DifficultyEstimator, HeuristicDifficulty
from router.agent.gate import PromptGate
from router.agent.orchestrator import Orchestrator
from router.agent.policy import PolicyThresholds, RoutingPolicy
from router.agent.pool import ModelPool
from router.agent.skills import SkillRegistry

log = logging.getLogger(__name__)

#: Supplies (domain, confidence, distribution) for a prompt. A trained
#: `router.inference.DomainHead` satisfies this; so does a fixed stub in tests.
DomainFn = Callable[[str], tuple[str, float, dict[str, float]]]
#: Supplies the capability axis for a prompt.
TaskTypeFn = Callable[[str], str]


def _approx_tokens(text: str) -> int:
    """Cheap token estimate. Good enough for cost bands, not for billing."""
    return max(1, len(text) // 4)


@dataclass(slots=True)
class AgentConfig:
    mode: str = "plan"  # "plan" | "execute"
    gate_enabled: bool = True
    max_sub_queries: int = 6
    #: Refuse to route a turn whose plan exceeds this many USD. None disables.
    max_cost_per_turn: float | None = None


class AgenticRouter:
    """The top-level object a caller uses."""

    def __init__(
        self,
        *,
        domain_fn: DomainFn,
        pool: ModelPool | None = None,
        skills: SkillRegistry | None = None,
        backend: LLMBackend | None = None,
        difficulty: DifficultyEstimator | None = None,
        task_type_fn: TaskTypeFn | None = None,
        thresholds: PolicyThresholds | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.backend = backend or HeuristicBackend()
        if self.config.mode not in ("plan", "execute"):
            raise ValueError(f"unknown mode {self.config.mode!r}; expected 'plan' or 'execute'")
        if self.config.mode == "execute" and not getattr(self.backend, "can_generate", False):
            raise ValueError(
                f"{type(self.backend).__name__} cannot generate answers, so mode='execute' "
                "would return rule-based filler as if it were a response. Pass a generating "
                "backend (e.g. LiteLLMBackend via --backend-model), or use mode='plan'."
            )
        self.pool = pool or ModelPool.from_yaml()
        self.skills = skills or SkillRegistry.from_yaml()
        self.domain_fn = domain_fn
        self.task_type_fn = task_type_fn or (lambda _: "other")
        self.difficulty = difficulty or HeuristicDifficulty()
        self.policy = RoutingPolicy(self.pool, self.skills, thresholds)
        self.gate = PromptGate(self.backend, enabled=self.config.gate_enabled)
        self.orchestrator = Orchestrator(
            self.backend, self.policy, self.signals_for,
            max_sub_queries=self.config.max_sub_queries,
        )

    # -- stage 2 ---------------------------------------------------------

    def signals_for(self, prompt: str) -> Signals:
        """Run every upstream head over a prompt."""
        domain, confidence, distribution = self.domain_fn(prompt)
        task_type = self.task_type_fn(prompt)
        n_tokens = _approx_tokens(prompt)
        difficulty = self.difficulty.estimate(
            prompt, domain=domain, task_type=task_type, n_tokens=n_tokens
        )
        return Signals(
            domain=domain,
            domain_confidence=confidence,
            difficulty=difficulty,
            task_type=task_type,
            n_tokens=n_tokens,
            domain_distribution=distribution,
            is_ambiguous=confidence < self.policy.thresholds.min_domain_confidence,
        )

    # -- the turn --------------------------------------------------------

    def route(self, prompt: str, session: Session | None = None) -> TurnResult:
        """Take one prompt through the full flow."""
        session = session or Session()
        has_history = bool(session.history)

        gate = self.gate.check(prompt, has_history=has_history)
        if not gate.is_sufficient:
            result = TurnResult(
                decision=RouteDecision(
                    action=Action.CLARIFY,
                    gate=gate,
                    rationale=[f"gate: {gate.reason}"],
                ),
                clarifying_questions=gate.missing_requirements,
            )
            session.record(prompt, result)
            return result

        # Follow-ups are routed with their history in scope -- the feedback
        # arrow in the design. But only referential ones: prepending history to
        # a self-contained prompt would hand the classifier text that is mostly
        # about the *previous* question.
        needs_context = session.needs_context(prompt)
        routable = session.contextualize(prompt) if needs_context else prompt
        signals = self.signals_for(routable)
        decision = self.policy.decide(routable, signals)
        decision.gate = gate

        if decision.action is Action.ORCHESTRATE:
            result = self._run_orchestrated(prompt, decision)
        else:
            result = self._run_direct(prompt, decision)

        self._enforce_budget(result)
        session.record(prompt, result)
        return result

    def _run_direct(self, prompt: str, decision: RouteDecision) -> TurnResult:
        answer = None
        if self.config.mode == "execute":
            answer = self._answer(prompt, decision.model, decision.skills, context="")
        return TurnResult(decision=decision, answer=answer,
                          usage={"backend": type(self.backend).__name__})

    def _run_orchestrated(self, prompt: str, decision: RouteDecision) -> TurnResult:
        plan = self.orchestrator.plan(prompt)
        decision.estimated_cost += plan.total_estimated_cost
        decision.rationale.append(
            f"decomposed into {len(plan.sub_queries)} sub-queries "
            f"(est. ${plan.total_estimated_cost:.4f} to execute)"
        )

        answer = None
        if self.config.mode == "execute":
            self.orchestrator.execute(
                plan,
                lambda sub, ctx: self._answer(sub.text, sub.assigned_model, sub.skills, context=ctx),
            )
            answer = self._synthesize(prompt, plan.sub_queries, decision.orchestrator)

        return TurnResult(
            decision=decision,
            answer=answer,
            sub_queries=plan.sub_queries,
            usage={"backend": type(self.backend).__name__, "n_sub_queries": len(plan.sub_queries)},
        )

    # -- execution -------------------------------------------------------

    def _answer(self, prompt: str, model: str | None, skills: list[str], *, context: str) -> str:
        system = f"You are {model}. Answer precisely."
        if skills:
            system += f" Tools available: {', '.join(skills)}."
        user = f"{context}\n\n{prompt}".strip() if context else prompt
        return self.backend.complete(system, user, max_tokens=1500)

    def _synthesize(self, prompt: str, sub_queries: list[SubQuery], orchestrator: str | None) -> str:
        parts = "\n\n".join(f"### {s.text}\n{s.answer or ''}" for s in sub_queries)
        return self.backend.complete(
            f"You are {orchestrator}. Synthesise the sub-answers into one coherent response.",
            f"Original request:\n{prompt}\n\nSub-answers:\n{parts}",
            max_tokens=2000,
        )

    def _enforce_budget(self, result: TurnResult) -> None:
        cap = self.config.max_cost_per_turn
        if cap is None or result.decision.estimated_cost <= cap:
            return
        # Downgrade rather than fail: a cheaper answer beats no answer.
        log.warning(
            "plan costs $%.4f > cap $%.4f; downgrading to the weak path",
            result.decision.estimated_cost, cap,
        )
        signals = result.decision.signals
        if signals is None:
            return
        downgraded = self.policy.single_model_decision(
            signals, Tier.WEAK,
            list(result.decision.rationale) + [f"over budget cap ${cap:.4f} -> downgraded"],
        )
        downgraded.gate = result.decision.gate
        result.decision = downgraded
        result.sub_queries = []
