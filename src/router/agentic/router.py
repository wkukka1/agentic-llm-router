"""``AgenticRouter`` -- a :class:`~router.routing.base.RouterModel` plus an orchestrator.

Flow for a live prompt:

1. **route** the prompt against the candidate pool (encoder-backed, so no
   prebuilt ``query_id`` is needed) -> best model + its predicted quality.
2. **triage** on that signal (:mod:`router.agentic.triage`): can the best model
   answer directly, or should the request be decomposed?
3. **dispatch**
   * *single*  -> send the prompt to the routed model's client, return the answer.
   * *compound* -> hand it to an orchestrator that decomposes the request and
     routes each sub-task (recursively), then synthesizes one answer. The router
     is available to the orchestrator as a tool.

    from router.routing import NIRTRouter
    from router.agentic import AgenticRouter, ClientRegistry

    ar = AgenticRouter(NIRTRouter.from_run("nirt-2d-projected"))   # echo clients
    res = ar.run("Explain quicksort, then write it in Rust and analyse its complexity.")
    res.mode            # "orchestrated"
    res.answer
    res.summary()

For real model calls pass ``clients=ClientRegistry.langchain(pool)`` and an
LLM-driven orchestrator / triage (see ``docs/agentic_router.md``).

Failure handling: a model call that raises inside an orchestration is recorded
as a failed :class:`SubCall` (``error`` set, see ``AgenticResult.errors()``) so
the sub-answers already paid for survive; a failing *root* call still raises.
``max_calls`` caps the model calls one :meth:`run` may make across the whole
recursion tree (``max_depth`` alone bounds depth, not fan-out) -- routed calls
*and* the helper LLMs (``LLMTriage``, ``LLMDecomposer``, ``LLMSynthesizer``,
and the ``LangChainToolOrchestrator`` tool-agent's own reasoning steps) all
count against it. The one gap: a tool-agent's reasoning steps are counted but
not directly enforced mid-reasoning (there's no ``SubCall`` to fail there);
the limit still bites the next routed call once exhausted.

``run()`` is safe to call concurrently on one shared ``AgenticRouter``
instance -- each call's routing cache and call count live in a context-local
state (thread- or asyncio-task-isolated), not on the instance.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..routing.base import RouterModel, RoutingResult
from .llm_clients import ClientRegistry, LLMClient
from .decompose import NaiveDecomposer, concat_synthesizer
from .orchestrator import RecursiveOrchestrator
from .result import AgenticResult, SubCall, total_cost
from .triage import HeuristicTriage, TriageDecision, best_from_result

__all__ = ["AgenticRouter"]

Decomposer = Callable[[str], list]
Synthesizer = Callable[..., str]
Triage = Callable[..., TriageDecision]


@dataclass
class _RunState:
    """One :meth:`AgenticRouter.run` call's state: routes already computed
    (so a sub-task triage isn't repeated) and the shared call budget.

    Lives in a :class:`contextvars.ContextVar`, not on the ``AgenticRouter``
    instance (AGENTICROUTER-004) -- each ``run()`` call (its own thread,
    asyncio task, or a re-entrant nested call) gets an isolated state, so
    concurrent runs can no longer wipe each other's cached triage decisions
    or share a call counter."""

    decisions: dict[str, TriageDecision] = field(default_factory=dict)
    calls: int = 0
    max_calls: Optional[int] = None

    def use_call(self, n: int = 1, *, enforce: bool = True) -> None:
        """Count ``n`` model call(s) against the budget. Every LLM
        invocation a run makes -- routed, triage, decompose, synthesize --
        goes through here, not just the routed calls (AGENTICROUTER-003).
        ``enforce=False`` counts without raising, for call sites that can't
        safely raise mid-call (see ``LangChainToolOrchestrator``)."""
        if enforce and self.max_calls is not None and self.calls + n > self.max_calls:
            raise RuntimeError(f"max_calls={self.max_calls} model calls exhausted for this run")
        self.calls += n


_current_run: "contextvars.ContextVar[Optional[_RunState]]" = contextvars.ContextVar(
    "agentic_router_current_run", default=None
)


class AgenticRouter:
    def __init__(
        self,
        router: RouterModel,
        clients: Optional[ClientRegistry | dict[str, LLMClient]] = None,
        *,
        triage: Optional[Triage] = None,
        orchestrator: object = None,
        decomposer: Optional[Decomposer] = None,
        synthesizer: Optional[Synthesizer] = None,
        encoder: object = None,
        lam: float = 0.0,
        max_depth: int = 2,
        max_calls: Optional[int] = None,
        query_resolver: Optional[Callable[[str], str]] = None,
    ):
        self.router = router
        if not isinstance(clients, ClientRegistry):
            clients = ClientRegistry(clients or None)
        self.clients = clients
        self.triage = triage or HeuristicTriage()
        self.orchestrator = orchestrator or RecursiveOrchestrator()
        self.decomposer = decomposer or NaiveDecomposer()
        self.synthesizer = synthesizer or concat_synthesizer
        self.encoder = encoder
        self.lam = float(lam)
        self.max_depth = int(max_depth)
        self.max_calls = None if max_calls is None else int(max_calls)
        self._resolver = query_resolver
        # AGENTICROUTER-003: route each helper's own LLM call through the same
        # per-run budget a routed call goes through (see _RunState.use_call).
        # Harmless no-op for helpers that don't expose ``on_call`` (the
        # heuristic/naive defaults, or a user-supplied plain callable).
        for helper in (self.triage, self.decomposer, self.synthesizer):
            if hasattr(helper, "on_call"):
                helper.on_call = self._use_call

    # ------------------------------------------------------------------ #
    # routing a single prompt                                            #
    # ------------------------------------------------------------------ #
    def route_decision(self, prompt: str) -> RoutingResult:
        """One-row :class:`RoutingResult` for ``prompt`` (encoder-backed, or via a
        ``query_resolver`` that maps text -> an existing ``query_id``).

        Raises ``LookupError`` when the router has nothing selectable for the
        prompt (e.g. the resolved ``query_id`` has no scores) -- otherwise the
        pool's first model would silently answer."""
        if self._resolver is not None:
            rr = self.router.route([self._resolver(prompt)], lam=self.lam)
        elif not getattr(self.router, "can_route_text", False):
            raise TypeError(
                f"{type(self.router).__name__} cannot score raw text; pass "
                "query_resolver= or use NIRTRouter / KNNRouter"
            )
        else:
            rr = self.router.route_text([prompt], encoder=self.encoder, lam=self.lam)
        if rr.any_fallback:
            raise LookupError(
                f"{self.router.name}: no usable score for this prompt "
                f"(query_id={rr.query_ids[0]!r}); refusing to dispatch to a default model"
            )
        return rr

    def triage_prompt(self, prompt: str) -> TriageDecision:
        rr = self.route_decision(prompt)
        model_id, q, scores = best_from_result(rr)
        decision = self.triage(prompt, model_id=model_id, predicted_quality=q, scores=scores)
        self._run_state().decisions[prompt] = decision
        return decision

    # ------------------------------------------------------------------ #
    # execution                                                          #
    # ------------------------------------------------------------------ #
    def run(self, prompt: str) -> AgenticResult:
        """Triage ``prompt`` and either answer it directly or orchestrate it."""
        token = _current_run.set(_RunState(max_calls=self.max_calls))
        try:
            decision = self.triage_prompt(prompt)
            root = self._solve(prompt, 0, decision=decision)
        finally:
            _current_run.reset(token)

        if root.mode == "single":
            reason = decision.reason
            if decision.orchestrate:
                reason += " -- nothing to decompose, answered directly"
            return AgenticResult(
                prompt=prompt, mode="single", answer=root.answer,
                triage_reason=reason,
                selected_model_id=root.model_id,
                # the model that actually answered's own score, not the pool
                # best decision.predicted_quality carries (AGENTICROUTER-001)
                predicted_quality=root.predicted_quality,
                steps=[root], cost=root.cost,
            )
        return AgenticResult(
            prompt=prompt, mode="orchestrated", answer=root.answer,
            triage_reason=decision.reason,
            selected_model_id=decision.selected_model_id,
            predicted_quality=decision.predicted_quality,
            steps=root.children, orchestrator=getattr(self.orchestrator, "name", "?"),
            cost=root.cost,
        )

    # -- recursion unit ------------------------------------------------
    def _solve(self, prompt: str, depth: int, *, decision: Optional[TriageDecision] = None) -> SubCall:
        """Solve one (sub-)prompt: a leaf ``SubCall`` (single) or a branch whose
        ``children`` are the orchestrator's sub-calls."""
        if decision is None:
            if depth >= self.max_depth:
                # the verdict would be ignored at the cap -> route only, skip triage
                return self._answer_directly(prompt, depth)
            decision = self.triage_prompt(prompt)

        if decision.mode == "single" or depth >= self.max_depth:
            return self._answer_directly(prompt, depth, model_id=decision.selected_model_id,
                                         predicted_quality=self._quality_of(decision))

        answer, children = self.orchestrator.run(prompt, self, depth)
        if (len(children) == 1 and children[0].prompt == prompt and not children[0].children
                and answer == children[0].answer):
            # nothing to decompose (and no synthesized answer of the orchestrator's
            # own) -> it collapses back to a single routed call at this depth
            children[0].depth = depth
            return children[0]
        return SubCall(
            prompt=prompt, model_id=decision.selected_model_id,
            predicted_quality=decision.predicted_quality, answer=answer,
            mode="orchestrated", depth=depth, children=children,
            # leaves only: a branch child's cost already includes its descendants
            cost=total_cost(n for c in children for n in c.leaves()),
        )

    @staticmethod
    def _quality_of(decision: TriageDecision) -> float:
        """Predicted quality of the model that will answer (not the pool best)."""
        return float(decision.scores.get(decision.selected_model_id, decision.predicted_quality))

    def _answer_directly(
        self, prompt: str, depth: int, *,
        model_id: Optional[str] = None, predicted_quality: float = float("nan"),
    ) -> SubCall:
        """Route ``prompt`` (if the model was not already chosen) and invoke it."""
        if model_id is None:
            known = self._run_state().decisions.get(prompt)
            if known is not None:          # triage already routed this prompt
                model_id, predicted_quality = known.selected_model_id, self._quality_of(known)
            else:
                rr = self.route_decision(prompt)
                model_id = rr.selected_model_ids[0]
                predicted_quality = float(rr.selected_scores[0])
        return self._call_model(prompt, model_id, depth, predicted_quality=predicted_quality)

    def _call_model(self, prompt: str, model_id: str, depth: int, *,
                    predicted_quality: float = float("nan")) -> SubCall:
        """Invoke ``model_id`` once, counting it against ``max_calls``. Inside an
        orchestration (``depth > 0``) a failure becomes a failed ``SubCall``; at
        the root it raises."""
        try:
            self._use_call()
            resp = self.clients.complete(model_id, prompt)
        except Exception as exc:
            if depth == 0:
                raise
            return SubCall(
                prompt=prompt, model_id=model_id, predicted_quality=float(predicted_quality),
                answer=f"[error: {type(exc).__name__}: {exc}]", mode="single", depth=depth,
                cost=None, error=repr(exc),
            )
        return SubCall(
            prompt=prompt, model_id=model_id, predicted_quality=float(predicted_quality),
            answer=resp.content, mode="single", depth=depth, cost=resp.cost,
        )

    # -- per-run state (AGENTICROUTER-004: context-local, not instance state) --
    def _run_state(self) -> _RunState:
        """The active run's state. Falls back to a fresh, context-local one
        for calls made outside :meth:`run` (e.g. driving ``_solve``/
        ``_call_model`` directly, as the LangChain tool tests do)."""
        rs = _current_run.get()
        if rs is None:
            rs = _RunState(max_calls=self.max_calls)
            _current_run.set(rs)
        return rs

    def _use_call(self, n: int = 1, *, enforce: bool = True) -> None:
        """Count ``n`` model call(s) against this run's shared budget. Every
        LLM invocation -- routed (``_call_model``) and helper (triage,
        decompose, synthesize, the LangChain tool-agent's own reasoning) --
        goes through here (AGENTICROUTER-003)."""
        self._run_state().use_call(n, enforce=enforce)

    # ------------------------------------------------------------------ #
    # convenience                                                        #
    # ------------------------------------------------------------------ #
    def with_langchain_orchestrator(self, llm, **kw) -> "AgenticRouter":
        """Return a copy that decomposes via a LangChain tool-calling agent."""
        from .orchestrator import LangChainToolOrchestrator

        clone = self._shallow_copy()
        clone.orchestrator = LangChainToolOrchestrator(llm, **kw)
        return clone

    def _shallow_copy(self) -> "AgenticRouter":
        c = AgenticRouter.__new__(AgenticRouter)
        c.__dict__.update(self.__dict__)
        return c

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"AgenticRouter(router={self.router.name!r}, "
                f"orchestrator={getattr(self.orchestrator, 'name', '?')!r}, "
                f"max_depth={self.max_depth})")
