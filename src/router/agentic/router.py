"""``AgenticRouter`` -- a :class:`~router.routing.base.Router` plus an orchestrator.

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
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..routing.base import Router, RoutingResult
from .llm_clients import ClientRegistry, LLMClient
from .decompose import NaiveDecomposer, concat_synthesizer
from .orchestrator import RecursiveOrchestrator
from .result import AgenticResult, SubCall
from .triage import HeuristicTriage, TriageDecision, best_from_result

__all__ = ["AgenticRouter"]

Decomposer = Callable[[str], list]
Synthesizer = Callable[..., str]
Triage = Callable[..., TriageDecision]


class AgenticRouter:
    def __init__(
        self,
        router: Router,
        clients: Optional[ClientRegistry | dict[str, LLMClient]] = None,
        *,
        triage: Optional[Triage] = None,
        orchestrator: object = None,
        decomposer: Optional[Decomposer] = None,
        synthesizer: Optional[Synthesizer] = None,
        encoder: object = None,
        lam: float = 0.0,
        max_depth: int = 2,
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
        self._resolver = query_resolver

    # ------------------------------------------------------------------ #
    # routing a single prompt                                            #
    # ------------------------------------------------------------------ #
    def route_decision(self, prompt: str) -> RoutingResult:
        """One-row :class:`RoutingResult` for ``prompt`` (encoder-backed, or via a
        ``query_resolver`` that maps text -> an existing ``query_id``)."""
        if self._resolver is not None:
            return self.router.route([self._resolver(prompt)], lam=self.lam)
        if not getattr(self.router, "can_route_text", False):
            raise TypeError(
                f"{type(self.router).__name__} cannot score raw text; pass "
                "query_resolver= or use NIRTRouter / KNNRouter"
            )
        return self.router.route_text([prompt], encoder=self.encoder, lam=self.lam)

    def triage_prompt(self, prompt: str) -> TriageDecision:
        rr = self.route_decision(prompt)
        model_id, q, scores = best_from_result(rr)
        return self.triage(prompt, model_id=model_id, predicted_quality=q, scores=scores)

    # ------------------------------------------------------------------ #
    # execution                                                          #
    # ------------------------------------------------------------------ #
    def run(self, prompt: str) -> AgenticResult:
        """Triage ``prompt`` and either answer it directly or orchestrate it."""
        decision = self.triage_prompt(prompt)
        root = self._solve(prompt, 0, decision=decision)

        if root.mode == "single":
            return AgenticResult(
                prompt=prompt, mode="single", answer=root.answer,
                triage_reason=decision.reason,
                selected_model_id=root.model_id,
                predicted_quality=decision.predicted_quality,
                steps=[root], cost=root.cost or 0.0,
            )
        total = sum((n.cost or 0.0) for s in root.children for n in s.walk())
        return AgenticResult(
            prompt=prompt, mode="orchestrated", answer=root.answer,
            triage_reason=decision.reason,
            selected_model_id=decision.selected_model_id,
            predicted_quality=decision.predicted_quality,
            steps=root.children, orchestrator=getattr(self.orchestrator, "name", "?"),
            cost=total,
        )

    # -- recursion unit ------------------------------------------------
    def _solve(self, prompt: str, depth: int, *, decision: Optional[TriageDecision] = None) -> SubCall:
        """Solve one (sub-)prompt: a leaf ``SubCall`` (single) or a branch whose
        ``children`` are the orchestrator's sub-calls."""
        if decision is None:
            decision = self.triage_prompt(prompt)

        if decision.mode == "single" or depth >= self.max_depth:
            return self._answer_directly(
                prompt, depth, model_id=decision.selected_model_id,
                predicted_quality=decision.predicted_quality,
            )

        answer, children = self.orchestrator.run(prompt, self, depth)
        if len(children) == 1 and children[0].prompt == prompt and not children[0].children:
            # nothing to decompose -> it collapses back to a single routed call
            return children[0]
        return SubCall(
            prompt=prompt, model_id=decision.selected_model_id,
            predicted_quality=decision.predicted_quality, answer=answer,
            mode="orchestrated", depth=depth, children=children,
            cost=sum((n.cost or 0.0) for c in children for n in c.walk()),
        )

    def _answer_directly(
        self, prompt: str, depth: int, *,
        model_id: Optional[str] = None, predicted_quality: float = float("nan"),
    ) -> SubCall:
        """Route ``prompt`` (if the model was not already chosen) and invoke it."""
        if model_id is None:
            rr = self.route_decision(prompt)
            model_id, predicted_quality, _ = best_from_result(rr)
        resp = self.clients.invoke(model_id, prompt)
        return SubCall(
            prompt=prompt, model_id=model_id, predicted_quality=float(predicted_quality),
            answer=resp.text, mode="single", depth=depth, cost=resp.cost,
        )

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
