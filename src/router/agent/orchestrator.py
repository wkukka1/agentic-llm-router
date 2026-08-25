"""The strong path: decompose, route each part, execute in dependency order.

This is what makes the system agentic rather than a one-shot classifier. The
orchestrator does not answer the prompt; it produces a *plan* of sub-queries and
routes each one back through the same signal + policy stack, so a hard prompt
decomposes into several cheap, well-targeted calls instead of one expensive
generic one.

Sub-queries are forced onto the single-model path. Recursion is what turns a
decomposition bug into an unbounded bill, and there is no evidence a second
level of planning helps.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from router.agent.backends import _DECOMPOSE_SYSTEM, LLMBackend, _extract_json
from router.agent.contracts import Action, RouteDecision, Signals, SubQuery, Tier
from router.agent.policy import RoutingPolicy

log = logging.getLogger(__name__)

#: Signature of the function that actually answers one routed sub-query.
AnswerFn = Callable[[SubQuery, str], str]


@dataclass(slots=True)
class OrchestrationPlan:
    sub_queries: list[SubQuery]
    total_estimated_cost: float


class Orchestrator:
    """Decomposes a hard prompt and routes each sub-query independently."""

    def __init__(
        self,
        backend: LLMBackend,
        policy: RoutingPolicy,
        signal_fn: Callable[[str], Signals],
        *,
        max_sub_queries: int = 6,
    ) -> None:
        self.backend = backend
        self.policy = policy
        self.signal_fn = signal_fn
        self.max_sub_queries = max_sub_queries

    def decompose(self, prompt: str) -> list[SubQuery]:
        """Split the prompt into sub-queries with a dependency order."""
        # Plain substitution, not str.format: the template contains JSON braces.
        system = _DECOMPOSE_SYSTEM.replace("$MAX_SUB_QUERIES", str(self.max_sub_queries))
        try:
            raw = self.backend.complete(system, prompt, max_tokens=1024)
        except Exception as exc:  # noqa: BLE001 - degrade to a single unit of work
            log.warning("decomposition failed (%s); treating prompt as one sub-query", exc)
            return [SubQuery(index=0, text=prompt)]

        parsed = _extract_json(raw)
        if not isinstance(parsed, list) or not parsed:
            log.warning("decomposition returned no usable sub-queries; using the whole prompt")
            return [SubQuery(index=0, text=prompt)]

        sub_queries: list[SubQuery] = []
        for i, item in enumerate(parsed[: self.max_sub_queries]):
            text = (item.get("text") if isinstance(item, dict) else str(item)) or ""
            if not text.strip():
                continue
            raw_deps = item.get("depends_on") or [] if isinstance(item, dict) else []
            # Only backward dependencies are valid; anything else would cycle.
            deps = [int(d) for d in raw_deps if isinstance(d, int) and 0 <= int(d) < i]
            sub_queries.append(SubQuery(index=len(sub_queries), text=text.strip(), depends_on=deps))

        return sub_queries or [SubQuery(index=0, text=prompt)]

    def plan(self, prompt: str) -> OrchestrationPlan:
        """Decompose and assign a model plus skills to every sub-query."""
        sub_queries = self.decompose(prompt)
        total = 0.0
        for sub in sub_queries:
            sub.signals = self.signal_fn(sub.text)
            decision = self._route_sub_query(sub.signals)
            sub.assigned_model = decision.model
            sub.skills = decision.skills
            total += decision.estimated_cost
        return OrchestrationPlan(sub_queries=sub_queries, total_estimated_cost=total)

    def _route_sub_query(self, signals: Signals) -> RouteDecision:
        """Route a sub-query, never recursing into another orchestration."""
        decision = self.policy.decide("", signals)
        if decision.action is Action.ORCHESTRATE:
            decision = self.policy.single_model_decision(
                signals, Tier.STRONG, ["sub-query forced onto the single-model path"]
            )
        return decision

    def execute(self, plan: OrchestrationPlan, answer_fn: AnswerFn) -> list[SubQuery]:
        """Run sub-queries in dependency order, feeding answers forward.

        This is the interleave step: a sub-query that depends on earlier work
        receives those answers as context, so the decomposition can be
        sequential where it needs to be and parallel where it does not.
        """
        by_index = {sub.index: sub for sub in plan.sub_queries}
        for sub in self._topological_order(plan.sub_queries):
            context = "\n\n".join(
                f"[answer to sub-query {d}] {by_index[d].answer}"
                for d in sub.depends_on
                if by_index.get(d) is not None and by_index[d].answer
            )
            sub.answer = answer_fn(sub, context)
        return plan.sub_queries

    @staticmethod
    def _topological_order(sub_queries: list[SubQuery]) -> list[SubQuery]:
        """Dependency order, falling back to input order if anything cycles."""
        remaining = list(sub_queries)
        done: set[int] = set()
        ordered: list[SubQuery] = []
        while remaining:
            ready = [s for s in remaining if all(d in done for d in s.depends_on)]
            if not ready:
                log.warning("unresolvable sub-query dependencies; falling back to input order")
                ordered.extend(remaining)
                break
            for sub in ready:
                ordered.append(sub)
                done.add(sub.index)
                remaining.remove(sub)
        return ordered
