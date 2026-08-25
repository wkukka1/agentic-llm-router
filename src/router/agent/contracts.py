"""Types the agentic routing flow moves through.

The flow, in order:

1. **Gate** the incoming prompt. If it is underspecified, stop and go back to
   the user for the missing requirements rather than routing a bad prompt.
2. **Signal** it: domain, difficulty, task type.
3. **Decide** weak or strong.
   * *weak*  -> one model answers directly.
   * *strong* -> an orchestrator decomposes into sub-queries, each of which is
     routed again (with skills attached) and the results are synthesised.
4. **Follow up**: the answer plus any follow-up feeds back in as a new turn,
   carrying the session's accumulated context.

Everything here is data. The behaviour lives in :mod:`router.agent.gate`,
:mod:`router.agent.policy` and :mod:`router.agent.orchestrator`, so the control
flow can be tested without a network call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: A prompt that opens with a bare referent or a discourse marker is continuing
#: the previous turn rather than starting a new one.
_REFERENTIAL = re.compile(
    r"^\s*(it|that|this|they|them|those|these|now|then|instead|same|"
    r"and|but|also|what about|how about|why not|do it|try)\b",
    re.IGNORECASE,
)


class Action(StrEnum):
    """What the router decided to do with a turn."""

    #: Prompt is underspecified; hand the missing requirements back to the user.
    CLARIFY = "clarify"
    #: Weak path: a single, cheap model answers directly.
    DIRECT = "direct"
    #: Strong path: an orchestrator decomposes and delegates.
    ORCHESTRATE = "orchestrate"


class Tier(StrEnum):
    """Capability tier of a pool member. Also the weak/strong axis of the router."""

    WEAK = "weak"
    STRONG = "strong"
    #: Can hold a multi-step plan and delegate. A superset of STRONG.
    ORCHESTRATOR = "orchestrator"


@dataclass(slots=True)
class Signals:
    """Everything the upstream heads know about a prompt.

    ``domain_confidence`` and ``difficulty`` are the two numbers the policy
    actually thresholds on, which is why both heads are required to be
    calibrated before they are allowed to feed this.
    """

    domain: str
    domain_confidence: float
    difficulty: float
    task_type: str = "other"
    n_tokens: int = 0
    domain_distribution: dict[str, float] = field(default_factory=dict)
    #: Populated when the heads disagree enough that the policy should hedge.
    is_ambiguous: bool = False

    @property
    def runner_up_domain(self) -> tuple[str, float] | None:
        ranked = sorted(self.domain_distribution.items(), key=lambda kv: -kv[1])
        return ranked[1] if len(ranked) > 1 else None


@dataclass(slots=True)
class GateResult:
    """Verdict of the prompt-quality gate."""

    is_sufficient: bool
    #: Requirements the prompt is missing, phrased as questions for the user.
    missing_requirements: list[str] = field(default_factory=list)
    reason: str = ""
    #: 0-1; low means the gate itself is unsure, so err towards routing.
    confidence: float = 1.0


@dataclass(slots=True)
class SubQuery:
    """One decomposed unit of work produced by the orchestrator."""

    index: int
    text: str
    #: Indices of sub-queries whose answers this one needs first.
    depends_on: list[int] = field(default_factory=list)
    signals: Signals | None = None
    assigned_model: str | None = None
    skills: list[str] = field(default_factory=list)
    answer: str | None = None


@dataclass(slots=True)
class RouteDecision:
    """The routing decision for one turn, with the evidence behind it.

    ``rationale`` exists because a router that cannot explain itself cannot be
    debugged: every field that moved the decision is recorded.
    """

    action: Action
    tier: Tier | None = None
    model: str | None = None
    orchestrator: str | None = None
    skills: list[str] = field(default_factory=list)
    signals: Signals | None = None
    gate: GateResult | None = None
    estimated_cost: float = 0.0
    rationale: list[str] = field(default_factory=list)

    def explain(self) -> str:
        head = f"{self.action.value}"
        if self.model:
            head += f" -> {self.model}"
        if self.orchestrator:
            head += f" via orchestrator {self.orchestrator}"
        return head + "".join(f"\n  - {r}" for r in self.rationale)


@dataclass(slots=True)
class TurnResult:
    """What one turn through the router produced."""

    decision: RouteDecision
    answer: str | None = None
    #: Populated on ORCHESTRATE, one entry per decomposed unit.
    sub_queries: list[SubQuery] = field(default_factory=list)
    #: Populated on CLARIFY: what to ask the user before doing any work.
    clarifying_questions: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_user_input(self) -> bool:
        return self.decision.action is Action.CLARIFY


@dataclass(slots=True)
class Session:
    """Multi-turn state, so a follow-up is routed with its history in scope.

    The feedback arrow in the design goes from the answer back into the
    classifier; carrying prior turns is what makes "and now do it in Rust" a
    routable prompt rather than a domain-less fragment.
    """

    turns: list[TurnResult] = field(default_factory=list)
    #: Prompts already answered, oldest first.
    history: list[str] = field(default_factory=list)
    total_cost: float = 0.0

    def record(self, prompt: str, result: TurnResult) -> None:
        self.history.append(prompt)
        self.turns.append(result)
        self.total_cost += result.decision.estimated_cost

    def needs_context(self, prompt: str, *, max_standalone_words: int = 5) -> bool:
        """Whether ``prompt`` cannot be understood without the previous turn.

        Only referential follow-ups get history prepended. Doing it
        unconditionally would be actively harmful: the concatenation becomes the
        classifier's input, so a self-contained prompt would be scored on text
        that is mostly about a different question.
        """
        if not self.history:
            return False
        text = (prompt or "").strip()
        # The referential opener is the reliable signal; the word count is only a
        # backstop for bare fragments ("the second one"). Keep it low -- a short
        # self-contained question must not inherit an unrelated previous turn.
        return bool(_REFERENTIAL.match(text)) or len(text.split()) <= max_standalone_words

    def contextualize(self, prompt: str, max_history: int = 3) -> str:
        """Prepend recent history so follow-ups carry their referents."""
        if not self.history:
            return prompt
        recent = self.history[-max_history:]
        return "\n".join(f"[previous] {h}" for h in recent) + f"\n[current] {prompt}"
