"""Stage 3: weak, strong, or orchestrate.

This is the decision the whole upstream stack exists to inform. It is kept
deliberately simple and fully inspectable -- thresholds on two calibrated
numbers plus explicit overrides -- because a routing policy that cannot be read
off the page cannot be tuned against a cost/quality target.

Escalation happens for three distinct reasons, and they are separate on purpose:

* **hard**      -- difficulty is above threshold. Route strong.
* **ambiguous** -- the domain head is not confident, so the cheap model's
                   suitability is unknown. Escalating is the safe play.
* **decomposable** -- the prompt contains several asks. Even at moderate
                   difficulty this wants an orchestrator, because one call has
                   to interleave several unrelated pieces of work.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from router.agent.contracts import Action, RouteDecision, Signals, Tier
from router.agent.pool import ModelPool
from router.agent.skills import SkillRegistry

log = logging.getLogger(__name__)

_MULTI_ASK = re.compile(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+|\band then\b|\balso\b", re.IGNORECASE)


@dataclass(slots=True)
class PolicyThresholds:
    """Every knob the weak/strong decision turns.

    Defaults are a starting point, not a result: they should be re-fit against
    LLMRouterBench once the pool has measured quality scores.
    """

    #: At or above this difficulty, take the strong path.
    strong_difficulty: float = 0.55
    #: At or above this difficulty, prefer an orchestrator over a single strong call.
    orchestrate_difficulty: float = 0.75
    #: Below this domain confidence, treat the prompt as ambiguous.
    min_domain_confidence: float = 0.45
    #: Ambiguity only escalates above this difficulty. Without this floor, an
    #: obviously trivial prompt ("what year did X happen?") pays for a strong
    #: model just because the topic head could not choose between two domains --
    #: but domain barely matters when nothing about the prompt is hard.
    ambiguity_difficulty_floor: float = 0.30
    #: Multi-part prompts orchestrate once difficulty clears this lower bar.
    decompose_difficulty: float = 0.45
    #: 0 = ignore cost, 1 = cost dominates model selection within a tier.
    cost_weight: float = 0.5
    #: Expected output tokens, used for cost estimation only.
    expected_output_tokens: int = 800


class RoutingPolicy:
    """Maps signals onto a :class:`RouteDecision`."""

    def __init__(
        self,
        pool: ModelPool,
        skills: SkillRegistry,
        thresholds: PolicyThresholds | None = None,
    ) -> None:
        self.pool = pool
        self.skills = skills
        self.thresholds = thresholds or PolicyThresholds()

    def is_decomposable(self, prompt: str) -> bool:
        return bool(_MULTI_ASK.search(prompt))

    def decide(self, prompt: str, signals: Signals) -> RouteDecision:
        t = self.thresholds
        rationale: list[str] = [
            f"domain={signals.domain} (p={signals.domain_confidence:.2f}), "
            f"task={signals.task_type}, difficulty={signals.difficulty:.2f}"
        ]

        ambiguous = signals.domain_confidence < t.min_domain_confidence
        ambiguity_escalates = ambiguous and signals.difficulty >= t.ambiguity_difficulty_floor
        if ambiguous:
            runner_up = signals.runner_up_domain
            note = f"; runner-up {runner_up[0]} at {runner_up[1]:.2f}" if runner_up else ""
            if ambiguity_escalates:
                rationale.append(
                    f"domain confidence {signals.domain_confidence:.2f} < {t.min_domain_confidence} "
                    f"-> ambiguous, escalating{note}"
                )
            else:
                rationale.append(
                    f"domain ambiguous ({signals.domain_confidence:.2f}{note}) but difficulty "
                    f"{signals.difficulty:.2f} < {t.ambiguity_difficulty_floor} -> not escalating"
                )

        decomposable = self.is_decomposable(prompt)
        if decomposable:
            rationale.append("prompt contains multiple asks -> decomposable")

        # Orchestrate when the work is genuinely multi-step: either very hard,
        # or several asks that individually clear the moderate bar.
        wants_orchestration = signals.difficulty >= t.orchestrate_difficulty or (
            decomposable and signals.difficulty >= t.decompose_difficulty
        )
        wants_strong = signals.difficulty >= t.strong_difficulty or ambiguity_escalates

        if wants_orchestration:
            rationale.append(
                f"difficulty {signals.difficulty:.2f} >= {t.orchestrate_difficulty} "
                if signals.difficulty >= t.orchestrate_difficulty
                else f"decomposable and difficulty >= {t.decompose_difficulty} "
            )
            return self._orchestrate_decision(signals, rationale)

        if wants_strong:
            return self.single_model_decision(signals, Tier.STRONG, rationale)

        rationale.append(
            f"difficulty {signals.difficulty:.2f} < {t.strong_difficulty}, no escalation trigger "
            f"-> weak path"
        )
        return self.single_model_decision(signals, Tier.WEAK, rationale)

    def single_model_decision(
        self, signals: Signals, tier: Tier, rationale: list[str]
    ) -> RouteDecision:
        """Select one model in ``tier`` and cost the call.

        Public because the orchestrator and the budget guard both need to force
        a turn onto the single-model path without re-running the whole policy.
        """
        t = self.thresholds
        model = self.pool.select(
            tier, signals, cost_weight=t.cost_weight, n_out=t.expected_output_tokens
        )
        chosen_skills = self.skills.select(signals, available=model.skills)
        cost = model.estimated_cost(max(signals.n_tokens, 1), t.expected_output_tokens)
        cost += sum(skill.cost for skill in chosen_skills)
        rationale.append(f"selected {model.name} from tier {tier.value} (est. ${cost:.4f})")
        return RouteDecision(
            action=Action.DIRECT,
            tier=tier,
            model=model.name,
            skills=[s.name for s in chosen_skills],
            signals=signals,
            estimated_cost=cost,
            rationale=rationale,
        )

    def _orchestrate_decision(self, signals: Signals, rationale: list[str]) -> RouteDecision:
        t = self.thresholds
        candidates = self.pool.candidates(Tier.ORCHESTRATOR, signals)
        if not candidates:
            rationale.append("no orchestrator in pool -> falling back to strong single model")
            return self.single_model_decision(signals, Tier.STRONG, rationale)

        orchestrator = max(candidates, key=lambda m: m.quality)
        rationale.append(f"orchestrator {orchestrator.name} will decompose and delegate")
        # Sub-query costs are added by the orchestrator as it routes each one;
        # this is the planning call only.
        planning_cost = orchestrator.estimated_cost(max(signals.n_tokens, 1), t.expected_output_tokens // 2)
        return RouteDecision(
            action=Action.ORCHESTRATE,
            tier=Tier.ORCHESTRATOR,
            orchestrator=orchestrator.name,
            signals=signals,
            estimated_cost=planning_cost,
            rationale=rationale,
        )


def fit_thresholds_to_quantiles(
    difficulties: list[float],
    *,
    strong_quantile: float = 0.70,
    orchestrate_quantile: float = 0.92,
    decompose_quantile: float = 0.55,
    base: PolicyThresholds | None = None,
) -> PolicyThresholds:
    """Set difficulty thresholds from the observed distribution, not absolutes.

    A difficulty estimator's raw scale is arbitrary -- the heuristic one is
    heavily compressed on benchmark-style prompts (99th percentile ~0.50), so a
    hand-set 0.55 threshold silently routes ~100% of traffic to the weak tier
    and leaves the strong tier dead. Expressing the thresholds as quantiles of
    real traffic makes the policy invariant to that scale, so swapping in the
    trained regressor later does not require re-tuning by hand.

    The quantiles *are* the routing budget: ``strong_quantile=0.70`` means "send
    the hardest 30% to a strong model".
    """
    import numpy as np

    thresholds = base or PolicyThresholds()
    values = np.asarray([d for d in difficulties if d is not None], dtype=float)
    if values.size == 0:
        log.warning("no difficulty samples; keeping default thresholds")
        return thresholds

    return PolicyThresholds(
        strong_difficulty=float(np.quantile(values, strong_quantile)),
        orchestrate_difficulty=float(np.quantile(values, orchestrate_quantile)),
        decompose_difficulty=float(np.quantile(values, decompose_quantile)),
        min_domain_confidence=thresholds.min_domain_confidence,
        ambiguity_difficulty_floor=float(np.quantile(values, decompose_quantile)),
        cost_weight=thresholds.cost_weight,
        expected_output_tokens=thresholds.expected_output_tokens,
    )
