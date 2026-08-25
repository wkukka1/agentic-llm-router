"""Score the routing *policy*, not the classifier.

A domain head with high macro-F1 is not the goal; a router that spends less
without dropping quality is. This module routes a labelled split in plan mode
and reports the economics: where traffic lands, what it costs, and how that
compares against the two trivial baselines (send everything to the cheapest
model, send everything to the best one).

Quality is deliberately *not* estimated here. Doing so would require executing
every prompt against every pool member; that is LLMRouterBench's job, and
inventing a quality number from the pool's prior `quality` field would be
circular. What this gives you is the cost side of the frontier plus the
distribution of decisions, which is what threshold tuning needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from router.agent.contracts import Action, Session, Tier
from router.agent.pipeline import AgenticRouter, _approx_tokens

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RoutingReport:
    per_prompt: pd.DataFrame
    summary: dict[str, Any]

    def render(self) -> str:
        lines = ["# Routing policy evaluation", ""]
        s = self.summary
        lines += [
            f"- prompts routed: **{s['n']}**",
            f"- mean estimated cost/prompt: **${s['mean_cost']:.5f}**",
            f"- vs. always-cheapest (${s['baseline_cheapest']:.5f}): "
            f"**{s['vs_cheapest_pct']:+.1f}%**",
            f"- vs. always-best (${s['baseline_best']:.5f}): "
            f"**{s['vs_best_pct']:+.1f}%** ({s['savings_vs_best_pct']:.1f}% saved)",
            "",
            "## Where traffic lands",
            "",
            pd.Series(s["action_share"]).to_frame("share").round(4).to_markdown(),
            "",
            pd.Series(s["tier_share"]).to_frame("share").round(4).to_markdown(),
            "",
            "## Model mix",
            "",
            pd.Series(s["model_share"]).to_frame("share").round(4).to_markdown(),
        ]
        if s.get("escalation_by_domain"):
            lines += [
                "",
                "## Escalation rate by predicted domain",
                "",
                pd.Series(s["escalation_by_domain"]).to_frame("escalated").round(4).to_markdown(),
            ]
        return "\n".join(lines) + "\n"


def evaluate_routing(
    router: AgenticRouter,
    prompts: list[str],
    *,
    expected_output_tokens: int = 800,
) -> RoutingReport:
    """Route every prompt in plan mode and summarise the decisions."""
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        # Each prompt is its own session: this measures single-turn routing,
        # not conversational carry-over.
        result = router.route(prompt, Session())
        decision = result.decision
        rows.append({
            "prompt": prompt[:200],
            "action": decision.action.value,
            "tier": decision.tier.value if decision.tier else None,
            "model": decision.model or decision.orchestrator,
            "domain": decision.signals.domain if decision.signals else None,
            "domain_confidence": decision.signals.domain_confidence if decision.signals else None,
            "difficulty": decision.signals.difficulty if decision.signals else None,
            "task_type": decision.signals.task_type if decision.signals else None,
            "n_sub_queries": len(result.sub_queries),
            "cost": decision.estimated_cost,
        })

    frame = pd.DataFrame(rows)
    pool = router.pool
    cheapest = min(pool.models, key=lambda m: m.cost_in + m.cost_out)
    best = max(pool.models, key=lambda m: m.quality)

    token_counts = [_approx_tokens(p) for p in prompts]
    baseline_cheapest = sum(
        cheapest.estimated_cost(n, expected_output_tokens) for n in token_counts
    ) / max(len(prompts), 1)
    baseline_best = sum(
        best.estimated_cost(n, expected_output_tokens) for n in token_counts
    ) / max(len(prompts), 1)
    mean_cost = float(frame["cost"].mean())

    escalated = frame[frame["action"] != Action.CLARIFY.value].copy()
    escalated["is_escalated"] = escalated["tier"].ne(Tier.WEAK.value)

    summary: dict[str, Any] = {
        "n": len(frame),
        "mean_cost": mean_cost,
        "baseline_cheapest": baseline_cheapest,
        "baseline_best": baseline_best,
        "vs_cheapest_pct": 100 * (mean_cost - baseline_cheapest) / max(baseline_cheapest, 1e-12),
        "vs_best_pct": 100 * (mean_cost - baseline_best) / max(baseline_best, 1e-12),
        "savings_vs_best_pct": 100 * (baseline_best - mean_cost) / max(baseline_best, 1e-12),
        "action_share": frame["action"].value_counts(normalize=True).to_dict(),
        # Clarified turns have no tier or model; label them rather than
        # rendering a bare NaN row.
        "tier_share": frame["tier"].fillna("(clarified)").value_counts(normalize=True).to_dict(),
        "model_share": frame["model"].fillna("(clarified)").value_counts(normalize=True).to_dict(),
        "escalation_by_domain": (
            escalated.groupby("domain")["is_escalated"].mean().sort_values(ascending=False).to_dict()
            if not escalated.empty else {}
        ),
    }
    return RoutingReport(per_prompt=frame, summary=summary)
