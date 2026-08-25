"""The model pool the router selects from.

A pool member is described by what it *costs*, what it is *good at*, and what
*tier* it occupies. Selection is then a constrained argmax rather than a
hard-coded if-chain, which is what lets the pool change without the policy
changing.

Costs are USD per million tokens and are configuration, not constants -- they
move, and the pool YAML is the single place to update them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from router.agent.contracts import Signals, Tier

#: Shipped default pool description.
DEFAULT_POOL_PATH = Path("configs/pool.yaml")


@dataclass(slots=True)
class ModelSpec:
    """One member of the routing pool."""

    name: str
    tier: Tier
    #: USD per 1M input / output tokens.
    cost_in: float
    cost_out: float
    context_window: int = 128_000
    #: Domains this model is unusually good at; contributes a selection bonus.
    strengths: list[str] = field(default_factory=list)
    #: Task types this model must not be given (e.g. no code model for translation).
    excludes: list[str] = field(default_factory=list)
    #: Skills (tools) the model is permitted to invoke.
    skills: list[str] = field(default_factory=list)
    #: Measured quality on a 0-1 scale; the tie-breaker when cost is equal.
    quality: float = 0.5
    #: Rough seconds-per-1k-output-tokens, for latency-aware routing.
    latency_s_per_1k: float = 5.0

    def estimated_cost(self, n_in: int, n_out: int) -> float:
        return (n_in * self.cost_in + n_out * self.cost_out) / 1_000_000


@dataclass(slots=True)
class ModelPool:
    """A set of :class:`ModelSpec` with tier- and signal-aware selection."""

    models: list[ModelSpec]

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("model pool is empty")
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate model names in pool: {names}")

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_POOL_PATH) -> ModelPool:
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dicts(raw["models"])

    @classmethod
    def from_dicts(cls, rows: list[dict]) -> ModelPool:
        return cls([ModelSpec(**{**row, "tier": Tier(row["tier"])}) for row in rows])

    def by_name(self, name: str) -> ModelSpec:
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(f"no model {name!r} in pool; have {[m.name for m in self.models]}")

    def candidates(self, tier: Tier, signals: Signals | None = None) -> list[ModelSpec]:
        """Pool members eligible for a tier, after applying hard exclusions.

        ORCHESTRATOR members are eligible for STRONG work too -- an orchestrator
        is a strong model that additionally knows how to delegate.
        """
        eligible = Tier.ORCHESTRATOR if tier is Tier.STRONG else None
        pool = [m for m in self.models if m.tier is tier or (eligible and m.tier is eligible)]
        if signals is not None:
            pool = [m for m in pool if signals.task_type not in m.excludes]
        return pool

    def score(self, model: ModelSpec, signals: Signals, *, cost_weight: float, n_out: int = 800) -> float:
        """Utility of a model for a prompt: quality, minus cost, plus fit.

        ``cost_weight`` is the single knob that moves the pool from
        cost-optimising to quality-optimising, and is what the router exposes to
        a caller with a budget.
        """
        cost = model.estimated_cost(max(signals.n_tokens, 1), n_out)
        # Normalised against a $10/1M reference so the weight is interpretable.
        cost_penalty = cost_weight * (cost / (10 * (signals.n_tokens + n_out) / 1_000_000))
        fit = 0.15 if signals.domain in model.strengths else 0.0
        return model.quality + fit - cost_penalty

    def select(
        self,
        tier: Tier,
        signals: Signals,
        *,
        cost_weight: float = 0.5,
        n_out: int = 800,
    ) -> ModelSpec:
        """Pick the highest-utility eligible model."""
        pool = self.candidates(tier, signals)
        if not pool:
            raise ValueError(
                f"no candidate in tier {tier.value} for task_type {signals.task_type!r}"
            )
        return max(pool, key=lambda m: self.score(m, signals, cost_weight=cost_weight, n_out=n_out))
