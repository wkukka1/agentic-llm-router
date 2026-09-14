"""``LLMProfile``: static registry config for one model -- provider, per-token
cost, context window, capabilities. Learned ability estimates (a fitted
NIRT/IRT ``theta_m``) live inside the loaded model checkpoint, not here."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMProfile:
    name: str
    provider: str
    model_id: str
    version: str = ""
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    context_window: int = 0
    capabilities: list[str] = field(default_factory=list)
