"""``LLMProfile``: static registry config for one model -- provider, per-token
cost, context window, capabilities. Learned ability estimates (a fitted
NIRT/IRT ``theta_m``) live inside the loaded model checkpoint, not here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMProfile:
    """``provider`` may be the model's *author* (as ``configs/model_registry.yaml``
    records it: ``meta``, ``mistralai`` ...); ``serving_provider`` is who is
    called (``openai``, ``together``, an OpenAI-compatible endpoint ...) and is
    what :class:`~router.llm.client.LLMClientFactory` dispatches on when set."""

    name: str
    provider: str
    model_id: str
    version: str = ""
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    context_window: int = 0
    capabilities: list[str] = field(default_factory=list)
    serving_provider: Optional[str] = None
