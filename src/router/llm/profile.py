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
    what :class:`~router.llm.client.LLMClientFactory` dispatches on when set.

    ``input_cost_per_token``/``output_cost_per_token`` default to ``None``
    (*unset/unknown*), not ``0.0`` -- a genuinely free model should set its
    price to ``0.0`` explicitly; leaving it unset must not be silently priced
    at zero (see :meth:`~router.llm.client.LLMClient.price`)."""

    name: str
    provider: str
    model_id: str
    version: str = ""
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    context_window: int = 0
    capabilities: list[str] = field(default_factory=list)
    serving_provider: Optional[str] = None
