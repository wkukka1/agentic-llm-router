"""Anthropic provider adapter -- structural stub, no live API call.

``evaluation.judge.client.JudgeClient`` already calls the Anthropic API for
real, but only for offline pairwise-judge labeling
(``docs/anchor_judge.md``) -- an evaluation concern, not a serving one, and
not reusable here (``evaluation`` can't be imported from ``router``). This
adapter is the serving-side equivalent; it is not implemented.
"""

from __future__ import annotations

from typing import Optional

from ..client import LLMResponse
from .base import ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    def __init__(self, model: str, *, api_key: Optional[str] = None, **client_kwargs):
        self.model = model
        self.api_key = api_key
        self.client_kwargs = client_kwargs

    def complete(self, prompt: str) -> LLMResponse:
        raise NotImplementedError(
            "AnthropicAdapter is a structural stub -- wire up the `anthropic` SDK here."
        )

    def stream(self, prompt: str) -> LLMResponse:
        raise NotImplementedError("AnthropicAdapter.stream is a structural stub.")
