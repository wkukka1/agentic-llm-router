"""Google (Gemini) provider adapter -- structural stub, no live API call."""

from __future__ import annotations

from typing import Optional

from ..client import LLMResponse
from .base import ProviderAdapter


class GoogleAdapter(ProviderAdapter):
    def __init__(self, model: str, *, api_key: Optional[str] = None, **client_kwargs):
        self.model = model
        self.api_key = api_key
        self.client_kwargs = client_kwargs

    def complete(self, prompt: str) -> LLMResponse:
        raise NotImplementedError(
            "GoogleAdapter is a structural stub -- wire up the `google-generativeai` "
            "SDK here."
        )

    def stream(self, prompt: str) -> LLMResponse:
        raise NotImplementedError("GoogleAdapter.stream is a structural stub.")
