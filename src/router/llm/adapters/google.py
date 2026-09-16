"""Google (Gemini) provider adapter -- structural stub, no live API call."""

from __future__ import annotations

from typing import Iterator

from ..client import LLMResponse
from .base import Prompt, ProviderAdapter


class GoogleAdapter(ProviderAdapter):
    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        raise NotImplementedError(
            "GoogleAdapter is a structural stub -- wire up the `google-generativeai` "
            "SDK here."
        )

    def stream(self, prompt: Prompt, **generation_kwargs) -> Iterator[str]:
        raise NotImplementedError("GoogleAdapter.stream is a structural stub.")
