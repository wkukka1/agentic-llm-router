"""OpenAI provider adapter -- structural stub, no live API call.

``router.agentic.llm_clients.LangChainClient`` already reaches OpenAI models
in practice, via LangChain's ``init_chat_model``. This adapter is the
direct-SDK shape from the production design (no LangChain indirection); it
is not implemented.
"""

from __future__ import annotations

from typing import Iterator

from ..client import LLMResponse
from .base import Prompt, ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        raise NotImplementedError(
            "OpenAIAdapter is a structural stub -- wire up the `openai` SDK here. "
            "For a working call today, use router.agentic.llm_clients.LangChainClient."
        )

    def stream(self, prompt: Prompt, **generation_kwargs) -> Iterator[str]:
        raise NotImplementedError("OpenAIAdapter.stream is a structural stub.")
