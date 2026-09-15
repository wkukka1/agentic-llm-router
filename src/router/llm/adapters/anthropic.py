"""Anthropic provider adapter -- structural stub, no live API call.

``evaluation.judge.client.JudgeClient`` already calls the Anthropic API for
real, but only for offline pairwise-judge labeling
(``docs/anchor_judge.md``) -- an evaluation concern, not a serving one, and
not reusable here (``evaluation`` can't be imported from ``router``). This
adapter is the serving-side equivalent; it is not implemented.

The Messages API requires ``max_tokens``: take it from ``generation_kwargs``,
else ``client_kwargs["max_tokens"]`` (both reach the adapter through
:meth:`~router.llm.client.LLMClientFactory.client_for` / ``LLMClient.complete``).
"""

from __future__ import annotations

from typing import Iterator

from ..client import LLMResponse
from .base import Prompt, ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        raise NotImplementedError(
            "AnthropicAdapter is a structural stub -- wire up the `anthropic` SDK here."
        )

    def stream(self, prompt: Prompt, **generation_kwargs) -> Iterator[str]:
        raise NotImplementedError("AnthropicAdapter.stream is a structural stub.")
