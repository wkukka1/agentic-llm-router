"""The :class:`Embedder` interface.

Concrete embedders (:class:`~decompose.embedders.bert.PromptEmbedder`,
:class:`~decompose.embedders.sentence_transformer.SentenceEmbedder`) wrap the
project's real, working encoder --
:class:`router.embeddings.encoder.TextEncoder` -- rather than reimplementing
HuggingFace tokenize/pool/normalize a second time. See each module's
docstring for why that's a deliberate, narrow exception to this package
otherwise being dependency-free.
"""

from __future__ import annotations

import abc


class Embedder(abc.ABC):
    name: str = "embedder"
    version: str = "0"
    dimension: int = 0

    @abc.abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError
