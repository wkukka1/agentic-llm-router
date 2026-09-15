"""``SentenceEmbedder``: a ``sentence-transformers`` model, wrapping
:class:`router.embeddings.encoder.TextEncoder` (``backend="sentence_transformer"``).

See :mod:`decompose.embedders.bert` for why wrapping ``TextEncoder`` --
rather than reimplementing it -- is a deliberate, narrow exception to
:mod:`decompose` otherwise being dependency-free.
"""

from __future__ import annotations

from router.embeddings.encoder import EncoderConfig, TextEncoder

from .base import Embedder
import numpy as np

class SentenceEmbedder(Embedder):
    name = "sentence_embedder"
    version = "0"

    def __init__(self, *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 normalize: bool = True, device: str = "auto", seed: int = 42):
        self._encoder = TextEncoder(EncoderConfig(
            name=self.name, backend="sentence_transformer", model_name=model_name,
            normalize=normalize, device=device, seed=seed,
        ))

    @property
    def dimension(self) -> int:
        return self._encoder.dim

    def embed(self, text: str) -> np.ndarray:
        return self._encoder.encode([text])[0]
    
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return self._encoder.encode(texts)
