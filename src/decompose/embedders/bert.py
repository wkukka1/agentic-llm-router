"""``PromptEmbedder``: a raw BERT encoder (CLS or mean pooling), wrapping
:class:`router.embeddings.encoder.TextEncoder` (``backend="bert"``) -- the
same encoder the NIRT routers use to embed live prompt text and LLM profiles
into one shared text space.

This is the one place :mod:`decompose` is allowed to depend on ``router``
(see the ``decompose.embedders.bert -> router.embeddings`` exception in
``pyproject.toml``'s import-linter config). The alternative was either
duplicating ~40 lines of tokenize/pool/normalize logic here, or moving
``TextEncoder`` out of ``router.embeddings`` -- which live NIRT routers also
depend on and which is a serving-side module for good reason (it embeds
requests at inference time). Reusing the tested implementation won.
"""

from __future__ import annotations

from router.embeddings.encoder import EncoderConfig, TextEncoder

from .base import Embedder


class BertEmbedder(Embedder):
    name = "bert_embedder"
    version = "0"

    def __init__(self, *, model_name: str = "bert-base-uncased", pooling: str = "mean",
                 normalize: bool = True, device: str = "auto", seed: int = 42):
        self._encoder = TextEncoder(EncoderConfig(
            name=self.name, backend="bert", model_name=model_name,
            pooling=pooling, normalize=normalize, device=device, seed=seed,
        ))

    @property
    def dimension(self) -> int:
        return self._encoder.dim

    def embed(self, text: str) -> list[float]:
        return self._encoder.encode([text])[0].tolist()
