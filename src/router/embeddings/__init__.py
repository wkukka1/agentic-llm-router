"""Query / profile embedding: configurable text-encoder pathways + on-disk store.

Phase 0 does NOT fine-tune the encoder. Inference is deterministic. Large builds
stream to disk and are resumable (:func:`build_store`).
"""

from .encoder import (
    EmbeddingStore,
    EncoderConfig,
    IncompleteEmbeddingStore,
    TextEncoder,
    available_pathways,
    build_store,
    default_store_dir,
    load_encoder,
)

__all__ = [
    "TextEncoder",
    "EncoderConfig",
    "EmbeddingStore",
    "IncompleteEmbeddingStore",
    "build_store",
    "load_encoder",
    "available_pathways",
    "default_store_dir",
]
