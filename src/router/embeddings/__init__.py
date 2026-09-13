"""Frozen encoders with an on-disk cache."""

from router.embeddings.encoder import EmbeddingEncoder, resolve_device

__all__ = ["EmbeddingEncoder", "resolve_device"]
