"""Turning prompts into the matrix a head is fitted on.

Shared by every head trained on the arena corpus, because which feature set a
signal needs is the measurement, not a property of the signal. Length needs an
encoder (0.387 free, 0.522 with one); the instruction-constraint attribute does
not (0.831 free, 0.825 with one). Same question, opposite answers, so both
options live in one place and each head records which it chose.
"""

from __future__ import annotations

import numpy as np

from router.signals import extract_many

#: Feature sets, cheapest first. Each is a claim about where the signal is.
FEATURE_SETS = ("length_only", "surface", "embedding", "surface+embedding")


def build_features(prompts: list[str], kind: str, embeddings: np.ndarray | None = None) -> np.ndarray:
    """Assemble the feature matrix for one of :data:`FEATURE_SETS`."""
    if kind not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {kind!r}; expected one of {FEATURE_SETS}")
    if kind in ("embedding", "surface+embedding") and embeddings is None:
        raise ValueError(f"{kind!r} needs precomputed embeddings")

    if kind == "length_only":
        # The cheapest thing that is not a constant, and the bar every richer
        # feature set has to clear to justify its cost.
        return np.log1p([len(p) for p in prompts]).reshape(-1, 1)
    surface, _ = extract_many(prompts)
    if kind == "surface":
        return surface
    if kind == "embedding":
        return embeddings
    return np.hstack([surface, embeddings])
