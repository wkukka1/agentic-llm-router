"""Tiny numeric helpers shared across subpackages."""

from __future__ import annotations

import numpy as np


def unit_rows(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """``x`` scaled to unit L2 norm along ``axis`` (norm floored at ``eps``)."""
    x = np.asarray(x)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)
