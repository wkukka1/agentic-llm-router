"""Temperature scaling, at serving time.

Lives in `router` rather than beside the rest of the metrics because the serving
heads apply it on every call, and `router` may not import `training`. Fitting
the temperature is an offline job and stays in `training.prompt_decomposition.metrics`; applying it
is a serving job and is here. The split is the dependency rule made concrete:
one function crossed the line, so one function moved.
"""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-12


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Re-sharpen or soften a probability matrix by ``temperature``.

    Works on probabilities rather than logits so it applies uniformly to every
    model in the registry, including the sklearn ones that never expose logits.
    """
    logits = np.log(np.clip(proba, _EPSILON, None)) / max(temperature, _EPSILON)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
