"""Post-hoc confidence calibration.

The router does not consume the argmax; it consumes the probability. Every
downstream decision in the design -- escalate to a stronger model, split into
sub-queries, ask a clarifying follow-up -- is a threshold on confidence, and a
threshold on a miscalibrated score is meaningless.

Temperature scaling is the right tool here: one parameter, fit on validation,
monotonic (so accuracy is unchanged) and it fixes the systematic
over-confidence that both cross-entropy fine-tuning and high-C logistic
regression produce.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

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


def fit_temperature(
    proba: np.ndarray,
    y_true_idx: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Find the temperature minimising validation NLL.

    ``T > 1`` softens an over-confident model; ``T < 1`` sharpens an
    under-confident one.
    """
    if len(y_true_idx) == 0:
        return 1.0

    def nll(temperature: float) -> float:
        scaled = apply_temperature(proba, temperature)
        picked = scaled[np.arange(len(y_true_idx)), y_true_idx]
        return float(-np.log(np.clip(picked, _EPSILON, None)).mean())

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x) if result.success else 1.0
