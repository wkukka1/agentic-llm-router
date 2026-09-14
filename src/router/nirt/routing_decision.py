"""The routing policy itself -- pure argmax, no ground truth involved.

Split out of ``evaluation.routing.oracle`` (which owns everything
oracle/regret-related, all needing observed outcomes) so serving code --
:mod:`router.routing.base` -- can make a decision without importing
``evaluation``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def routing_decision(
    pred: np.ndarray,
    *,
    lam: float = 0.0,
    model_costs: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-query model index for ``argmax_m [pred - lam * C(m)]``.

    ``model_costs`` is a length-``M`` per-model cost vector ``C(m)`` (the spec's
    cost-aware router uses the model cost, not a per-cell cost). ``eligible`` is
    an optional ``[Q, M]`` bool mask -- ineligible cells are never selected.
    """
    util = np.asarray(pred, np.float64).copy()
    # a missing prediction (no observation for that cell) must never be picked
    util[~np.isfinite(util)] = -np.inf
    if lam and model_costs is not None:
        util = util - lam * np.asarray(model_costs, np.float64)[None, :]
    if eligible is not None:
        util = np.where(np.asarray(eligible, bool), util, -np.inf)
    rows_all_bad = ~np.isfinite(util).any(axis=1)
    if rows_all_bad.any():   # nothing eligible -> fall back to column 0, flagged by caller
        util[rows_all_bad, 0] = -1e18
    return util.argmax(axis=1)
