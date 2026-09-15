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

    Exact ties go to the lowest column index (``np.argmax`` semantics), i.e. the
    first-listed model in ``pred``'s column order -- not cost-broken. This is the
    router's live decision rule; it is deliberately independent of
    ``evaluation.routing.oracle``'s tie-*inclusive* oracle labels, which grade the
    router against ground truth and must never feed back into it.
    """
    util = np.asarray(pred, np.float64).copy()
    # a missing prediction (no observation for that cell) must never be picked
    util[~np.isfinite(util)] = -np.inf
    if lam and model_costs is not None:
        util = util - lam * np.asarray(model_costs, np.float64)[None, :]
        # re-mask: a NaN / inf cost yields a NaN utility, which argmax would pick
        util[~np.isfinite(util)] = -np.inf
    if eligible is not None:
        util = np.where(np.asarray(eligible, bool), util, -np.inf)
    rows_all_bad = no_selectable_rows(util)
    if rows_all_bad.any():   # nothing eligible -> fall back to column 0, flagged by caller
        util[rows_all_bad, 0] = -1e18
    return util.argmax(axis=1)


def no_selectable_rows(util: np.ndarray) -> np.ndarray:
    """``[Q]`` bool: rows with no finite utility, i.e. the ones
    :func:`routing_decision` silently sends to column 0."""
    return ~np.isfinite(np.asarray(util, np.float64)).any(axis=1)
