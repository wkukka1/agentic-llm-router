"""The routing policy itself -- pure argmax, no ground truth involved.

Split out of ``evaluation.routing.oracle`` (which owns everything
oracle/regret-related, all needing observed outcomes) so serving code --
:mod:`router.routing.base` -- can make a decision without importing
``evaluation``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def linear_cost(value, pool_mean_costs) -> np.ndarray:
    """Map a cost onto ``[0, 1]`` by the candidate pool's per-model mean-cost
    range: ``(value - min) / (max - min)``, clipped. ``0`` when every pool
    model costs the same (no range to normalize against).

    The one normalization an alpha-blended reward (IRT-Router's ``alpha *
    quality - (1 - alpha) * cost``) and :func:`alpha_to_lam`-driven routing
    both need -- shared here so ``evaluation.nirt.routing.reward`` (offline
    policy scoring) and live routing agree on the same scale."""
    c = np.asarray(pool_mean_costs, dtype=np.float64)
    lo, hi = float(c.min()), float(c.max())
    v = np.asarray(value, dtype=np.float64)
    return np.zeros_like(v) if hi <= lo else np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def alpha_to_lam(alpha: float) -> float:
    """The cost weight ``lam`` whose ``argmax[pred - lam * C]`` agrees with
    the IRT-Router reward's ``argmax[alpha * pred - (1 - alpha) * C]`` for any
    fixed query (both constant-positive-rescale the same ordering): ``lam =
    (1 - alpha) / alpha``. ``alpha = 1`` -> ``lam = 0`` (quality-only);
    ``alpha`` close to ``0`` -> a very large ``lam`` (near cost-only).

    ``alpha`` must be in ``(0, 1]`` -- ``0`` would need an infinite ``lam``."""
    a = float(alpha)
    if not 0.0 < a <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return (1.0 - a) / a


def cost_matrix(model_costs, shape: tuple) -> np.ndarray:
    """``model_costs`` as a ``[Q, M]`` array for a ``shape == (Q, M)`` score
    matrix. Accepts exactly ``(M,)`` (per-model ``C(m)``, broadcast over queries)
    or exactly ``(Q, M)`` (per-cell ``C(q, m)``); anything else raises rather than
    broadcasting silently -- a 1-D array is always read as per-model, even when
    ``Q == M``."""
    c = np.asarray(model_costs, np.float64)
    q, m = shape
    if c.ndim == 1 and c.shape == (m,):
        return np.broadcast_to(c[None, :], (q, m))
    if c.ndim == 2 and c.shape == (q, m):
        return c
    raise ValueError(
        f"model_costs has shape {c.shape}; expected ({m},) per-model or ({q}, {m}) per-cell "
        f"for a {q}x{m} score matrix"
    )


def cost_aware_utility(
    pred: np.ndarray,
    *,
    lam: float = 0.0,
    model_costs: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``[Q, M]`` utility ``pred - lam * C``, non-finite cells and ineligible
    candidates masked to ``-inf``.

    ``model_costs`` is either a length-``M`` per-model vector ``C(m)`` or a
    ``[Q, M]`` per-cell matrix ``C(q, m)`` (e.g. a predicted expected cost), see
    :func:`cost_matrix`. ``eligible`` is an optional ``[Q, M]`` bool mask --
    ineligible cells are never selected.

    The one place the router's cost-aware utility formula is defined --
    :func:`routing_decision` (the served decision rule) and
    ``evaluation.routing.oracle._cost_aware_oracle`` (the offline cost-aware
    oracle, which needs the full utility matrix, not just the argmax) both call
    this instead of reimplementing ``pred - lam * C``.
    """
    util = np.asarray(pred, np.float64).copy()
    C = cost_matrix(model_costs, util.shape) if model_costs is not None else None
    # a missing prediction (no observation for that cell) must never be picked
    util[~np.isfinite(util)] = -np.inf
    if lam and C is not None:
        util = util - lam * C
        # re-mask: a NaN / inf cost yields a NaN utility, which argmax would pick
        util[~np.isfinite(util)] = -np.inf
    if eligible is not None:
        util = np.where(np.asarray(eligible, bool), util, -np.inf)
    return util


def routing_decision(
    pred: np.ndarray,
    *,
    lam: float = 0.0,
    model_costs: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-query model index for ``argmax_m [pred - lam * C(m)]``.

    Exact ties go to the lowest column index (``np.argmax`` semantics), i.e. the
    first-listed model in ``pred``'s column order -- not cost-broken. This is the
    router's live decision rule; it is deliberately independent of
    ``evaluation.routing.oracle``'s tie-*inclusive* oracle labels, which grade the
    router against ground truth and must never feed back into it.
    """
    util = cost_aware_utility(pred, lam=lam, model_costs=model_costs, eligible=eligible)
    rows_all_bad = no_selectable_rows(util)
    if rows_all_bad.any():   # nothing eligible -> fall back to column 0, flagged by caller
        util[rows_all_bad, 0] = -1e18
    return util.argmax(axis=1)


def no_selectable_rows(util: np.ndarray) -> np.ndarray:
    """``[Q]`` bool: rows with no finite utility, i.e. the ones
    :func:`routing_decision` silently sends to column 0."""
    return ~np.isfinite(np.asarray(util, np.float64)).any(axis=1)


def regret(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    selected: Optional[np.ndarray] = None,
    lam: float = 0.0,
    model_costs: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``[Q]`` per-query regret: ``max(0, row_max(true) - true[q, m_hat])``,
    ``m_hat`` the router's pick -- :func:`routing_decision` on ``pred``
    (``lam``/``model_costs``/``eligible`` forwarded), or an already-computed
    ``selected`` array to avoid recomputing it.

    Dense-matrix regret on a quality-only (``lam=0``) selection: the one place
    this is defined for ``evaluation.nirt.evaluate.ranking_metrics`` and
    ``evaluation.routing.oracle.routing_evaluation`` (XD-06), both of which
    already agreed on it (tie-break by lowest column index, clipped at 0) --
    consolidated so they can't silently drift apart.

    Not used by ``training.nirt.train._val_regret``: that function operates on
    long/grouped training-batch data (not a dense ``[Q, M]`` matrix), excludes
    queries with fewer than 2 scored candidates, tie-breaks by row order, and
    returns 0.0 on an empty split -- real, deliberate differences for its job
    (early-stopping during training), not an accidental duplicate of this.
    """
    true = np.asarray(true, np.float64)
    if selected is None:
        selected = routing_decision(pred, lam=lam, model_costs=model_costs, eligible=eligible)
    qi = np.arange(len(selected))
    return np.maximum(true.max(axis=1) - true[qi, selected], 0.0)
