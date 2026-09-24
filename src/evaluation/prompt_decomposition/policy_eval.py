"""Score a *selection rule* with the project's own routing metrics.

`evaluation.routing.oracle.routing_evaluation` takes a prediction matrix and
selects with argmax internally, so a rule that deliberately picks something
other than the argmax -- :class:`router.policy.MarginRoutingPolicy` -- has no
way in. Reimplementing regret and oracle-hit here would fork the definitions
and make the numbers incomparable with `route_eval.py`.

So the selection is made here and handed to his scorer as an indicator matrix:
the chosen model gets 1, everything else 0, and ``lam=0``. Its argmax is then
exactly our choice, and every reported figure -- selected quality, regret,
oracle hit, cost -- comes out of his code against the same ground truth. Only
metrics that read the *values* of a prediction matrix would be meaningless that
way, and routing metrics read the selection.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from evaluation.routing.oracle import routing_evaluation

log = logging.getLogger(__name__)


def margin_selection(pred: np.ndarray, cost: np.ndarray, *, margin: float,
                     lam: float = 0.0) -> np.ndarray:
    """Cheapest candidate whose utility is within ``margin`` of the best.

    Delegates to :func:`router.nirt.routing_decision.routing_decision` -- the
    rule the *served* router applies -- rather than reimplementing it. It used
    to have its own copy, and the copy had drifted: it broke ties on cost even
    at ``margin=0``, so the sweep's "argmax (incumbent)" control row was not
    actually the incumbent. It quietly collected the tie-breaking savings and
    flattered the baseline, which matters most on a predictor whose scores are
    quantised (a k-neighbour mean ties constantly).

    ``cost`` is the ``[Q, M]`` per-cell cost matrix these evaluations carry,
    while the served rule takes a per-model vector; the column means are used,
    which is the same ``C(m)`` the router itself routes on.
    """
    from router.nirt.routing_decision import routing_decision

    model_costs = np.nanmean(np.asarray(cost, np.float64), axis=0)
    return routing_decision(pred, lam=lam, model_costs=model_costs, margin=margin)


def score_selection(picks: np.ndarray, true: np.ndarray, cost: np.ndarray | None,
                    model_ids: Sequence[str], *, tolerance: float = 0.01,
                    query_ids: Sequence[str] | None = None,
                    strategy: str = "selection") -> dict:
    """Route the given choice through the project's routing metrics."""
    indicator = np.zeros_like(true, dtype=float)
    indicator[np.arange(len(picks)), picks] = 1.0
    return routing_evaluation(indicator, true, cost, model_ids, lam=0.0,
                              tolerance=tolerance, query_ids=query_ids, strategy=strategy)


def margin_sweep(pred: np.ndarray, true: np.ndarray, cost: np.ndarray,
                 model_ids: Sequence[str], *, margins: Sequence[float] = (0.0, 0.01, 0.02,
                                                                          0.05, 0.1, 0.2),
                 lam: float = 0.0, query_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per margin, plus the fixed-model and oracle reference rows.

    ``margin=0`` is the incumbent argmax policy, so the table contains its own
    control: any claim about the margin rule is a difference against the first
    row, measured the same way.
    """
    rows = []
    for margin in margins:
        picks = margin_selection(pred, cost, margin=margin, lam=lam)
        m = score_selection(picks, true, cost, model_ids, query_ids=query_ids,
                            strategy=f"margin={margin:g}")
        rows.append({
            "policy": "argmax (incumbent)" if margin == 0 else f"margin={margin:g}",
            "quality": m["mean_selected_quality"], "cost_per_1k": m["mean_selected_cost_per_1k"],
            "regret": m["mean_regret"], "oracle_hit": m["oracle_hit_rate_any_best"],
            "top_model_share": float((picks == np.argmax(true.mean(axis=0))).mean()),
        })

    best = int(true.mean(axis=0).argmax())
    fixed = score_selection(np.full(len(true), best), true, cost, model_ids,
                            query_ids=query_ids, strategy="fixed")
    rows.append({"policy": f"always {model_ids[best]}", "quality": fixed["mean_selected_quality"],
                 "cost_per_1k": fixed["mean_selected_cost_per_1k"], "regret": fixed["mean_regret"],
                 "oracle_hit": fixed["oracle_hit_rate_any_best"], "top_model_share": 1.0})

    top = true.max(axis=1, keepdims=True)
    cheapest_best = np.where(true >= top, cost, np.inf).argmin(axis=1)
    orc = score_selection(cheapest_best, true, cost, model_ids, query_ids=query_ids,
                          strategy="oracle")
    rows.append({"policy": "oracle (cheapest correct)", "quality": orc["mean_selected_quality"],
                 "cost_per_1k": orc["mean_selected_cost_per_1k"], "regret": orc["mean_regret"],
                 "oracle_hit": orc["oracle_hit_rate_any_best"],
                 "top_model_share": float((cheapest_best == best).mean())})
    return pd.DataFrame(rows)


def frontier_gain(table: pd.DataFrame, cheap_model: str, strong_model: str,
                  per_model: pd.DataFrame) -> pd.DataFrame:
    """Each row's gain over mixing the two reference models at the same cost.

    A cost-quality point means nothing on its own: randomly splitting traffic
    between a cheap model and a strong one traces a straight line between them
    for free, and a policy that sits on that line has bought nothing with its
    predictions. This is the column that says whether the rule is doing work.
    """
    lo_q, lo_c = per_model.loc[cheap_model, "quality"], per_model.loc[cheap_model, "cost_per_1k"]
    hi_q, hi_c = per_model.loc[strong_model, "quality"], per_model.loc[strong_model, "cost_per_1k"]
    out = table.copy()
    span = hi_c - lo_c
    mixture = lo_q + (out["cost_per_1k"] - lo_c) / span * (hi_q - lo_q) if span else np.nan
    # A tolerance, because the endpoints land on the boundary by construction
    # and their two costs come from different code paths (this frame's mean vs
    # `routing_evaluation`'s). Without it the `always <strongest>` row -- the
    # one a reader checks first -- reads NaN from float noise alone.
    slack = abs(span) * 1e-9
    out["mixture_at_same_cost"] = mixture.where(
        (out["cost_per_1k"] >= lo_c - slack) & (out["cost_per_1k"] <= hi_c + slack))
    out["vs_mixture"] = out["quality"] - out["mixture_at_same_cost"]
    return out
