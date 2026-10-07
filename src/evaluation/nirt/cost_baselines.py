"""Baselines and metrics for the expected-cost probe (``BaselineNIRT`` cost head).

Every predictor here estimates the ex-ante expected cost ``E[C | q, m]`` (USD) and
is graded against the REALIZED per-cell cost:

* **B0** global train-mean cost
* **B1** per-model train-mean cost -- today's routing ``C(m)``
* **B2** analytic ``p_in(m) * L_in(q, m) + p_out(m) * mean_train(L_out | m)``:
  input length is known before routing, output length is not. Prices are either
  train-fitted (``fit_token_prices``) or list prices; NaN wherever a model is
  unpriced or a token count is missing.

Inputs are long ``(query_id, model_id, cost[, input_tokens, output_tokens])``
frames, one row per cell. Token counts come from
:func:`training.data.tokens.cell_token_table` (not in the NIRT observation table).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _per_token(price: Optional[dict], side: str) -> float:
    """USD per token from either a fitted price (``p_in`` / ``p_out``, per token;
    :func:`training.data.tokens.fit_token_prices`) or a list price
    (``input_per_1m`` / ``output_per_1m``; ``configs/model_prices.yaml``)."""
    if not price:
        return np.nan
    if f"p_{side}" in price:
        return float(price[f"p_{side}"])
    v = price.get(f"{'input' if side == 'in' else 'output'}_per_1m")
    return float(v) / 1e6 if v is not None else np.nan


def cost_baselines(train: pd.DataFrame, cells: pd.DataFrame,
                   prices: Optional[dict] = None) -> dict[str, np.ndarray]:
    """``{"B0", "B1", "B2"}`` -> expected-USD predictions aligned to ``cells`` rows.
    Statistics come from ``train`` only."""
    tr_cost = pd.to_numeric(train["cost"], errors="coerce")
    ok = np.isfinite(tr_cost) & (tr_cost >= 0)   # $0 is a real price (free models), not missing
    tr = train.loc[ok].assign(cost=tr_cost[ok])
    models = cells["model_id"].astype(str)

    b0 = np.full(len(cells), float(tr["cost"].mean()))
    per_model = tr.groupby(tr["model_id"].astype(str))["cost"].mean()
    b1 = models.map(per_model).to_numpy(np.float64)
    b1 = np.where(np.isfinite(b1), b1, b0)          # unseen model -> global mean

    b2 = np.full(len(cells), np.nan)
    if prices and "input_tokens" in cells.columns:
        out_tok = pd.to_numeric(train.get("output_tokens"), errors="coerce") \
            if "output_tokens" in train.columns else pd.Series(np.nan, index=train.index)
        mean_out = out_tok.groupby(train["model_id"].astype(str)).mean()
        p_in = models.map(lambda m: _per_token(prices.get(m), "in"))
        p_out = models.map(lambda m: _per_token(prices.get(m), "out"))
        l_in = pd.to_numeric(cells["input_tokens"], errors="coerce")
        b2 = (p_in.astype(float) * l_in + p_out.astype(float) * models.map(mean_out)
              ).to_numpy(np.float64)
    return {"B0": b0, "B1": b1, "B2": b2}


def cost_prediction_metrics(pred: np.ndarray, true: np.ndarray,
                            query_ids: np.ndarray) -> dict:
    """Error of expected-USD predictions against realized cost, over cells where
    both are finite and positive. ``coverage`` = that fraction of rows.

    ``mae_log`` / ``spearman`` (all cells) / ``within_query_spearman`` (mean over
    queries with >= 3 scored models: the ranking a router actually uses) /
    ``weighted_abs_error`` (``sum|pred - true| / sum true``, in USD)."""
    from scipy.stats import spearmanr

    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    ok = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    out = {"coverage": float(ok.mean()) if len(ok) else 0.0, "n": int(ok.sum())}
    if ok.sum() < 2:
        return {**out, "mae_log": np.nan, "spearman": np.nan,
                "within_query_spearman": np.nan, "weighted_abs_error": np.nan}
    p, t, q = pred[ok], true[ok], np.asarray(query_ids)[ok]
    rhos = []
    for idx in pd.Series(np.arange(len(q))).groupby(q).indices.values():
        if len(idx) >= 3 and np.ptp(t[idx]) > 0 and np.ptp(p[idx]) > 0:
            rhos.append(spearmanr(p[idx], t[idx]).correlation)
    return {
        **out,
        "mae_log": float(np.abs(np.log(p) - np.log(t)).mean()),
        "spearman": float(spearmanr(p, t).correlation) if np.ptp(p) > 0 else np.nan,
        "within_query_spearman": float(np.mean(rhos)) if rhos else np.nan,
        "weighted_abs_error": float(np.abs(p - t).sum() / t.sum()),
    }


def lambda_grid(pred_quality: np.ndarray, pred_cost: np.ndarray, *,
                quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), n: int = 9) -> np.ndarray:
    """A lambda grid derived from the per-cell cost distribution: ``dq / dc_k``,
    ``dq`` the median within-query spread of predicted quality and ``dc_k`` the
    ``k``-th quantile of the within-query spread of predicted cost -- the trade
    rates at which cost differences start to flip decisions. Returned log-spaced
    over that range, with 0 prepended. ``[Q, M]`` inputs."""
    pq, pc = np.asarray(pred_quality, np.float64), np.asarray(pred_cost, np.float64)
    dq = float(np.nanmedian(np.nanmax(pq, 1) - np.nanmin(pq, 1)))
    dc = np.nanmax(pc, 1) - np.nanmin(pc, 1)
    dc = dc[np.isfinite(dc) & (dc > 0)]
    if not len(dc) or not dq > 0:
        return np.array([0.0])
    rates = dq / np.quantile(dc, quantiles)
    return np.concatenate([[0.0], np.geomspace(rates.min(), rates.max(), n)])
