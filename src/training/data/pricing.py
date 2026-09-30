"""Per-model token pricing for sources that carry no precomputed cost.

RouterBench ships a token-weighted ``total_cost`` per (model, query); it is used
verbatim. lm-evaluation-harness and external benchmarks carry only token counts
(or none), so their per-(model, query) cost is filled here from a documented
price snapshot in ``configs/model_prices.yaml``:

    cost = input_tokens  * input_per_1m  / 1e6
         + output_tokens * output_per_1m / 1e6

A model with no price entry keeps ``cost = NaN`` -- the pool-expansion battery
drops such models from the *routing* eval (not the *prediction* eval) and
records it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from router.config import Config
from .model_registry import canonical_model_id

PRICE_CONFIG = "configs/model_prices.yaml"


@lru_cache(maxsize=8)
def _load_raw(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_prices(cfg: Optional[Config] = None, *, path: Optional[str] = None) -> dict[str, dict]:
    """``canonical_model_id -> {input_per_1m, output_per_1m, source, ...}``.

    Keys are canonicalised through the model registry so a price written under a
    HF repo path or a native label still matches the ingested ``model_id``.
    """
    root = getattr(cfg, "root", None)
    fp = path or (str(root / PRICE_CONFIG) if root else PRICE_CONFIG)
    raw = _load_raw(fp)
    out: dict[str, dict] = {}
    for key, val in (raw.get("prices") or {}).items():
        if not isinstance(val, dict) or "input_per_1m" not in val:
            continue
        cid = canonical_model_id(key)
        out[cid] = {**val, "_key": key}
        # also index by the given key verbatim (already-canonical configs)
        out.setdefault(str(key), out[cid])
    return out


def price_snapshot(cfg: Optional[Config] = None, *, path: Optional[str] = None) -> dict:
    root = getattr(cfg, "root", None)
    fp = path or (str(root / PRICE_CONFIG) if root else PRICE_CONFIG)
    raw = _load_raw(fp)
    return {k: raw.get(k) for k in ("snapshot_date", "currency", "unit", "basis")}


def cost_for(model_id: str, input_tokens, output_tokens, prices: dict) -> float:
    p = prices.get(model_id) or prices.get(canonical_model_id(model_id))
    if not p:
        return float("nan")
    it = float(input_tokens) if input_tokens is not None and not pd.isna(input_tokens) else 0.0
    ot = float(output_tokens) if output_tokens is not None and not pd.isna(output_tokens) else 0.0
    return it * float(p["input_per_1m"]) / 1e6 + ot * float(p["output_per_1m"]) / 1e6


def fill_costs(df: pd.DataFrame, cfg: Optional[Config] = None, *,
               path: Optional[str] = None, sources: Optional[set[str]] = None) -> pd.DataFrame:
    """Fill ``cost`` where it is missing, from token counts x price.

    Only touches rows with ``cost`` NaN. Rows from ``sources`` (default: every
    source except ``routerbench``) are eligible. Rows whose model has no price
    or no token counts are left NaN.
    """
    if "cost" not in df.columns:
        df = df.copy()
        df["cost"] = np.nan
    prices = load_prices(cfg, path=path)
    if not prices:
        return df

    out = df.copy()
    need = out["cost"].isna()
    if sources is None:
        need &= out["source"].astype(str) != "routerbench"
    else:
        need &= out["source"].astype(str).isin(sources)
    if not need.any():
        return out

    sub = out.loc[need]
    it = pd.to_numeric(sub.get("input_tokens"), errors="coerce").fillna(0.0).astype(float)
    ot = pd.to_numeric(sub.get("output_tokens"), errors="coerce").fillna(0.0).astype(float)
    # one price lookup per distinct model, then a vectorised cost (== cost_for per row)
    rates = {}
    for mid in sub["model_id"].astype(str).unique():
        p = prices.get(mid) or prices.get(canonical_model_id(mid))
        rates[mid] = ((float(p["input_per_1m"]), float(p["output_per_1m"])) if p
                      else (np.nan, np.nan))
    mids = sub["model_id"].astype(str)
    in_rate = mids.map(lambda m: rates[m][0]).astype(float)
    out_rate = mids.map(lambda m: rates[m][1]).astype(float)
    out.loc[need, "cost"] = it * in_rate / 1e6 + ot * out_rate / 1e6
    return out


def priced_model_ids(cfg: Optional[Config] = None, *, path: Optional[str] = None) -> set[str]:
    return {k for k in load_prices(cfg, path=path)}


def impute_missing_usage(df: pd.DataFrame, *, min_priced_rows: int = 30) -> pd.DataFrame:
    """Price rows whose usage accounting is missing: 0 input tokens, 0 output
    tokens and ``cost == 0`` on a row that carries a real score.

    A prompt cannot have zero input tokens, so these are answered calls whose
    usage was not recorded, not free ones. Ingesting them as $0 understates the
    model's cost and lets it win cost tie-breaks (KNOWN BUG: training.data.loaders
    MOD-001). A genuinely free model is unaffected: its rows carry real token
    counts, so they never match.

    For each affected model the per-token rates are recovered by least squares
    from that model's own priced rows (``cost = in * r_in + out * r_out``) --
    the source's own pricing, not an external table. Tokens are then estimated:

    * input: the median input count for the same query on other models, else
      the model's mean for that dataset;
    * output: the model's mean for that dataset (else its overall mean).

    A model with fewer than ``min_priced_rows`` priced rows, or a non-positive
    fitted rate, has no trustworthy price: its affected rows get NaN tokens and
    NaN cost (the schema's "unknown", never 0) and ``metadata["usage_missing"]``.
    Every imputed row is marked ``metadata["usage_imputed"] = True``; rows with
    real usage are not touched.
    """
    out = df.copy()
    # float, not the schema's nullable Int64, so the masks below never hold <NA>
    tok_in = pd.to_numeric(out["input_tokens"], errors="coerce").astype(float)
    tok_out = pd.to_numeric(out["output_tokens"], errors="coerce").astype(float)
    cost = pd.to_numeric(out["cost"], errors="coerce").astype(float)
    missing = (tok_in == 0) & (tok_out == 0) & (cost == 0)
    if not missing.any():
        return out

    priced = (tok_in > 0) & (cost > 0)
    for model in out.loc[missing, "model_id"].unique():
        rows = missing & (out["model_id"] == model)
        own = priced & (out["model_id"] == model)
        rate = None
        if own.sum() >= min_priced_rows:
            coef, *_ = np.linalg.lstsq(
                np.column_stack([tok_in[own], tok_out[own]]).astype(float),
                cost[own].to_numpy(float), rcond=None)
            if (coef > 0).all():
                rate = coef
        flag = "usage_imputed"
        if rate is None:
            out.loc[rows, ["input_tokens", "output_tokens", "cost"]] = np.nan
            flag = "usage_missing"
        else:
            elsewhere = priced & (out["model_id"] != model)
            by_query = tok_in[elsewhere].groupby(out.loc[elsewhere, "query_id"]).median()
            own_in = tok_in[own].groupby(out.loc[own, "dataset"]).mean()
            own_out = tok_out[own].groupby(out.loc[own, "dataset"]).mean()
            ds = out.loc[rows, "dataset"]
            est_in = out.loc[rows, "query_id"].map(by_query).fillna(ds.map(own_in)).fillna(tok_in[own].mean())
            est_out = ds.map(own_out).fillna(tok_out[own].mean())
            est_in, est_out = est_in.round(), est_out.round()
            out.loc[rows, "input_tokens"] = est_in
            out.loc[rows, "output_tokens"] = est_out
            out.loc[rows, "cost"] = (est_in * rate[0] + est_out * rate[1]).to_numpy()
        out.loc[rows, "metadata"] = out.loc[rows, "metadata"].map(
            lambda m: {**(m or {}), flag: True})
    return out
