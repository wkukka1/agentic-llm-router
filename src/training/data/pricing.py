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
