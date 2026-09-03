"""Assemble the canonical processed tables from normalized observations.

Outputs (under ``paths.processed``):
  responses.parquet  one row per observation, canonical schema + chance-correction
  queries.parquet    one row per unique query_id
  models.parquet     one row per unique model_id

``build_response_matrix`` also exposes :func:`pivot` to materialize ``R[q, m]``
for a single chosen metric. Different metrics are never merged onto one scale --
the caller picks exactly one ``metric_type`` (or ``score_column``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..config import Config
from . import schemas
from .chance_correction import apply_chance_correction
from .loaders import load_all
from .model_registry import model_info
from .normalize import content_hash
from .pricing import fill_costs


def collapse_duplicate_observations(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse exact ``(query_id, model_id, metric_type)`` duplicates.

    Some source benchmarks (e.g. RouterBench) contain a query verbatim more than
    once. The canonical response matrix keeps one row per
    (query_id, model_id, metric_type); duplicates are averaged and the fan-in is
    recorded in ``metadata['collapsed_from']``. Arena rows are exempt (pairwise;
    keyed on the battle) and pass through untouched.
    """
    arena = df["metric_type"].astype("string").isin(schemas.MetricType.PAIRWISE)
    non_arena, keep = df[~arena].copy(), df[arena]
    key = ["query_id", "model_id", "metric_type"]
    dup = non_arena.duplicated(key, keep=False)
    n_collapsed = int(dup.sum())
    if not n_collapsed:
        return df, 0

    dup_rows = non_arena[dup]
    uniq_rows = non_arena[~dup]

    num_cols = ["score", "cost", "latency", "input_tokens", "output_tokens"]
    agg = dup_rows.groupby(key, as_index=False, sort=False)[num_cols].mean()
    sizes = dup_rows.groupby(key, as_index=False, sort=False).size()

    first = dup_rows.drop_duplicates(key).drop(columns=num_cols)
    collapsed = first.merge(agg, on=key).merge(sizes, on=key)
    collapsed["metadata"] = [
        {**(md if isinstance(md, dict) else {}), "collapsed_from": int(n)}
        for md, n in zip(collapsed["metadata"], collapsed["size"])
    ]
    collapsed = collapsed.drop(columns=["size"])

    out = pd.concat([uniq_rows, collapsed, keep], ignore_index=True)
    return schemas.coerce_response_frame(out), n_collapsed


def build_tables(
    cfg: Config,
    sources: Optional[Iterable[str]] = None,
    responses: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    raw = responses if responses is not None else load_all(cfg, sources)
    raw = schemas.coerce_response_frame(raw)
    raw, n_collapsed = collapse_duplicate_observations(raw)
    if n_collapsed:
        import warnings

        warnings.warn(
            f"collapsed {n_collapsed} exact duplicate observations "
            f"(same query_id/model_id/metric_type) by averaging."
        )
    raw = apply_chance_correction(raw, cfg)

    # Fill per-(model, query) cost for non-RouterBench sources from the documented
    # price snapshot (configs/model_prices.yaml). RouterBench costs are untouched;
    # models without a price stay cost=NaN and are dropped from the routing eval.
    raw = fill_costs(raw, cfg)

    raw["content_hash"] = raw["query"].map(content_hash)

    queries = _build_queries(raw)
    models = _build_models(raw)
    return {"responses": raw, "queries": queries, "models": models}


def _build_queries(df: pd.DataFrame) -> pd.DataFrame:
    first = df.sort_values("source").groupby("query_id", as_index=False).agg(
        query=("query", "first"),
        dataset=("dataset", "first"),
        source=("source", "first"),
        content_hash=("content_hash", "first"),
        n_observations=("score", "size"),
        n_models=("model_id", "nunique"),
    )
    first["cluster_id"] = pd.NA          # filled by clustering stage (train only)
    first["metadata"] = [{} for _ in range(len(first))]
    return first


def _build_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, g in df.groupby("model_id"):
        info = model_info(model_id)
        rows.append(
            {
                "model_id": model_id,
                "model_name": info.model_name,
                "provider": info.provider,
                "family": info.family,
                "version": info.version,
                "profile_text": "",   # populated by model-profile stage (Phase 0 opt.)
                "source_datasets": sorted(g["dataset"].dropna().unique().tolist()),
                "n_observations": len(g),
                "sources": sorted(g["source"].dropna().unique().tolist()),
                "metadata": {
                    "native_names": sorted(
                        {
                            (m or {}).get("native_model_name")
                            for m in g["metadata"]
                            if (m or {}).get("native_model_name")
                        }
                    ),
                },
            }
        )
    return pd.DataFrame(rows)


def write_tables(tables: dict[str, pd.DataFrame], cfg: Config) -> dict[str, Path]:
    out_dir = cfg.path("processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in tables.items():
        p = out_dir / f"{name}.parquet"
        _to_parquet(frame, p)
        paths[name] = p
    return paths


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    # JSON-encode dict/list columns for portable parquet round-tripping.
    import json

    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, (dict, list))).any():
            out[col] = out[col].map(
                lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
            )
    out.to_parquet(path, index=False)


def read_responses(cfg: Config) -> pd.DataFrame:
    import json

    df = pd.read_parquet(cfg.path("processed") / "responses.parquet")
    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].map(
            lambda v: json.loads(v) if isinstance(v, str) and v else (v or {})
        )
    return df


def pivot_qm(frame: pd.DataFrame, value: str, *, aggfunc: str = "mean") -> pd.DataFrame:
    """Dense ``[query_id x model_id]`` matrix of ``frame[value]``.

    The one place the ``pivot_table(index="query_id", columns="model_id", ...)``
    idiom lives -- shared by the response matrix, the Phase 1 facade, the routing
    layer, the classical-IRT baselines and the pool-expansion battery.
    """
    return frame.pivot_table(index="query_id", columns="model_id", values=value, aggfunc=aggfunc)


def pivot(
    responses: pd.DataFrame,
    metric_type: str,
    score_column: str = "score",
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Return R[q, m] for a single metric. Rows = query_id, cols = model_id."""
    sub = responses[responses["metric_type"] == metric_type]
    if sub.empty:
        raise ValueError(f"no observations with metric_type={metric_type!r}")
    return pivot_qm(sub, score_column, aggfunc=aggfunc)
