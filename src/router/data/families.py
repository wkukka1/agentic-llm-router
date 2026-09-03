"""Coarse benchmark-family lookup (``configs/*.yaml -> profiles.task_families``).

One place for the ``dataset name -> family`` mapping that model profiles, the
taxonomy labels, OOD splits and the evaluators all need. ``family_of`` matches a
family key exactly first, then as a substring of the dataset name.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..config import Config, section


def task_family_map(cfg: Config) -> dict[str, str]:
    """``{lower-cased dataset key: family}`` from ``profiles.task_families``."""
    fams = section(cfg, "profiles").get("task_families", {}) or {}
    return {str(name).lower(): family for family, names in fams.items() for name in names}


def family_of(dataset: str, fam_map: dict[str, str]) -> Optional[str]:
    """Family for ``dataset`` -- exact key match first, then substring, else ``None``."""
    d = str(dataset or "").lower()
    if d in fam_map:
        return fam_map[d]
    return next((fam for key, fam in fam_map.items() if key in d), None)


def family_labels(
    query_ids: Sequence[str],
    cfg: Config,
    *,
    queries_df=None,
    fallback: str = "other",
) -> list[str]:
    """Family label per query id (``fallback`` for queries with no mapped family).

    ``queries_df`` (columns ``query_id``, ``dataset``) is read from
    ``processed/queries.parquet`` when not supplied.
    """
    fam_map = task_family_map(cfg)
    if queries_df is None:
        import pandas as pd

        queries_df = pd.read_parquet(
            cfg.path("processed") / "queries.parquet", columns=["query_id", "dataset"]
        )
    ds = queries_df.set_index("query_id")["dataset"].to_dict()
    return [family_of(ds.get(q, "") or "", fam_map) or fallback for q in query_ids]
