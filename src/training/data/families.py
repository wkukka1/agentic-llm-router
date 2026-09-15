"""Coarse benchmark-family lookup (``configs/*.yaml -> profiles.task_families``).

One place for the ``dataset name -> family`` mapping that model profiles, the
taxonomy labels, OOD splits and the evaluators all need. ``family_of`` matches a
family key exactly first, then as a substring of the dataset name.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from router.config import Config, section


def task_family_map(cfg: Config) -> dict[str, str]:
    """``{lower-cased dataset key: family}`` from ``profiles.task_families``."""
    fams = section(cfg, "profiles").get("task_families", {}) or {}
    return {str(name).lower(): family for family, names in fams.items() for name in names}


_TOKEN = re.compile(r"[a-z0-9]+")


def _find_tokens(needle: list[str], hay: list[str]) -> int:
    """Start index of ``needle`` as a contiguous run in ``hay``, else -1."""
    k = len(needle)
    for i in range(len(hay) - k + 1):
        if hay[i:i + k] == needle:
            return i
    return -1


def family_of(dataset: str, fam_map: dict[str, str]) -> Optional[str]:
    """Family for ``dataset``, independent of the YAML key order.

    1. exact key match;
    2. a key matching whole tokens of the name (``mmlu`` in
       ``mmlu-high-school-mathematics``, but ``math`` does NOT match inside
       ``mathematics``) -- earliest match wins, then the longer key, so a
       benchmark prefix beats a subject word;
    3. plain substring fallback, with the same earliest-then-longest rule.
    """
    d = str(dataset or "").lower()
    if d in fam_map:
        return fam_map[d]
    toks = _TOKEN.findall(d)
    best = None
    for key, fam in fam_map.items():
        pos = _find_tokens(_TOKEN.findall(key), toks) if _TOKEN.findall(key) else -1
        if pos >= 0:
            cand = (pos, -len(key), fam)
            best = cand if best is None or cand[:2] < best[:2] else best
    if best is None:
        for key, fam in fam_map.items():
            pos = d.find(key)
            if pos >= 0:
                cand = (pos, -len(key), fam)
                best = cand if best is None or cand[:2] < best[:2] else best
    return best[2] if best else None


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
