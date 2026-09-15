"""Data quality checks and the Phase 0 summary report.

Two severity levels:
  * ERROR   -- genuinely invalid data; the pipeline should stop.
  * WARNING -- recoverable / missing-metadata; report and continue.

The check set is driven by the acceptance criteria: duplicate observations,
missing model / query ids, out-of-range scores, inconsistent metric ranges,
impossible token counts, conflicting-id duplicate queries, un-canonicalized
model aliases, missing multiple-choice metadata, and malformed rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import schemas
from .model_registry import is_canonical
from .normalize import content_hash


@dataclass
class QualityIssue:
    level: str          # "error" | "warning"
    code: str
    message: str
    count: int = 0
    examples: list[Any] = field(default_factory=list)


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            lines = [f"  [{i.code}] {i.message} (n={i.count})" for i in self.errors]
            raise ValueError("Data quality check failed:\n" + "\n".join(lines))

    def render(self) -> str:
        out = ["=== Phase 0 data summary ==="]
        for k, v in self.summary.items():
            out.append(f"{k:<32} {v}")
        if self.issues:
            out.append("\n=== Issues ===")
            for i in self.issues:
                out.append(f"[{i.level.upper():7}] {i.code}: {i.message} (n={i.count})")
                for ex in i.examples[:3]:
                    out.append(f"           e.g. {ex}")
        else:
            out.append("\nNo issues detected.")
        return "\n".join(out)


def _pairwise_mask(df: pd.DataFrame) -> pd.Series:
    return df["metric_type"].astype("string").isin(schemas.MetricType.PAIRWISE)


def check_responses(df: pd.DataFrame) -> QualityReport:
    report = QualityReport()
    issues = report.issues

    # -- structural -----------------------------------------------------------
    for col in schemas.RESPONSE_REQUIRED_NON_NULL:
        n = int(df[col].isna().sum())
        if n:
            issues.append(QualityIssue("error", f"null_{col}", f"{col} is null", n,
                                       df.index[df[col].isna()].tolist()[:5]))

    # -- score ranges -------------------------------------------------------
    bad_range_total = 0
    for metric, (lo, hi) in schemas.METRIC_RANGES.items():
        m = df["metric_type"].eq(metric)
        if not m.any():
            continue
        s = pd.to_numeric(df.loc[m, "score"], errors="coerce")
        bad = m & ((df["score"] < lo) | (df["score"] > hi))
        nb = int(bad.sum())
        if nb:
            bad_range_total += nb
            issues.append(QualityIssue(
                "error", "score_out_of_range",
                f"{metric}: {nb} scores outside [{lo}, {hi}]", nb,
                df.loc[bad, ["model_id", "dataset", "score"]].head(5).to_dict("records"),
            ))
    unknown_metrics = sorted(set(df["metric_type"].dropna()) - schemas.MetricType.ALL)
    if unknown_metrics:
        issues.append(QualityIssue("warning", "unknown_metric_type",
                                   f"metric types not in vocabulary: {unknown_metrics}",
                                   len(unknown_metrics)))

    # -- duplicate observations ------------------------------------------
    # Canonical rule: one observation per (query_id, model_id, metric_type).
    # Documented exception: pairwise sources (Arena, GPT-4 Judge) fan a battle
    # into 2 rows and the same (query, model) can be compared against many
    # opponents. So we check absolute metrics strictly and pairwise on
    # (pair_id, role).
    non_pairwise = df[~_pairwise_mask(df)]
    dup_key = ["query_id", "model_id", "metric_type"]
    dmask = non_pairwise.duplicated(subset=dup_key, keep=False)
    ndup = int(dmask.sum())
    if ndup:
        issues.append(QualityIssue(
            "error", "duplicate_observation",
            f"{ndup} rows share (query_id, model_id, metric_type)", ndup,
            non_pairwise.loc[dmask, dup_key].head(5).to_dict("records"),
        ))
    pairwise = df[_pairwise_mask(df)]
    if not pairwise.empty:
        # Each battle legitimately yields 2 rows (one per participant). A given
        # (pair_id, role) must be unique.
        akey = pd.DataFrame({
            "pair_id": pairwise["metadata"].apply(lambda m: (m or {}).get("pair_id")),
            "role": pairwise["metadata"].apply(lambda m: (m or {}).get("role")),
        })
        adup = akey.duplicated(keep=False) & akey["pair_id"].notna()
        if int(adup.sum()):
            issues.append(QualityIssue(
                "error", "duplicate_pairwise_battle",
                f"{int(adup.sum())} pairwise rows share a (pair_id, role)", int(adup.sum()),
            ))

    # -- token counts -----------------------------------------------------
    for col in ("input_tokens", "output_tokens"):
        neg = df[col].notna() & (df[col] < 0)
        nn = int(neg.sum())
        if nn:
            issues.append(QualityIssue("error", f"impossible_{col}",
                                       f"{col} negative", nn))
    for col in ("cost", "latency"):
        neg = df[col].notna() & (df[col] < 0)
        if int(neg.sum()):
            issues.append(QualityIssue("error", f"impossible_{col}",
                                       f"{col} negative", int(neg.sum())))

    # -- model canonicalization ----------------------------------------
    non_canon = sorted({m for m in df["model_id"].dropna().unique() if not is_canonical(m)})
    if non_canon:
        issues.append(QualityIssue(
            "warning", "uncanonicalized_model",
            f"model_ids not in registry (add an alias): {non_canon}", len(non_canon),
        ))

    # -- multiple-choice metadata -------------------------------------
    mc_missing = df["is_multiple_choice"].fillna(False) & df["n_choices"].isna()
    nmm = int(mc_missing.sum())
    if nmm:
        issues.append(QualityIssue(
            "warning", "missing_mc_choices",
            f"{nmm} MC observations lack n_choices (no chance correction applied)", nmm,
            sorted(df.loc[mc_missing, "dataset"].dropna().unique())[:10],
        ))

    # -- conflicting-id duplicate queries -----------------------------
    # same normalized content, different query_id across sources is expected and
    # handled by the split dedup. But same query_id with *different* content is a
    # hash/id bug.
    from .normalize import normalize_query_text

    # text is per query: work on distinct (query_id, query) pairs, not every row
    pairs = df[["query_id", "query"]].drop_duplicates()
    norm = pairs.assign(_norm=pairs["query"].map(normalize_query_text))
    qtext = norm.groupby("query_id")["_norm"].nunique()
    conflicting = qtext[qtext > 1]
    if len(conflicting):
        issues.append(QualityIssue(
            "error", "query_id_content_conflict",
            f"{len(conflicting)} query_ids map to >1 distinct normalized query "
            f"(hash collision / id bug)",
            len(conflicting), conflicting.index.tolist()[:5],
        ))
    # raw-text variants that normalize together are expected (dedup target) but
    # worth reporting as a warning.
    raw_var = pairs.groupby("query_id")["query"].nunique()
    raw_var = raw_var[raw_var > 1]
    if len(raw_var) and not len(conflicting):
        issues.append(QualityIssue(
            "warning", "query_raw_text_variants",
            f"{len(raw_var)} query_ids collapse >1 raw text via normalization",
            len(raw_var), raw_var.index.tolist()[:5],
        ))

    # -- summary --------------------------------------------------------
    report.summary = _summarize(df)
    return report


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
    mc = df["is_multiple_choice"].fillna(False).astype(bool)
    corrected = df["chance_corrected"] if "chance_corrected" in df.columns else pd.Series(False, index=df.index)
    non_pairwise = df[~_pairwise_mask(df)]
    dmask = non_pairwise.duplicated(subset=["query_id", "model_id", "metric_type"], keep=False)
    return {
        "Total observations": len(df),
        "Unique queries": df["query_id"].nunique(),
        "Unique models": df["model_id"].nunique(),
        "Unique datasets": df["dataset"].nunique(),
        "Observations by metric": df["metric_type"].value_counts().to_dict(),
        "Observations by source": df["source"].value_counts().to_dict(),
        "Correctness observations": int(df["metric_type"].isin(schemas.MetricType.CORRECTNESS).sum()),
        "Pairwise observations": int(_pairwise_mask(df).sum()),
        "Missing cost": int(df["cost"].isna().sum()),
        "Missing input_tokens": int(df["input_tokens"].isna().sum()),
        "Missing output_tokens": int(df["output_tokens"].isna().sum()),
        "Missing latency": int(df["latency"].isna().sum()),
        "Multiple-choice observations": int(mc.sum()),
        "Chance-corrected observations": int(pd.Series(corrected).fillna(False).astype(bool).sum()),
        "Duplicate observations (absolute metrics)": int(dmask.sum()),
        "Distinct content hashes": (
            df["content_hash"].nunique() if "content_hash" in df.columns
            else df["query"].drop_duplicates().map(content_hash).nunique()
        ),
    }
