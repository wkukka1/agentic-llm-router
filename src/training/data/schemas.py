"""Canonical Phase 0 data schemas.

Every source dataset (RouterBench, Chatbot Arena, lm-evaluation-harness) is
normalized into these three tables *before* anything downstream touches it.
Downstream code (response matrix, splits, embeddings, Phase 1) must depend only
on the columns declared here, never on a source's native format.

Design rules
------------
* Missing values are represented explicitly as ``NA``/``None`` -- never
  fabricated. In particular ``input_tokens``, ``output_tokens``, ``cost`` and
  ``latency`` are left null when a source does not report them. They are *not*
  filled with 0 (0 tokens / 0 cost is a different, wrong claim).
* ``score`` always holds the original source score on its original scale.
  Any transformed value (e.g. chance-corrected) lives in a separate column.
* ``metric_type`` plus ``metadata`` must be enough to tell F1 / EM / pass@1 /
  accuracy / arena-preference apart. Metrics are never silently rescaled onto a
  common axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Controlled vocabularies                                                     #
# --------------------------------------------------------------------------- #
class Source:
    ROUTERBENCH = "routerbench"
    ARENA = "chatbot_arena"
    GPT4_JUDGE = "gpt4_judge"
    LM_HARNESS = "lm_eval_harness"
    IRT_ROUTER = "irt_router"        # the IRT-Router paper's own 20-LLM x 12-dataset suite
    ANCHOR_JUDGE = "anchor_judge"    # manufactured anchor-topology judge on EXISTING
                                     # routerbench: query_ids (docs/anchor_judge.md) --
                                     # query_id is carried through verbatim from the
                                     # correctness data, never re-derived under this source.
    ALL = {ROUTERBENCH, ARENA, GPT4_JUDGE, LM_HARNESS, IRT_ROUTER, ANCHOR_JUDGE}

    # Sources whose observations are pairwise preferences, not absolute scores.
    PAIRWISE = {ARENA, GPT4_JUDGE, ANCHOR_JUDGE}


class MetricType:
    """Original metric identity. Kept distinguishable end-to-end."""

    ACCURACY = "accuracy"            # 0/1 or graded correctness (RouterBench `performance`)
    EXACT_MATCH = "exact_match"
    F1 = "f1"
    PASS_AT_1 = "pass@1"
    MC_ACCURACY = "mc_accuracy"      # multiple-choice accuracy (chance-correctable)
    ARENA_PREFERENCE = "arena_preference"  # human pairwise outcome in {1.0, 0.5, 0.0}
    JUDGE_PREFERENCE = "judge_preference"  # LLM-judge pairwise outcome in {1.0, 0.5, 0.0}
    LIKELIHOOD_ACC = "acc"           # lm-harness `acc`
    LIKELIHOOD_ACC_NORM = "acc_norm"  # lm-harness `acc_norm`
    ALL = {
        ACCURACY, EXACT_MATCH, F1, PASS_AT_1, MC_ACCURACY,
        ARENA_PREFERENCE, JUDGE_PREFERENCE, LIKELIHOOD_ACC, LIKELIHOOD_ACC_NORM,
    }

    # Pairwise-preference metrics: kept OUT of the absolute response matrix.
    PAIRWISE = {ARENA_PREFERENCE, JUDGE_PREFERENCE}
    # Absolute correctness metrics: the primary AMIRT response signal.
    CORRECTNESS = {ACCURACY, MC_ACCURACY, EXACT_MATCH, F1, PASS_AT_1,
                   LIKELIHOOD_ACC, LIKELIHOOD_ACC_NORM}


# Valid closed interval for each metric's *raw* score.
METRIC_RANGES: dict[str, tuple[float, float]] = {
    MetricType.ACCURACY: (0.0, 1.0),
    MetricType.EXACT_MATCH: (0.0, 1.0),
    MetricType.F1: (0.0, 1.0),
    MetricType.PASS_AT_1: (0.0, 1.0),
    MetricType.MC_ACCURACY: (0.0, 1.0),
    MetricType.ARENA_PREFERENCE: (0.0, 1.0),
    MetricType.JUDGE_PREFERENCE: (0.0, 1.0),
    MetricType.LIKELIHOOD_ACC: (0.0, 1.0),
    MetricType.LIKELIHOOD_ACC_NORM: (0.0, 1.0),
}


# --------------------------------------------------------------------------- #
# Dataclasses (documentation + single-row construction / tests)               #
# --------------------------------------------------------------------------- #
@dataclass
class ResponseObservation:
    query_id: str
    model_id: str
    query: str
    score: float
    metric_type: str
    dataset: str
    split: Optional[str]
    source: str
    is_multiple_choice: bool
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency: Optional[float] = None
    cost: Optional[float] = None
    n_choices: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    query_id: str
    query: str
    dataset: str
    source: str
    cluster_id: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # embedding + relevance_vector are stored in dedicated artifacts keyed by
    # query_id (see embeddings/ and taxonomy/relevance.py), not inline here.


@dataclass
class Model:
    model_id: str
    model_name: str
    provider: str
    profile_text: str = ""
    source_datasets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Column contracts for the parquet tables                                     #
# --------------------------------------------------------------------------- #
RESPONSE_COLUMNS: list[str] = [
    "query_id", "model_id", "query", "score", "metric_type", "dataset",
    "split", "source", "is_multiple_choice", "n_choices",
    "input_tokens", "output_tokens", "latency", "cost", "metadata",
]

# Columns that must be non-null in every response observation.
RESPONSE_REQUIRED_NON_NULL: list[str] = [
    "query_id", "model_id", "query", "score", "metric_type", "source",
]

# Columns explicitly allowed to be null (missing, not zero).
RESPONSE_NULLABLE: list[str] = [
    "split", "n_choices", "input_tokens", "output_tokens", "latency", "cost",
]

QUERY_COLUMNS: list[str] = [
    "query_id", "query", "dataset", "source", "cluster_id", "metadata",
]

MODEL_COLUMNS: list[str] = [
    "model_id", "model_name", "provider", "profile_text",
    "source_datasets", "metadata",
]


def empty_response_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in RESPONSE_COLUMNS})


def coerce_response_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with canonical columns, order, and dtypes.

    Adds any missing optional columns as null. Raises on missing required
    columns so a broken loader fails loudly rather than silently.
    """
    missing_required = [
        c for c in RESPONSE_REQUIRED_NON_NULL if c not in df.columns
    ]
    if missing_required:
        raise ValueError(f"response frame missing required columns: {missing_required}")

    out = df.copy()
    for col in RESPONSE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["query_id"] = out["query_id"].astype("string")
    out["model_id"] = out["model_id"].astype("string")
    out["query"] = out["query"].astype("string")
    out["metric_type"] = out["metric_type"].astype("string")
    out["dataset"] = out["dataset"].astype("string")
    out["split"] = out["split"].astype("string")
    out["source"] = out["source"].astype("string")
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["is_multiple_choice"] = out["is_multiple_choice"].fillna(False).astype(bool)
    for col in ("n_choices", "input_tokens", "output_tokens"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in ("latency", "cost"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["metadata"] = out["metadata"].apply(
        lambda v: v if isinstance(v, dict) else {}
    )
    return out[RESPONSE_COLUMNS]
