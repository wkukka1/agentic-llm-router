"""The training-data facade.

A training script imports **only this module**. It never touches RouterBench
pickles, Arena battle rows, lm-harness logs, or the canonical column layout
directly.

    from router.config import load_config
    from training.data.facade import load_training_data

    d = load_training_data(load_config())

    R      = d.correctness_matrix(split="train")          # R[q, m], absolute correctness
    long   = d.correctness(split="train")                 # same, long form + n_choices
    pairs  = d.pairwise(source="chatbot_arena", split="train")   # (q, m_a, m_b, winner)
    q_emb  = d.query_embeddings(pathway="irt")            # query_id -> BERT vector
    m_emb  = d.profile_embeddings(pathway="irt")          # model_id -> profile vector (theta_m init)
    prof   = d.model_profiles()                           # per-model text + empirical facts
    warm   = d.warm_model_ids()
    cold   = d.cold_start_model_ids()                     # place these from profile_embedding()

Design decisions baked in (per Phase 0 review):
* Correctness (RouterBench / lm-harness) is the **primary** signal; Arena and
  GPT-4-Judge are **pairwise** and are returned separately, never folded into R.
* **No guessing parameter.** Multiple-choice observations carry a data-level
  ``corrected_score``; ``score_effective`` uses it where available. ``n_choices``
  is also exposed for reference but this facade is not expected to fit a ``c_q``.
* Splits are leakage-free query-content groups; cold-start models are excluded
  from ``warm`` views.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Literal, Optional

import pandas as pd

from router.config import Config
from . import schemas
from .response_matrix import read_responses
from .splits import load_splits

ModelScope = Literal["warm", "all", "cold"]
ScoreKind = Literal["effective", "raw", "corrected"]


@dataclass
class TrainingData:
    cfg: Config
    responses: pd.DataFrame
    queries: pd.DataFrame
    models: pd.DataFrame
    splits: dict
    profiles: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # model scopes                                                       #
    # ------------------------------------------------------------------ #
    @cached_property
    def _cold(self) -> set[str]:
        cs = self.splits.get("cold_start_models") or {}
        return set(cs.get("model_ids", []))

    def model_ids(self) -> list[str]:
        return sorted(self.responses["model_id"].dropna().unique().tolist())

    def cold_start_model_ids(self) -> list[str]:
        return sorted(self._cold)

    def warm_model_ids(self) -> list[str]:
        return sorted(m for m in self.model_ids() if m not in self._cold)

    def _scope_models(self, scope: ModelScope) -> Optional[set[str]]:
        if scope == "all":
            return None
        if scope == "warm":
            return set(self.warm_model_ids())
        if scope == "cold":
            return set(self._cold)
        raise ValueError(f"unknown model scope: {scope}")

    # ------------------------------------------------------------------ #
    # query sets / splits                                                #
    # ------------------------------------------------------------------ #
    def split_query_ids(self, split: Optional[str]) -> Optional[set[str]]:
        if split is None:
            return None
        if split not in ("train", "validation", "test", "ood"):
            raise ValueError(f"unknown split: {split}")
        entry = self.splits.get(split)
        if entry is None:
            raise ValueError(f"split '{split}' not built for this config")
        return set(entry["query_ids"])

    def query_texts(self, split: Optional[str] = None) -> dict[str, str]:
        q = self.queries
        ids = self.split_query_ids(split)
        if ids is not None:
            q = q[q["query_id"].isin(ids)]
        return dict(zip(q["query_id"], q["query"]))

    # ------------------------------------------------------------------ #
    # correctness (primary IRT signal)                                   #
    # ------------------------------------------------------------------ #
    def correctness(
        self,
        split: Optional[str] = None,
        metrics: Optional[Iterable[str]] = None,
        models: ModelScope = "warm",
        score_kind: ScoreKind = "effective",
    ) -> pd.DataFrame:
        """Long-form absolute-correctness observations.

        Columns: query_id, model_id, dataset, metric_type, is_multiple_choice,
        n_choices, score_raw, score_corrected, score_effective, score.
        (`score` == the column chosen by `score_kind`.)
        """
        df = self.responses
        df = df[df["metric_type"].isin(schemas.MetricType.CORRECTNESS)]
        if metrics is not None:
            df = df[df["metric_type"].isin(set(metrics))]

        keep = self.split_query_ids(split)
        if keep is not None:
            df = df[df["query_id"].isin(keep)]
        scope = self._scope_models(models)
        if scope is not None:
            df = df[df["model_id"].isin(scope)]

        out = pd.DataFrame(
            {
                "query_id": df["query_id"].astype(str),
                "model_id": df["model_id"].astype(str),
                "dataset": df["dataset"].astype(str),
                "metric_type": df["metric_type"].astype(str),
                "is_multiple_choice": df["is_multiple_choice"].fillna(False).astype(bool),
                "n_choices": df["n_choices"].astype("Int64"),
                "score_raw": pd.to_numeric(df["score"], errors="coerce"),
                "score_corrected": pd.to_numeric(
                    df.get("corrected_score"), errors="coerce"
                ),
            }
        )
        out["score_effective"] = out["score_corrected"].where(
            out["score_corrected"].notna(), out["score_raw"]
        )
        out["score"] = {
            "effective": out["score_effective"],
            "raw": out["score_raw"],
            "corrected": out["score_corrected"],
        }[score_kind]
        return out.reset_index(drop=True)

    def correctness_matrix(
        self,
        split: Optional[str] = None,
        metric: str = schemas.MetricType.ACCURACY,
        models: ModelScope = "warm",
        score_kind: ScoreKind = "effective",
        combine_metrics: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """``R[q, m]`` for one metric (rows=query_id, cols=model_id).

        Pass ``combine_metrics`` (e.g. ``["accuracy", "mc_accuracy"]``) to pool
        RouterBench's free-form and MC correctness into one matrix -- this is
        safe only because ``score_effective`` already put MC on a chance-adjusted
        [0, 1] axis. Different *kinds* of metric (F1 vs EM vs pass@1) are never
        pooled automatically.
        """
        metrics = list(combine_metrics) if combine_metrics else [metric]
        long = self.correctness(split=split, metrics=metrics, models=models,
                                score_kind=score_kind)
        if long.empty:
            raise ValueError(f"no correctness observations for metrics={metrics}")
        from router.nirt.frames import pivot_qm

        return pivot_qm(long, "score")

    def n_choices(self, split: Optional[str] = None) -> dict[str, int]:
        """`query_id -> n_choices` for multiple-choice queries (reference only)."""
        df = self.responses
        df = df[df["is_multiple_choice"].fillna(False) & df["n_choices"].notna()]
        keep = self.split_query_ids(split)
        if keep is not None:
            df = df[df["query_id"].isin(keep)]
        return (
            df.groupby("query_id")["n_choices"].first().astype(int).to_dict()
        )

    # ------------------------------------------------------------------ #
    # pairwise preference (auxiliary signal)                             #
    # ------------------------------------------------------------------ #
    def pairwise(
        self,
        source: Optional[str] = None,
        split: Optional[str] = None,
        models: ModelScope = "warm",
    ) -> pd.DataFrame:
        """One row per battle: query_id, model_a, model_b, winner, score_a,
        source, pair_id.

        ``winner`` in {"a", "b", "tie"}; ``score_a`` in {1.0, 0.5, 0.0} is the
        outcome for ``model_a``. Reconstructed from the role='a' observation.
        """
        df = self.responses[self.responses["metric_type"].isin(schemas.MetricType.PAIRWISE)]
        if source is not None:
            df = df[df["source"] == source]

        md = df["metadata"].apply(lambda m: m or {})
        role = md.apply(lambda m: m.get("role"))
        a = df[role == "a"].copy()
        a_md = a["metadata"].apply(lambda m: m or {})

        out = pd.DataFrame(
            {
                "query_id": a["query_id"].astype(str),
                "model_a": a["model_id"].astype(str),
                "model_b": a_md.apply(lambda m: m.get("opponent_model_id")).astype(str),
                "score_a": pd.to_numeric(a["score"], errors="coerce"),
                "source": a["source"].astype(str),
                "pair_id": a_md.apply(lambda m: m.get("pair_id")),
            }
        )
        out["winner"] = out["score_a"].map({1.0: "a", 0.0: "b", 0.5: "tie"})

        keep = self.split_query_ids(split)
        if keep is not None:
            out = out[out["query_id"].isin(keep)]
        scope = self._scope_models(models)
        if scope is not None:
            out = out[out["model_a"].isin(scope) & out["model_b"].isin(scope)]
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # LLM profiles (NIRT-style model text pathway)                        #
    # ------------------------------------------------------------------ #
    def model_profiles(self) -> Optional[pd.DataFrame]:
        return self.profiles

    def profile_text(self, model_id: str, column: str = "profile_text") -> Optional[str]:
        if self.profiles is None:
            return None
        row = self.profiles[self.profiles["model_id"] == model_id]
        return None if row.empty else str(row.iloc[0][column])

    # ------------------------------------------------------------------ #
    # embeddings (query + profile, per encoder pathway)                  #
    # ------------------------------------------------------------------ #
    def _default_pathway(self) -> str:
        from router.embeddings import available_pathways

        return self.cfg.get("embedding.default_pathway") or available_pathways(self.cfg)[0]

    def _load_store(self, kind: str, pathway: Optional[str]):
        from router.embeddings import EmbeddingStore, default_store_dir

        pw = pathway or self._default_pathway()
        d = default_store_dir(self.cfg, kind, pw)
        if not EmbeddingStore.exists(d):
            return None
        return EmbeddingStore.load(d)

    def query_embeddings(self, pathway: Optional[str] = None):
        """`query_id -> vector` :class:`EmbeddingStore` for a pathway, or None."""
        return self._load_store("query", pathway)

    def profile_embeddings(self, pathway: Optional[str] = None):
        """`model_id -> vector` :class:`EmbeddingStore` for a pathway, or None.

        This is the vector used to initialise ``theta_m`` (and to place
        cold-start models from their description alone).
        """
        return self._load_store("model_profile", pathway)

    def profile_embedding(self, model_id: str, pathway: Optional[str] = None):
        store = self.profile_embeddings(pathway)
        if store is None or model_id not in store:
            return None
        return store.get(model_id)

    def query_features(self, name: Optional[str] = None):
        """Structured per-query difficulty features (`training.data.query_features`),
        or ``None`` if not built. Concatenated to the query embedding when
        ``data.query_features`` names a variant in the NIRT config."""
        from .query_features import load_query_features

        return load_query_features(self.cfg, name or "default")

    # ------------------------------------------------------------------ #
    # NIRT training representation                                       #
    # ------------------------------------------------------------------ #
    def nirt_observations(self, **build_kw) -> pd.DataFrame:
        """Lightweight ``(query_id, model_id, target, split, …)`` table -- no
        embeddings. Reads ``nirt_observations.parquet`` if present, else builds."""
        from .nirt import build_nirt_observations, read_nirt_observations

        path = self.cfg.path("processed") / "nirt_observations.parquet"
        if path.exists() and not build_kw:
            return read_nirt_observations(self.cfg)
        return build_nirt_observations(self.cfg, data=self, **build_kw)

    def nirt_dataset(self, split: Optional[str] = "train", pathway: str = "irt",
                     query_pathway: Optional[str] = None,
                     query_features: Optional[str] = None, **kw):
        """:class:`~router.data.nirt.NIRTDataset` for ``split``: yields
        ``(query_embedding, model_embedding, target)`` per observation.

        ``query_pathway`` (default: ``pathway``) swaps only the query store.
        ``query_features`` names a structured-feature variant
        (`training.data.query_features`) concatenated to the query embedding."""
        from .nirt import nirt_dataset_from_config

        return nirt_dataset_from_config(self.cfg, split=split, pathway=pathway,
                                        query_pathway=query_pathway,
                                        query_features=query_features, data=self, **kw)

    # ------------------------------------------------------------------ #
    # summary                                                            #
    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        r = self.responses
        return {
            "observations": len(r),
            "queries": r["query_id"].nunique(),
            "models_total": r["model_id"].nunique(),
            "models_warm": len(self.warm_model_ids()),
            "models_cold_start": len(self._cold),
            "correctness_observations": int(
                r["metric_type"].isin(schemas.MetricType.CORRECTNESS).sum()
            ),
            "pairwise_observations": int(
                r["metric_type"].isin(schemas.MetricType.PAIRWISE).sum()
            ),
            "by_metric": r["metric_type"].value_counts().to_dict(),
            "by_source": r["source"].value_counts().to_dict(),
            "mc_observations": int(r["is_multiple_choice"].fillna(False).sum()),
            "train_queries": len(self.splits["train"]["query_ids"]),
            "validation_queries": len(self.splits["validation"]["query_ids"]),
            "test_queries": len(self.splits["test"]["query_ids"]),
            "profiles_built": self.profiles is not None,
            "query_embeddings": {
                pw: (self.query_embeddings(pw) is not None)
                for pw in _pathways(self.cfg)
            },
            "profile_embeddings": {
                pw: (self.profile_embeddings(pw) is not None)
                for pw in _pathways(self.cfg)
            },
        }


def _pathways(cfg: Config) -> list[str]:
    from router.embeddings import available_pathways

    return available_pathways(cfg)


def load_training_data(cfg: Config) -> TrainingData:
    responses = read_responses(cfg)
    processed = cfg.path("processed")
    queries = pd.read_parquet(processed / "queries.parquet")
    models = pd.read_parquet(processed / "models.parquet")
    splits = load_splits(cfg)
    missing = [k for k, v in splits.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"splits not built ({missing}); run scripts/data/build_splits.py"
        )
    profiles = None
    prof_path = processed / "model_profiles.parquet"
    if prof_path.exists():
        from ..models.profiles import read_model_profiles

        profiles = read_model_profiles(cfg)
    return TrainingData(cfg, responses, queries, models, splits, profiles)
