"""NIRT training representation.

Two layers, kept separate so 768-d vectors are never copied into every row:

1. **observation table** (``data/processed/nirt_observations.parquet``) -- one
   lightweight row per ``(query_id, model_id, metric)`` training example:
   ``query_id, model_id, target, score_raw, metric_type, source, split, cost,
   is_multiple_choice, n_choices``. No embeddings.

2. **:class:`NIRTDataset`** -- joins the observation table to the query and
   profile :class:`~router.embeddings.EmbeddingStore`s *by id at access time*.
   ``dataset[i]`` returns ``{query_embedding, model_embedding, target, ...}``;
   the embeddings are memmap views, not copies.

Only the **absolute correctness** signal (RouterBench / lm-harness) goes here --
that is the primary NIRT response signal. Pairwise preference (Arena / Judge)
has its own opponent-aware path via ``Phase1Data.pairwise`` and is not folded in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from . import schemas
from .phase1 import Phase1Data, load_phase1

NIRT_OBS_COLUMNS = [
    "query_id", "model_id", "target", "score_raw", "metric_type",
    "source", "split", "cost", "is_multiple_choice", "n_choices",
]

try:  # optional torch base class
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _TorchDataset = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# observation table                                                           #
# --------------------------------------------------------------------------- #
def build_nirt_observations(
    cfg: Config,
    *,
    data: Optional[Phase1Data] = None,
    score_kind: str = "effective",
    metrics: Optional[Iterable[str]] = None,
    models: str = "warm",
    drop_unsplit: bool = True,
) -> pd.DataFrame:
    """Lightweight ``(query, model, target)`` table for every correctness obs.

    ``target`` follows ``score_kind`` (``effective`` = chance-corrected where
    applicable, else raw). ``split`` is attached from ``data/splits``; rows whose
    query is in no split (only reachable via cold-start models) are dropped when
    ``drop_unsplit``.
    """
    d = data or load_phase1(cfg)
    long = d.correctness(split=None, metrics=metrics, models=models,
                         score_kind=score_kind)

    split_of: dict[str, str] = {}
    for name in ("train", "validation", "test", "ood"):
        entry = d.splits.get(name)
        if entry is None:
            continue
        for qid in entry["query_ids"]:
            split_of[qid] = name

    out = pd.DataFrame({
        "query_id": long["query_id"],
        "model_id": long["model_id"],
        "target": pd.to_numeric(long["score"], errors="coerce"),
        "score_raw": pd.to_numeric(long["score_raw"], errors="coerce"),
        "metric_type": long["metric_type"],
        "is_multiple_choice": long["is_multiple_choice"].astype(bool),
        "n_choices": long["n_choices"].astype("Int64"),
        "split": long["query_id"].map(split_of).astype("string"),
    })

    # cost + source live on the raw response rows, not the correctness view
    corr_rows = d.responses[d.responses["metric_type"].isin(schemas.MetricType.CORRECTNESS)]
    extra = (
        corr_rows.groupby(["query_id", "model_id"], as_index=False)
        .agg(cost=("cost", "mean"), source=("source", "first"))
    )
    out = out.merge(extra, on=["query_id", "model_id"], how="left")

    if drop_unsplit:
        out = out[out["split"].notna()]
    out = out[NIRT_OBS_COLUMNS].reset_index(drop=True)
    return out


def write_nirt_observations(df: pd.DataFrame, cfg: Config) -> Path:
    out = cfg.path("processed") / "nirt_observations.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def read_nirt_observations(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("processed") / "nirt_observations.parquet")


# --------------------------------------------------------------------------- #
# dataset                                                                     #
# --------------------------------------------------------------------------- #
class NIRTDataset(_TorchDataset):
    """Joins the observation table to embedding stores by id (no vector copies).

    ``dataset[i]`` -> dict with ``query_embedding`` (float32, ``q_dim``),
    ``model_embedding`` (float32, ``m_dim``), ``target`` (float32) plus
    ``metric``, ``source``, ``cost`` and, if ``return_ids``, ``query_id`` /
    ``model_id``.
    """

    def __init__(
        self,
        observations: pd.DataFrame,
        query_store,
        profile_store,
        *,
        return_ids: bool = False,
    ):
        self.query_store = query_store
        self.profile_store = profile_store
        self.return_ids = return_ids

        mask = (
            observations["query_id"].isin(query_store._index)
            & observations["model_id"].isin(profile_store._index)
            & observations["target"].notna()
        )
        self.obs = observations.loc[mask].reset_index(drop=True)
        self.dropped = int((~mask).sum())

        self._q_rows = query_store.rows_of(self.obs["query_id"].tolist())
        self._m_rows = profile_store.rows_of(self.obs["model_id"].tolist())
        self._y = self.obs["target"].to_numpy(np.float32)
        self._cost = pd.to_numeric(self.obs["cost"], errors="coerce").to_numpy(np.float32)
        self._metric = self.obs["metric_type"].astype(str).to_numpy()
        self._source = self.obs["source"].astype(str).to_numpy()
        self._qmat = query_store.matrix
        self._mmat = profile_store.matrix

    # -- shape --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.obs)

    @property
    def query_dim(self) -> int:
        return int(self._qmat.shape[1])

    @property
    def model_dim(self) -> int:
        return int(self._mmat.shape[1])

    @property
    def targets(self) -> np.ndarray:
        return self._y

    @property
    def query_ids(self) -> np.ndarray:
        return self.obs["query_id"].to_numpy()

    @property
    def model_ids(self) -> np.ndarray:
        return self.obs["model_id"].to_numpy()

    # -- access ------------------------------------------------------
    def __getitem__(self, i: int) -> dict:
        item = {
            "query_embedding": np.asarray(self._qmat[self._q_rows[i]], dtype=np.float32),
            "model_embedding": np.asarray(self._mmat[self._m_rows[i]], dtype=np.float32),
            "target": np.float32(self._y[i]),
            "metric": self._metric[i],
            "source": self._source[i],
            "cost": np.float32(self._cost[i]),
        }
        if self.return_ids:
            item["query_id"] = self.obs["query_id"].iat[i]
            item["model_id"] = self.obs["model_id"].iat[i]
        return item

    def gather(self, index: Optional[Sequence[int]] = None) -> dict:
        """Materialize ``(Q, M, y)`` for the whole dataset (or a subset).

        Copies the gathered embedding rows -- for `n` observations this is
        ``n * (q_dim + m_dim) * 4`` bytes, so prefer iterating / DataLoader for
        the full training set.
        """
        qr, mr = self._q_rows, self._m_rows
        y, cost, metric, source = self._y, self._cost, self._metric, self._source
        if index is not None:
            index = np.asarray(index)
            qr, mr, y, cost, metric, source = (
                qr[index], mr[index], y[index], cost[index], metric[index], source[index]
            )
        return {
            "query_embedding": np.asarray(self._qmat[qr], dtype=np.float32),
            "model_embedding": np.asarray(self._mmat[mr], dtype=np.float32),
            "target": y,
            "cost": cost,
            "metric": metric,
            "source": source,
        }

    # -- torch helpers ---------------------------------------------
    @staticmethod
    def collate(batch: list[dict]) -> dict:
        import torch

        out = {
            "query_embedding": torch.as_tensor(
                np.stack([b["query_embedding"] for b in batch])
            ),
            "model_embedding": torch.as_tensor(
                np.stack([b["model_embedding"] for b in batch])
            ),
            "target": torch.as_tensor(np.array([b["target"] for b in batch], np.float32)),
            "cost": torch.as_tensor(np.array([b["cost"] for b in batch], np.float32)),
            "metric": [b["metric"] for b in batch],
            "source": [b["source"] for b in batch],
        }
        if batch and "query_id" in batch[0]:
            out["query_id"] = [b["query_id"] for b in batch]
            out["model_id"] = [b["model_id"] for b in batch]
        return out

    def dataloader(self, batch_size: int = 256, shuffle: bool = True, **kw):
        from torch.utils.data import DataLoader

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self.collate, **kw)

    # -- construction --------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        split: Optional[str] = None,
        pathway: str = "irt",
        query_pathway: Optional[str] = None,
        data: Optional[Phase1Data] = None,
        observations: Optional[pd.DataFrame] = None,
        return_ids: bool = False,
        **build_kw,
    ) -> "NIRTDataset":
        """``query_pathway`` (default: ``pathway``) overrides only the *query*
        embedding store -- used to feed a kNN-imputed query representation while
        the profile store stays on ``pathway``."""
        d = data or load_phase1(cfg)
        if observations is None:
            path = cfg.path("processed") / "nirt_observations.parquet"
            observations = (
                pd.read_parquet(path) if path.exists()
                else build_nirt_observations(cfg, data=d, **build_kw)
            )
        if split is not None:
            observations = observations[observations["split"] == split]

        q_pathway = query_pathway or pathway
        q_store = d.query_embeddings(q_pathway)
        m_store = d.profile_embeddings(pathway)
        if q_store is None:
            raise FileNotFoundError(
                f"query embeddings for pathway '{q_pathway}' not built; run "
                f"scripts/embeddings/build_query_embeddings.py --pathway {q_pathway}"
            )
        if m_store is None:
            raise FileNotFoundError(
                f"profile embeddings for pathway '{pathway}' not built; run "
                f"scripts/embeddings/build_profile_embeddings.py --pathway {pathway}"
            )
        return cls(observations, q_store, m_store, return_ids=return_ids)
