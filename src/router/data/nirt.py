"""``NIRTDataset`` -- the serving-side NIRT batch container.

Joins an observation table (one lightweight row per ``(query_id, model_id,
metric)``: ``target, score_raw, metric_type, source, split, cost,
is_multiple_choice, n_choices`` -- no embeddings) to the query and profile
:class:`~router.embeddings.EmbeddingStore`\\ s *by id at access time*.
``dataset[i]`` returns ``{query_embedding, model_embedding, target, ...}``;
the embeddings are memmap views, not copies.

Building the observation table (``data/processed/nirt_observations.parquet``)
from raw training data, and the ``Config -> TrainingData`` resolution
:meth:`NIRTDataset.from_config` needs, are training-only concerns -- see
:mod:`training.data.nirt` (:func:`~training.data.nirt.build_nirt_observations`,
:func:`~training.data.nirt.nirt_dataset_from_config`). A live router
constructs this class directly from a pre-built store and an in-memory obs
frame; it never builds the observation table itself.

Only the **absolute correctness** signal (RouterBench / lm-harness) goes here --
that is the primary NIRT response signal. Pairwise preference (Arena / Judge)
has its own opponent-aware path via ``TrainingData.pairwise`` and is not folded in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config

if TYPE_CHECKING:  # pragma: no cover - type hints only, not a runtime import
    from training.data.facade import TrainingData

try:  # optional torch base class
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _TorchDataset = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False


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
        feature_store=None,
        return_ids: bool = False,
    ):
        self.query_store = query_store
        self.profile_store = profile_store
        self.feature_store = feature_store
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
        # structured per-query features (router.data.query_features), concatenated
        # to the query embedding -- a query missing from the store (shouldn't
        # happen for a store built over the full corpus) falls back to zeros,
        # like the relevance / warm-up joins in nirt/baseline/data.py.
        self._feat = None
        if feature_store is not None:
            qids = self.obs["query_id"].tolist()
            rows = np.array([feature_store.row_of(q) if q in feature_store else -1 for q in qids])
            self._feat = np.zeros((len(qids), feature_store.dim), dtype=np.float32)
            hit = rows >= 0
            self._feat[hit] = np.asarray(feature_store.matrix)[rows[hit]]

    # -- shape --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.obs)

    @property
    def query_dim(self) -> int:
        d = int(self._qmat.shape[1])
        if self._feat is not None:
            d += self._feat.shape[1]
        return d

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
        qv = np.asarray(self._qmat[self._q_rows[i]], dtype=np.float32)
        if self._feat is not None:
            qv = np.concatenate([qv, self._feat[i]])
        item = {
            "query_embedding": qv,
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
        feat = self._feat
        if index is not None:
            index = np.asarray(index)
            qr, mr, y, cost, metric, source = (
                qr[index], mr[index], y[index], cost[index], metric[index], source[index]
            )
            if feat is not None:
                feat = feat[index]
        qe = np.asarray(self._qmat[qr], dtype=np.float32)
        if feat is not None:
            qe = np.concatenate([qe, feat], axis=1)
        return {
            "query_embedding": qe,
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
        data: "TrainingData",
        split: Optional[str] = None,
        pathway: str = "irt",
        query_pathway: Optional[str] = None,
        query_features: Optional[str] = None,
        observations: Optional[pd.DataFrame] = None,
        return_ids: bool = False,
    ) -> "NIRTDataset":
        """``query_pathway`` (default: ``pathway``) overrides only the *query*
        embedding store -- used to feed a kNN-imputed query representation while
        the profile store stays on ``pathway``. ``query_features`` names a
        structured-feature variant (``training.data.query_features``) concatenated
        to the query embedding; ``None`` -> unchanged (no concat).

        Serving-safe: needs ``data`` (anything exposing ``query_embeddings`` /
        ``profile_embeddings`` / ``query_features``, e.g. ``TrainingData``) passed
        in explicitly, and either ``observations`` or an already-built
        ``nirt_observations.parquet`` -- it never resolves ``data`` from a bare
        config or builds the observation table itself. For that convenience
        (used by the training pipeline), see
        :func:`training.data.nirt.nirt_dataset_from_config`. Live routers use
        ``NIRTDataset(...)`` directly with a pre-built store and an in-memory
        obs frame."""
        if observations is None:
            path = cfg.path("processed") / "nirt_observations.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not built; run scripts/data/build_nirt_dataset.py "
                    f"or training.data.nirt.nirt_dataset_from_config(cfg, ...)"
                )
            observations = pd.read_parquet(path)
        if split is not None:
            observations = observations[observations["split"] == split]

        q_pathway = query_pathway or pathway
        q_store = data.query_embeddings(q_pathway)
        m_store = data.profile_embeddings(pathway)
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
        feat_store = None
        if query_features:
            feat_store = data.query_features(query_features)
            if feat_store is None:
                raise FileNotFoundError(
                    f"query features '{query_features}' not built; run "
                    f"scripts/embeddings/build_query_features.py --name {query_features}"
                )
        return cls(observations, q_store, m_store, feature_store=feat_store, return_ids=return_ids)
