"""Auxiliary pairwise (Bradley-Terry) signal for ``NIRTModel`` (item 2 of the
capacity workstream, `docs/nirt_capacity.md`).

RouterBench's ~29k-query correctness signal is data-starved -- the model
overfits within a handful of epochs (see the P1/overfitting findings). Chatbot
Arena ships ~115k human pairwise battles over a much larger, mostly different
model pool, on the SAME frozen query encoder space (`TrainingData.pairwise`,
excluded from ``nirt_observations`` per the Phase 0 design: pairwise preference
is never folded into the absolute-correctness matrix ``R[q, m]``).

This module reuses the model's own bilinear score
``z(q, m) = a_m . theta_q(e_q) - b_m`` -- ``theta_q`` is the SAME query head the
correctness task trains -- as a Bradley-Terry strength, and trains
``BCE(sigmoid(z_a - z_b), y_ab)`` as an auxiliary objective. This pulls
*additional, independent supervision* onto the shared query representation; it
does not add a second model or change the correctness forward pass. Enable with
``train.preference.enabled: true`` in ``configs/nirt.yaml`` -- default off, so
existing runs are untouched.

Only ``model_params="projected"`` can score Arena models absent from the
correctness training set (a ``free`` per-model embedding has no row for them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class PairwiseArrays:
    e_q: np.ndarray       # (N, query_dim)
    e_a: np.ndarray       # (N, profile_dim)
    e_b: np.ndarray       # (N, profile_dim)
    y: np.ndarray         # (N,) in {0, 0.5, 1} -- outcome for model_a
    n_dropped: int

    def __len__(self) -> int:
        return len(self.y)


def build_pairwise_arrays(
    data,
    *,
    source: Optional[str] = None,
    split: str = "train",
    pathway: str = "irt",
    query_pathway: Optional[str] = None,
    query_features: Optional[str] = None,
) -> PairwiseArrays:
    """Materialise battles joinable to the query/profile embedding stores.

    ``source`` (``None`` -> every pairwise source, e.g. ``"chatbot_arena"`` /
    ``"gpt4_judge"``) and ``split`` select the battles; rows whose query or
    either model lack an embedding are dropped (``n_dropped``).

    ``query_features`` (the NIRT run's ``data.query_features``) concatenates the
    same structured features the correctness dataset feeds the shared query
    head, zero-filled for battle queries without a feature row -- exactly like
    :class:`~router.data.nirt.NIRTDataset`."""
    df = data.pairwise(source=source, split=split, models="all")
    q_store = data.query_embeddings(query_pathway or pathway)
    m_store = data.profile_embeddings(pathway)
    if q_store is None or m_store is None:
        raise FileNotFoundError(f"embeddings for pathway '{pathway}' not built")
    feat_store = None
    if query_features:
        feat_store = data.query_features(query_features)
        if feat_store is None:
            raise FileNotFoundError(f"query features '{query_features}' not built")
    q_dim = q_store.dim + (feat_store.dim if feat_store is not None else 0)

    keep = (
        df["query_id"].isin(q_store._index)
        & df["model_a"].isin(m_store._index)
        & df["model_b"].isin(m_store._index)
        & df["score_a"].notna()
    )
    dropped = int((~keep).sum())
    df = df.loc[keep].reset_index(drop=True)
    if df.empty:
        return PairwiseArrays(
            np.zeros((0, q_dim), np.float32), np.zeros((0, m_store.dim), np.float32),
            np.zeros((0, m_store.dim), np.float32), np.zeros(0, np.float32), dropped,
        )
    e_q = np.ascontiguousarray(q_store.gather(df["query_id"].tolist()), dtype=np.float32)
    if feat_store is not None:
        qids = df["query_id"].tolist()
        rows = np.array([feat_store.row_of(q) if q in feat_store else -1 for q in qids], dtype=np.int64)
        feats = np.zeros((len(qids), feat_store.dim), dtype=np.float32)
        hit = rows >= 0
        feats[hit] = np.asarray(feat_store.matrix)[rows[hit]]
        e_q = np.ascontiguousarray(np.concatenate([e_q, feats], axis=1), dtype=np.float32)
    e_a = np.ascontiguousarray(m_store.gather(df["model_a"].tolist()), dtype=np.float32)
    e_b = np.ascontiguousarray(m_store.gather(df["model_b"].tolist()), dtype=np.float32)
    y = df["score_a"].to_numpy(np.float32)
    return PairwiseArrays(e_q, e_a, e_b, y, dropped)


def pairwise_logit_diff(model, e_q, e_a, e_b):
    """``z(q, model_a) - z(q, model_b)`` using the model's own query head +
    model-side heads, scored via :meth:`NIRTModel.score` -- the same formula
    ``forward`` uses, just evaluated for two models against one shared
    ``theta_q``. Requires ``model.model_params == "projected"``.

    ``a_a``/``a_b`` come from two *separate* forward calls, so a
    ``center_discrimination`` model must not let each one centre on its own
    batch -- that would subtract different reference points from ``a_a`` and
    ``a_b`` and bias ``z_a - z_b`` by ``-(r_a - r_b) . theta``. Both calls
    share one reference (the mean discrimination over this pair-batch's own
    ``e_a``/``e_b`` pool) via :func:`router.nirt.predict.pool_centering` --
    a no-op when centring is disabled."""
    if model.model_params != "projected":
        raise ValueError(
            "pairwise training needs model_params='projected' -- a 'free' model "
            "has no row for battle participants absent from the correctness pool"
        )
    from router.nirt.predict import pool_centering

    theta = model.latent_query(e_q)
    pool_ref = torch.cat([e_a, e_b], dim=0).detach().cpu().numpy()
    with pool_centering(model, pool_ref):
        a_a, b_a = model.model_parameters(e_a)
        a_b, b_b = model.model_parameters(e_b)
    z_a = model.score(theta, a_a, b_a)
    z_b = model.score(theta, a_b, b_b)
    return z_a - z_b


def pairwise_loss(model, e_q, e_a, e_b, y):
    """Mean ``BCE(sigmoid(z_a - z_b), y)`` -- the Bradley-Terry auxiliary loss."""
    import torch.nn.functional as F

    diff = pairwise_logit_diff(model, e_q, e_a, e_b)
    return F.binary_cross_entropy_with_logits(diff, y)
