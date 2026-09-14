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
) -> PairwiseArrays:
    """Materialise battles joinable to the query/profile embedding stores.

    ``source`` (``None`` -> every pairwise source, e.g. ``"chatbot_arena"`` /
    ``"gpt4_judge"``) and ``split`` select the battles; rows whose query or
    either model lack an embedding are dropped (``n_dropped``)."""
    df = data.pairwise(source=source, split=split, models="all")
    q_store = data.query_embeddings(query_pathway or pathway)
    m_store = data.profile_embeddings(pathway)
    if q_store is None or m_store is None:
        raise FileNotFoundError(f"embeddings for pathway '{pathway}' not built")

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
            np.zeros((0, q_store.dim), np.float32), np.zeros((0, m_store.dim), np.float32),
            np.zeros((0, m_store.dim), np.float32), np.zeros(0, np.float32), dropped,
        )
    e_q = np.ascontiguousarray(q_store.gather(df["query_id"].tolist()), dtype=np.float32)
    e_a = np.ascontiguousarray(m_store.gather(df["model_a"].tolist()), dtype=np.float32)
    e_b = np.ascontiguousarray(m_store.gather(df["model_b"].tolist()), dtype=np.float32)
    y = df["score_a"].to_numpy(np.float32)
    return PairwiseArrays(e_q, e_a, e_b, y, dropped)


def pairwise_logit_diff(model, e_q, e_a, e_b):
    """``z(q, model_a) - z(q, model_b)`` using the model's own query head +
    model-side heads -- the exact score ``NIRTModel.forward`` uses (bilinear
    base plus the interaction residual when ``model.interaction`` is set), just
    evaluated for two models against one shared ``theta_q``. Requires
    ``model.model_params == "projected"``."""
    if model.model_params != "projected":
        raise ValueError(
            "pairwise training needs model_params='projected' -- a 'free' model "
            "has no row for battle participants absent from the correctness pool"
        )
    theta = model.latent_query(e_q)
    a_a, b_a = model.model_parameters(e_a)
    a_b, b_b = model.model_parameters(e_b)
    if model.difficulty == "vector":
        z_a = (a_a * (theta - b_a)).sum(-1)
        z_b = (a_b * (theta - b_b)).sum(-1)
    else:
        z_a = (a_a * theta).sum(-1) - b_a
        z_b = (a_b * theta).sum(-1) - b_b
    if model.interaction:
        feats_a = torch.cat([theta, a_a, theta * a_a], dim=-1)
        feats_b = torch.cat([theta, a_b, theta * a_b], dim=-1)
        z_a = z_a + model.interaction_gamma * model.interaction_net(feats_a).squeeze(-1)
        z_b = z_b + model.interaction_gamma * model.interaction_net(feats_b).squeeze(-1)
    return z_a - z_b


def pairwise_loss(model, e_q, e_a, e_b, y):
    """Mean ``BCE(sigmoid(z_a - z_b), y)`` -- the Bradley-Terry auxiliary loss."""
    import torch.nn.functional as F

    diff = pairwise_logit_diff(model, e_q, e_a, e_b)
    return F.binary_cross_entropy_with_logits(diff, y)
