"""In-house router baselines -- inference only.

KNN = what RouterBench ships; MLP = IRT-structure ablation ("NIRT minus the
bilinear form"). Fitting (``fit_mlp_router``) lives in
``training.trainers.mlp_router`` -- serving code only ever scores with an
already-fit reference table or model, it never trains one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .frames import pivot_qm


def train_correctness_matrix(data, *, pathway: str, train_obs: Optional[pd.DataFrame] = None):
    """``(C [Qtr x M] with NaN for missing, model_ids, query_ids)`` for the train split
    (or ``train_obs`` if given), restricted to queries present in the pathway store."""
    if train_obs is None:
        obs = data.nirt_observations()
        train_obs = obs[obs["split"] == "train"]
    piv = pivot_qm(train_obs, "target")
    q_store = data.query_embeddings(pathway)
    if q_store is None:
        raise FileNotFoundError(f"query embeddings for pathway '{pathway}' not built")
    qids = [q for q in piv.index if q in q_store._index]
    piv = piv.loc[qids]
    return piv.to_numpy(np.float64), list(piv.columns), np.asarray(qids)


def _fit_knn(data, *, k: int, pathway: str, train_obs: Optional[pd.DataFrame] = None):
    """Fit a cosine-kNN index over the train-split query embeddings, plus the
    column-mean-imputed train correctness matrix to average over. Shared by
    :func:`knn_router_matrix` (query by id) and
    :func:`knn_router_matrix_from_embeddings` (query by a raw embedding batch,
    e.g. a live encoded prompt)."""
    from sklearn.neighbors import NearestNeighbors

    C, model_ids, tr_qids = train_correctness_matrix(data, pathway=pathway, train_obs=train_obs)
    C_imp = np.where(np.isnan(C), np.nanmean(C, axis=0), C)
    q_store = data.query_embeddings(pathway)
    E_tr = np.asarray(q_store.gather(list(tr_qids)), dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=min(k, len(tr_qids)), metric="cosine").fit(E_tr)
    return nn, C_imp, model_ids


def knn_router_matrix(
    data,
    eval_query_ids,
    *,
    k: int = 5,
    pathway: str = "retrieval",
    train_obs: Optional[pd.DataFrame] = None,
    fitted: Optional[tuple] = None,
) -> pd.DataFrame:
    """RouterBench-style KNN router: predict a query's per-model correctness as the
    mean over its ``k`` nearest training queries (cosine, retrieval embeddings).

    Pass ``fitted`` (a ``(nn, C_imp, model_ids)`` tuple from :func:`_fit_knn`) to
    reuse an index already fit elsewhere instead of refitting it here -- e.g. a
    long-lived :class:`~router.routing.routers.KNNRouter` fits once and reuses it
    across calls."""
    nn, C_imp, model_ids = fitted if fitted is not None else _fit_knn(
        data, k=k, pathway=pathway, train_obs=train_obs
    )
    q_store = data.query_embeddings(pathway)
    ids = [q for q in map(str, eval_query_ids) if q in q_store._index]
    E_ev = np.asarray(q_store.gather(ids), dtype=np.float64)
    _, idx = nn.kneighbors(E_ev)
    pred = C_imp[idx].mean(axis=1)
    return pd.DataFrame(pred, index=ids, columns=model_ids)


def knn_router_matrix_from_embeddings(
    data,
    e_q: np.ndarray,
    *,
    k: int = 5,
    pathway: str = "retrieval",
    train_obs: Optional[pd.DataFrame] = None,
    model_ids: Optional[list] = None,
    fitted: Optional[tuple] = None,
) -> pd.DataFrame:
    """:func:`knn_router_matrix`, but for a raw ``[N, d]`` embedding batch (e.g.
    an already-encoded live prompt) instead of pre-built query ids. Rows are
    positional (``0..N-1``); pass ``model_ids`` to reindex the columns.

    See :func:`knn_router_matrix` for ``fitted``."""
    nn, C_imp, fitted_model_ids = fitted if fitted is not None else _fit_knn(
        data, k=k, pathway=pathway, train_obs=train_obs
    )
    _, idx = nn.kneighbors(np.asarray(e_q, dtype=np.float64))
    pred = C_imp[idx].mean(axis=1)
    df = pd.DataFrame(pred, index=list(range(len(e_q))), columns=fitted_model_ids)
    return df.reindex(columns=model_ids) if model_ids is not None else df


def build_mlp_router(in_dim: int, n_models: int, hidden: int, dropout: float):
    from torch import nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_models)
    )


def mlp_query_matrix(data, query_ids, *, pathway: str = "irt",
                     query_pathway: Optional[str] = None,
                     query_features: Optional[str] = None) -> tuple[np.ndarray, list]:
    """``(X [N x D], ids)`` -- the MLP router's input rows, built the same way
    :class:`~router.data.nirt.NIRTDataset` builds NIRT's: the ``query_pathway``
    store (default ``pathway``) optionally concatenated with the
    ``query_features`` variant (zero-filled for queries missing from it). Ids
    absent from the query store are dropped."""
    q_pw = query_pathway or pathway
    q_store = data.query_embeddings(q_pw)
    if q_store is None:
        raise FileNotFoundError(f"query embeddings for pathway '{q_pw}' not built")
    ids = [q for q in map(str, query_ids) if q in q_store._index]
    X = np.asarray(q_store.gather(ids), dtype=np.float32)
    if query_features:
        feat_store = data.query_features(query_features)
        if feat_store is None:
            raise FileNotFoundError(f"query features '{query_features}' not built")
        F = np.zeros((len(ids), feat_store.dim), dtype=np.float32)
        rows = np.array([feat_store.row_of(q) if q in feat_store else -1 for q in ids], dtype=np.int64)
        hit = rows >= 0
        F[hit] = np.asarray(feat_store.matrix)[rows[hit]]
        X = np.concatenate([X, F], axis=1)
    return X, ids


def mlp_router_matrix(model, model_ids, data, eval_query_ids, *, pathway: str = "irt",
                      query_pathway: Optional[str] = None,
                      query_features: Optional[str] = None) -> pd.DataFrame:
    import torch

    X_np, ids = mlp_query_matrix(data, eval_query_ids, pathway=pathway,
                                 query_pathway=query_pathway, query_features=query_features)
    first = next((m for m in model.modules() if isinstance(m, torch.nn.Linear)), None)
    if first is not None and X_np.shape[1] != first.in_features:
        raise ValueError(
            f"MLP router expects {first.in_features}-d inputs but pathway={pathway!r} "
            f"(query_pathway={query_pathway!r}, query_features={query_features!r}) gives "
            f"{X_np.shape[1]}-d -- was it fit on a different pathway?"
        )
    X = torch.from_numpy(X_np)
    with torch.no_grad():
        p = torch.sigmoid(model(X)).numpy()
    return pd.DataFrame(p, index=ids, columns=model_ids)
