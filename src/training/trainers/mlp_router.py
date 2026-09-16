"""Fit the IRT-free ``e_q -> R^M`` MLP router.

Inference for an already-fit model (:func:`router.nirt.baselines_infer.mlp_router_matrix`)
and the model constructor (:func:`router.nirt.baselines_infer.build_mlp_router`) live in
``router`` since serving needs them too; only the training loop itself lives here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from router.determinism import seed_everything
from router.nirt.baselines_infer import (
    build_mlp_router,
    mlp_query_matrix,
    train_correctness_matrix,
)


def _masked_matrix(data, obs: pd.DataFrame, *, pathway, query_pathway, query_features):
    """``(X, y, mask, model_ids)`` for one split's observations."""
    C, model_ids, qids = train_correctness_matrix(
        data, pathway=query_pathway or pathway, train_obs=obs
    )
    obs_mask = ~np.isnan(C)
    y = np.where(obs_mask, C, 0.0).astype(np.float32)
    X, _ = mlp_query_matrix(data, qids, pathway=pathway, query_pathway=query_pathway,
                            query_features=query_features)
    return X, y, obs_mask.astype(np.float32), model_ids


def fit_mlp_router(
    data,
    *,
    hidden: int = 128,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    batch_size: int = 4096,
    patience: Optional[int] = None,
    pathway: str = "irt",
    query_pathway: Optional[str] = None,
    query_features: Optional[str] = None,
    seed: int = 42,
    train_obs: Optional[pd.DataFrame] = None,
    val_obs: Optional[pd.DataFrame] = None,
    verbose: bool = False,
):
    """Direct ``e_q -> R^M`` per-model correctness head (masked BCE). The IRT-free ablation.

    ``query_pathway`` / ``query_features`` mirror the NIRT ``data.*`` keys so a
    head-to-head sees identical inputs. When ``patience`` is set, the validation
    split (``val_obs``, default the ``validation`` rows of
    ``data.nirt_observations()``) drives early stopping on masked BCE, like
    :func:`training.nirt.train.fit`, and the best-epoch weights are returned.
    """
    import torch
    from torch.nn import functional as F

    seed_everything(seed)
    if train_obs is None or (patience is not None and val_obs is None):
        obs = data.nirt_observations()
        train_obs = obs[obs["split"] == "train"] if train_obs is None else train_obs
        if patience is not None and val_obs is None:
            val_obs = obs[obs["split"] == "validation"]
    kw = dict(pathway=pathway, query_pathway=query_pathway, query_features=query_features)
    X, y, m, model_ids = _masked_matrix(data, train_obs, **kw)
    if len(X) == 0:
        raise ValueError("fit_mlp_router: no train queries with embeddings")

    val = None
    if patience is not None:
        Xv, yv, mv, v_ids = _masked_matrix(data, val_obs, **kw)
        # align validation columns to the train model order (unseen models dropped)
        col = {mid: j for j, mid in enumerate(v_ids)}
        yv_al = np.zeros((len(Xv), len(model_ids)), np.float32)
        mv_al = np.zeros_like(yv_al)
        for j, mid in enumerate(model_ids):
            if mid in col:
                yv_al[:, j], mv_al[:, j] = yv[:, col[mid]], mv[:, col[mid]]
        if len(Xv) and mv_al.sum() > 0:
            val = tuple(torch.from_numpy(a) for a in (Xv, yv_al, mv_al))

    def masked_bce(logit, yt, mt):
        return (F.binary_cross_entropy_with_logits(logit, yt, reduction="none") * mt).sum() \
            / mt.sum().clamp(min=1)

    model = build_mlp_router(X.shape[1], len(model_ids), hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, yt, mt = torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(m)
    n, bs = len(Xt), int(batch_size)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, cnt = 0.0, 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = masked_bce(model(Xt[b]), yt[b], mt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            w = float(mt[b].sum())
            tot, cnt = tot + loss.item() * w, cnt + w
        msg = f"[mlp-router] epoch {epoch} loss {tot / max(cnt, 1.0):.4f}"
        if val is not None:
            model.eval()
            with torch.no_grad():
                vl = float(masked_bce(model(val[0]), val[1], val[2]))
            msg += f" val_bce {vl:.4f}"
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        if verbose:
            print(msg)
        if val is not None and bad >= int(patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, model_ids


def fit_mlp_router_as_router(data=None, *, pathway: str = "irt", name: Optional[str] = None, **fit_kw):
    """Fit and wrap the result as a :class:`router.routing.routers.MLPRouter`.

    Replaces the old ``MLPRouter.train(...)`` classmethod -- a serving class
    must not call a trainer, so the fit step moved here.
    """
    from router.routing.routers import MLPRouter, load_training_data_for_router

    d = load_training_data_for_router(data)
    model, model_ids = fit_mlp_router(d, pathway=pathway, **fit_kw)
    return MLPRouter(model, model_ids, data=d, pathway=pathway, name=name)
