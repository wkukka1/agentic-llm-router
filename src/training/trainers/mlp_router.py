"""Fit the IRT-free ``e_q -> R^M`` MLP router.

Inference for an already-fit model (:func:`router.nirt.baselines_infer.mlp_router_matrix`)
and the model constructor (:func:`router.nirt.baselines_infer._build_mlp_router`) live in
``router`` since serving needs them too; only the training loop itself lives here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from router.determinism import seed_everything
from router.nirt.baselines_infer import _build_mlp_router, _train_correctness_matrix


def fit_mlp_router(
    data,
    *,
    hidden: int = 128,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    pathway: str = "irt",
    seed: int = 42,
    train_obs: Optional[pd.DataFrame] = None,
    verbose: bool = False,
):
    """Direct ``e_q -> R^M`` per-model correctness head (masked BCE). The IRT-free ablation."""
    import torch
    from torch.nn import functional as F

    seed_everything(seed)
    C, model_ids, tr_qids = _train_correctness_matrix(data, pathway=pathway, train_obs=train_obs)
    obs_mask = ~np.isnan(C)
    y = np.where(obs_mask, C, 0.0).astype(np.float32)
    m = obs_mask.astype(np.float32)
    X = np.asarray(data.query_embeddings(pathway).gather(list(tr_qids)), dtype=np.float32)

    model = _build_mlp_router(X.shape[1], len(model_ids), hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, yt, mt = torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(m)
    n, bs = len(Xt), 4096
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            logit = model(Xt[b])
            loss = (F.binary_cross_entropy_with_logits(logit, yt[b], reduction="none") * mt[b]).sum() / mt[b].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose:
            print(f"[mlp-router] epoch {epoch} loss {loss.item():.4f}")
    model.eval()
    return model, model_ids


def fit_mlp_router_as_router(data=None, *, pathway: str = "irt", name: Optional[str] = None, **fit_kw):
    """Fit and wrap the result as a :class:`router.routing.routers.MLPRouter`.

    Replaces the old ``MLPRouter.train(...)`` classmethod -- a serving class
    must not call a trainer, so the fit step moved here.
    """
    from router.routing.routers import MLPRouter, _load_training_data

    d = _load_training_data(data)
    model, model_ids = fit_mlp_router(d, pathway=pathway, **fit_kw)
    return MLPRouter(model, model_ids, data=d, pathway=pathway, name=name)
