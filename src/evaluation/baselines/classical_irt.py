"""Classical IRT baseline -- the reference NIRT is measured against.

Classical (multidimensional) 2PL with **no query features**: every query gets a
free latent vector ``theta_q``, every model a discrimination ``a_m`` and
difficulty ``b_m``.

    P(Y_qm = 1) = sigmoid(a_m . theta_q - b_m)

Because ``theta_q`` is a free per-query parameter, this model **cannot score a
query it never saw** -- it is not a router. Two evaluation protocols make the
comparison honest (see docs/nirt_model.md):

* ``fit_classical_irt(..., protocol="cell")`` -- *transductive ceiling*: fit on a
  random ``1 - holdout_frac`` of the (query, model) cells of one split, predict
  the held-out cells. This is the matrix-factorisation ceiling: "how much signal
  is in the response matrix if you had an oracle query representation". For
  routing it still needs a subset of the query's responses at inference time.
* ``protocol="main_effects"`` -- fit ``a_m``, ``b_m`` on the train/validation
  splits other than ``split`` (train+val for the default ``split="test"``),
  then predict the unseen ``split`` queries with ``theta_q = 0`` -> ``sigmoid(-b_m)``.
  A genuine lower bound: no query information at all.

NIRT's inductive, text-only numbers should sit between the two.

This fits parameters and immediately scores held-out cells in one call -- it
never produces a saved, servable artifact, so it lives in ``evaluation``, not
``training``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from router.determinism import seed_everything
from router.nirt.frames import pivot_qm
from training.nirt.metrics import prediction_metrics

_SPLITS = ("train", "validation", "test")


@dataclass
class ClassicalIRTResult:
    protocol: str
    dim: int
    query_ids: list[str]
    model_ids: list[str]
    pred_matrix: np.ndarray               # (Q, M) predicted P(correct), every cell
    true_matrix: np.ndarray               # (Q, M) observed graded target
    heldout_mask: np.ndarray              # (Q, M) bool -- cells scored (not trained on)
    metrics: dict = field(default_factory=dict)          # prediction metrics on held-out cells
    train_metrics: dict = field(default_factory=dict)


def fit_classical_irt(
    data,
    *,
    split: str = "test",
    protocol: str = "cell",
    dim: int = 1,
    holdout_frac: float = 0.15,
    epochs: int = 500,
    lr: float = 0.05,
    weight_decay: float = 1e-3,        # regularises the free per-query theta -- without it
                                       # the matrix factorisation overfits sparse rows badly
    constrain_discrimination: Optional[bool] = None,
    seed: int = 42,
    models: Optional[list[str]] = None,
    query_ids: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> ClassicalIRTResult:
    import torch
    from torch.nn import functional as F

    seed_everything(seed)
    if constrain_discrimination is None:
        constrain_discrimination = dim == 1

    obs = data.nirt_observations()
    if models is not None:
        obs = obs[obs["model_id"].isin(models)]

    if protocol == "cell":
        if query_ids is not None:
            sp = obs[obs["query_id"].isin(set(map(str, query_ids)))].copy()
        else:
            sp = obs[obs["split"] == split].copy()
        true_df = pivot_qm(sp, "target")
        q_ids = list(true_df.index)
        m_ids = list(true_df.columns)
        true = true_df.to_numpy(np.float32)
        obs_mask = ~np.isnan(true)
        rng = np.random.default_rng(seed)
        held = obs_mask & (rng.random(true.shape) < holdout_frac)
        train_mask = obs_mask & ~held
    elif protocol == "main_effects":
        if query_ids is not None:
            raise ValueError("query_ids is only supported with protocol='cell'")
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        # fit on the train/validation splits that are NOT being scored -- otherwise
        # b_m is fit on the very outcomes it's evaluated against
        fit_splits = [s for s in ("train", "validation") if s != split]
        fit_sp = obs[obs["split"].isin(fit_splits)]
        eval_sp = obs[obs["split"] == split]
        if fit_sp.empty:
            raise ValueError(f"no observations in fit splits {fit_splits} for split={split!r}")
        m_ids = sorted(set(fit_sp["model_id"]) | set(eval_sp["model_id"]))
        fit_df = pivot_qm(fit_sp, "target").reindex(columns=m_ids)
        eval_df = pivot_qm(eval_sp, "target").reindex(columns=m_ids)
        assert not set(fit_df.index) & set(eval_df.index), "fit and eval queries overlap"
        q_ids = list(fit_df.index) + list(eval_df.index)
        true = np.vstack([fit_df.to_numpy(np.float32), eval_df.to_numpy(np.float32)])
        obs_mask = ~np.isnan(true)
        held = np.zeros_like(obs_mask)
        held[len(fit_df):] = obs_mask[len(fit_df):]        # score only eval queries
        train_mask = obs_mask & ~held
        true_df = pd.DataFrame(true, index=q_ids, columns=m_ids)
    else:
        raise ValueError(f"protocol must be 'cell' or 'main_effects', got {protocol!r}")

    Q, M = true.shape
    tr_q, tr_m = np.nonzero(train_mask)
    y_tr = torch.from_numpy(true[tr_q, tr_m])
    qi = torch.from_numpy(tr_q.astype(np.int64))
    mi = torch.from_numpy(tr_m.astype(np.int64))

    theta = torch.nn.Embedding(Q, dim)
    a = torch.nn.Embedding(M, dim)
    b = torch.nn.Embedding(M, 1)
    torch.nn.init.normal_(a.weight, std=0.1)
    torch.nn.init.zeros_(b.weight)
    if protocol == "main_effects":
        torch.nn.init.zeros_(theta.weight)               # theta pinned at 0
        theta.weight.requires_grad_(False)
    else:
        torch.nn.init.normal_(theta.weight, std=0.1)

    params = [a.weight, b.weight] + ([theta.weight] if protocol == "cell" else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        th = theta(qi)
        am = a(mi)
        if constrain_discrimination:
            am = F.softplus(am)
        logit = (am * th).sum(-1) - b(mi).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and epoch % 50 == 0:
            print(f"[classical-irt/{protocol}] epoch {epoch:3d} loss {loss.item():.4f}")

    with torch.no_grad():
        aw = F.softplus(a.weight) if constrain_discrimination else a.weight
        pred = torch.sigmoid(theta.weight @ aw.T - b.weight.squeeze(-1)).numpy()

    hq, hm = np.nonzero(held)
    metrics = prediction_metrics(true[hq, hm], pred[hq, hm])
    train_metrics = prediction_metrics(true[tr_q, tr_m], pred[tr_q, tr_m])

    return ClassicalIRTResult(
        protocol=protocol,
        dim=dim,
        query_ids=list(map(str, q_ids)),
        model_ids=list(map(str, m_ids)),
        pred_matrix=pred,
        true_matrix=true,
        heldout_mask=held,
        metrics=metrics,
        train_metrics=train_metrics,
    )
