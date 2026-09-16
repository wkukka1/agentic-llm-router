"""Score a trained NIRT / IRT-Router model -- no ground truth involved.

Split out of ``evaluation.nirt.evaluate`` (which owns the label-needing
``evaluate_split`` / ``cold_start_eval``) so that live routers --
:class:`router.routing.routers.NIRTRouter` -- can get a prediction matrix
without importing anything evaluation-only.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd

from .frames import pivot_qm


@contextmanager
def pool_centering(model, pool_profile_embeddings: Optional[np.ndarray]):
    """Within the block, a ``center_discrimination`` + ``projected`` model
    centres on the mean over ``pool_profile_embeddings`` (``(M, profile_dim)``)
    instead of the current forward batch, so scores don't depend on how rows
    are chunked. A no-op for every other model; the previous reference is
    restored on exit."""
    uses_ref = (
        pool_profile_embeddings is not None
        and getattr(model, "center_discrimination", False)
        and getattr(model, "model_params", None) == "projected"
        and hasattr(model, "set_discrimination_reference")
    )
    if not uses_ref:
        yield
        return
    prev = model._a_ref
    model.set_discrimination_reference(pool_profile_embeddings)
    try:
        yield
    finally:
        model._a_ref = prev


def predict_dataset(model, dataset, model_index: dict, batch_size: int = 8192):
    """Return ``(y_true, y_prob)`` for every observation in ``dataset``.

    Rows are gathered chunk by chunk (never the full ``n_obs x (D_q + D_m)``
    copy). Projected-mode centring uses the dataset's whole model pool as its
    reference, so predictions don't depend on ``batch_size`` or row order."""
    import torch

    n = len(dataset)
    y = np.ascontiguousarray(dataset.targets, dtype=np.float64)
    model_ids = np.asarray(dataset.model_ids)

    projected = model.model_params == "projected"
    pool_ref = None
    if projected:
        _, first = np.unique(model_ids.astype(str), return_index=True)
        if len(first):
            pool_ref = dataset.gather(np.sort(first))["model_embedding"]
    else:
        midx = np.array([model_index.get(str(m), -1) for m in model_ids], dtype=np.int64)
        if (midx < 0).any():
            missing = sorted({str(m) for m, i in zip(model_ids, midx) if i < 0})
            raise ValueError(
                f"model_params='free' cannot score models absent from training: {missing}. "
                f"Use a 'projected' run for cold-start / unseen models."
            )

    out = np.empty(n, dtype=np.float64)
    model.eval()
    with torch.no_grad(), pool_centering(model, pool_ref):
        for i in range(0, n, batch_size):
            idx = np.arange(i, min(i + batch_size, n))
            g = dataset.gather(idx)
            q = torch.from_numpy(np.ascontiguousarray(g["query_embedding"], dtype=np.float32))
            if projected:
                ref = torch.from_numpy(np.ascontiguousarray(g["model_embedding"], dtype=np.float32))
            else:
                ref = torch.from_numpy(midx[idx])
            out[idx] = torch.sigmoid(model(q, ref)).numpy()
    return y, out


def predict_matrix_from_dataset(model, model_index: dict, dataset) -> pd.DataFrame:
    """Predicted ``P(correct)`` as a dense [query x model] DataFrame for an
    already-built :class:`NIRTDataset` (any obs frame -- named split or OOD)."""
    _, prob = predict_dataset(model, dataset, model_index)
    frame = pd.DataFrame(
        {"query_id": dataset.query_ids, "model_id": dataset.model_ids, "pred": prob}
    )
    return pivot_qm(frame, "pred")


def predict_matrix(model, model_index: dict, data, split: str, pathway: str = "irt",
                   query_pathway: Optional[str] = None,
                   query_features: Optional[str] = None) -> pd.DataFrame:
    """Predicted ``P(correct)`` as a dense [query x model] DataFrame for ``split``.

    ``query_pathway`` (default: ``pathway``) swaps only the query store -- pass
    the run's ``data.query_pathway`` when scoring a kNN-imputed run.
    ``query_features`` re-attaches the run's structured feature variant
    (``data.query_features``), if any."""
    return predict_matrix_from_dataset(
        model, model_index,
        data.nirt_dataset(split=split, pathway=pathway, query_pathway=query_pathway,
                          query_features=query_features),
    )
