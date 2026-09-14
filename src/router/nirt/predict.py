"""Score a trained NIRT / IRT-Router model -- no ground truth involved.

Split out of ``evaluation.nirt.evaluate`` (which owns the label-needing
``evaluate_split`` / ``cold_start_eval``) so that live routers --
:class:`router.routing.routers.NIRTRouter` -- can get a prediction matrix
without importing anything evaluation-only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .frames import pivot_qm


def predict_dataset(model, dataset, model_index: dict, batch_size: int = 8192):
    """Return ``(y_true, y_prob)`` for every observation in ``dataset``."""
    import torch

    g = dataset.gather()
    q = torch.from_numpy(np.ascontiguousarray(g["query_embedding"], dtype=np.float32))
    y = np.ascontiguousarray(g["target"], dtype=np.float64)

    projected = model.model_params == "projected"
    if projected:
        ref_src = torch.from_numpy(np.ascontiguousarray(g["model_embedding"], dtype=np.float32))
    else:
        midx = np.array(
            [model_index.get(str(m), -1) for m in dataset.model_ids], dtype=np.int64
        )
        if (midx < 0).any():
            missing = sorted({str(m) for m, i in zip(dataset.model_ids, midx) if i < 0})
            raise ValueError(
                f"model_params='free' cannot score models absent from training: {missing}. "
                f"Use a 'projected' run for cold-start / unseen models."
            )
        ref_src = torch.from_numpy(midx)

    out = np.empty(len(y), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(out), batch_size):
            sl = slice(i, i + batch_size)
            out[sl] = torch.sigmoid(model(q[sl], ref_src[sl])).numpy()
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
