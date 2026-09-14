"""Shrinkage-to-`model_mean` calibration for queries far from training (item 4
of the capacity workstream, `docs/nirt_capacity.md`).

The P1 diagnostic found every learned NIRT predictor is *worse than the
per-model training rate* (`model_mean`) on IRT-Router's held-out-dataset (OOD)
split (NIRT 0.606, paper K25 0.590, vs `model_mean` 0.547) -- the query-side
head extrapolates confidently and wrongly onto queries unlike anything it
trained on. The fix: blend the raw prediction toward `model_mean` in
proportion to how *novel* the query's embedding is (mean cosine similarity to
its `k` nearest training queries). No parameters are fit to labels and nothing
here touches training, so it cannot leak -- which is also why this lives in
`router`, not `evaluation`: a live router can apply it to a real request, not
just measure it after the fact.

    from router.nirt.shrinkage import novelty_weight, shrink_predictions

    w = novelty_weight(eval_emb, train_emb, k=5)          # (Nq,) in [0, 1]
    blended = shrink_predictions(pred, w, model_mean_rate)  # (Nq, M)

``w`` is applied uniformly across a query's whole model row, so it dampens
miscalibration without destroying the *relative* ranking across models the way
a global row-mean collapse would.
"""

from __future__ import annotations

import numpy as np


def _unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def novelty_weight(
    eval_emb: np.ndarray,
    train_emb: np.ndarray,
    *,
    k: int = 1,
    midpoint: float = 0.35,
    temp: float = 0.08,
) -> np.ndarray:
    """Trust weight ``w(q) in [0, 1]`` per eval-query row.

    ``w`` is a sigmoid of the cosine similarity to the ``k`` nearest *training*
    embeddings (their **max**, i.e. the single closest match when ``k=1``;
    averaging the top-``k`` for ``k>1`` trades a little noise-robustness for
    sensitivity -- a genuinely novel query with even one lucky near-neighbour
    would otherwise look familiar), centred at ``midpoint`` with slope
    ``1/temp``: ``w ~ 1`` (fully trust the raw prediction) when the query
    closely resembles training data, ``w ~ 0`` (fully shrink to `model_mean`)
    when it doesn't. Brute-force cosine (fine up to ~10^5 train queries); swap
    in a FAISS bank for larger pools.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    e = _unit(np.asarray(eval_emb, dtype=np.float64))
    t = _unit(np.asarray(train_emb, dtype=np.float64))
    sims = e @ t.T                                   # (Nq, Ntr)
    kk = min(k, sims.shape[1])
    topk = np.partition(sims, -kk, axis=1)[:, -kk:]
    sim = topk.max(axis=1) if kk == 1 else topk.mean(axis=1)
    return 1.0 / (1.0 + np.exp(-(sim - midpoint) / max(temp, 1e-6)))


def shrink_predictions(
    pred: np.ndarray, weight: np.ndarray, model_mean_rate: np.ndarray
) -> np.ndarray:
    """``w * pred + (1 - w) * model_mean_rate``, broadcast over the model axis.

    ``pred`` is ``(Nq, M)``, ``weight`` is ``(Nq,)`` (clipped to ``[0, 1]``),
    ``model_mean_rate`` is ``(M,)`` (the per-model training correctness rate --
    see ``router.nirt.routing.train_quality`` / ``metrics.marginal_baselines``).
    """
    pred = np.asarray(pred, dtype=np.float64)
    w = np.clip(np.asarray(weight, dtype=np.float64), 0.0, 1.0)[:, None]
    rate = np.asarray(model_mean_rate, dtype=np.float64)[None, :]
    return w * pred + (1 - w) * rate
