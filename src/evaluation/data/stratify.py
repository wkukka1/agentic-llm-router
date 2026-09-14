"""Stratified query sampling for the anchor-judge overlap set (docs/anchor_judge.md).

Queries where every model in the pool is correct, or none is, carry zero
routing signal -- the oracle is unanimous and disagreement can't be measured.
``sample_stratified_queries`` splits the correctness matrix into a
"discriminative" stratum (correctness varies across the pool) and a
"natural" stratum (an unweighted sample of everything else), and returns them
as two disjoint, separately-labelled samples rather than one reweighted
blend -- the discriminative rate is the scientific number, the natural rate
is the deployment-relevant one, and keeping them apart avoids an
inverse-probability-weighting estimator that's easy to get subtly wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TOL = 1e-9


def sample_stratified_queries(
    R: pd.DataFrame,
    *,
    n_discriminative: int,
    n_natural: int,
    seed: int,
    tol: float = _TOL,
) -> pd.DataFrame:
    """Sample query_ids from a dense correctness matrix into two strata.

    ``R``: rows=query_id, columns=model_id, values in [0, 1], no NaN (caller
    filters to the pool / split it cares about first).

    Returns a frame indexed by ``query_id`` with columns ``stratum``
    (``"discriminative"`` or ``"natural"``), ``row_mean_accuracy``,
    ``row_std``. The two strata are disjoint -- a query drawn into
    ``natural`` is excluded from the discriminative draw pool.
    """
    if R.isna().any().any():
        raise ValueError("R must not contain NaN -- filter to a complete pool/split first")

    rng = np.random.default_rng(seed)
    row_mean = R.mean(axis=1)
    row_std = R.std(axis=1)

    all_ids = R.index.to_numpy()
    n_natural = min(n_natural, len(all_ids))
    natural_ids = rng.choice(all_ids, size=n_natural, replace=False) if n_natural else np.array([])

    discriminative_mask = (row_mean > tol) & (row_mean < 1 - tol)
    discriminative_ids = R.index[discriminative_mask].to_numpy()
    remaining_disc = np.setdiff1d(discriminative_ids, natural_ids, assume_unique=False)
    n_discriminative = min(n_discriminative, len(remaining_disc))
    disc_ids = (
        rng.choice(remaining_disc, size=n_discriminative, replace=False)
        if n_discriminative else np.array([])
    )

    rows = []
    for ids, stratum in ((natural_ids, "natural"), (disc_ids, "discriminative")):
        for qid in ids:
            rows.append({
                "query_id": qid,
                "stratum": stratum,
                "row_mean_accuracy": float(row_mean[qid]),
                "row_std": float(row_std[qid]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out.set_index(pd.Index([], name="query_id"))
    return out.set_index("query_id")
