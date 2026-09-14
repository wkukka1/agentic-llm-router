"""P7f: leave-one-model-out (LOMO) evaluation of the few-shot cold-start pathway.

For each model already present in a trained pool, pretend it is a brand-new
model: sample ``k`` of its own observations as a probe set, warm-start
``(a_m*, b_m*)`` from a caller-supplied zero-shot prior via
:func:`training.nirt.coldstart.fit_probe`, and score the result on its
REMAINING (held-out) observations. Sweeping ``k`` gives a learning curve --
the comparison points are the ``warm_mean`` floor (the documented zero-shot
cold-start negative result) and the fully warm ``model_params=free`` estimate
(the ceiling this approaches as ``k`` grows).

Pure numpy / pandas, evaluation-only (may import ``training``, must never be
imported by it or by ``router`` -- see ``docs/architecture.md``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from training.nirt.coldstart import fit_probe
from training.nirt.metrics import prediction_metrics


def lomo_eval(
    theta_q: dict,
    obs: pd.DataFrame,
    priors: dict,
    *,
    k_values: Sequence[int] = (0, 10, 25, 50, 100),
    n_repeats: int = 3,
    ridge: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """``theta_q``: ``query_id -> theta_q`` (the FROZEN trained query head's
    output, precomputed once -- see :func:`training.nirt.coldstart.fit_probe`
    for why this is not itself a training input). ``obs``: long table with
    ``query_id, model_id, target`` for the model pool being evaluated.
    ``priors``: ``model_id -> (prior_a, prior_b)`` -- typically the zero-shot
    profile-projection estimate for that model.

    For each ``model_id`` with a prior, and each ``k``, draws ``n_repeats``
    random ``k``-observation probe sets (``k=0`` is deterministic, 1 rep),
    fits, and scores on the rest. Returns a long DataFrame: ``model_id, k,
    repeat, n_probe, n_eval`` plus every :func:`training.nirt.metrics.prediction_metrics`
    column (``bce``, ``auc``, ...).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mid, g in obs.groupby("model_id"):
        if mid not in priors:
            continue
        prior_a, prior_b = priors[mid]
        qids = g["query_id"].to_numpy()
        y = g["target"].to_numpy(dtype=np.float64)
        n = len(g)
        idx_all = np.arange(n)
        for k in k_values:
            reps = n_repeats if k > 0 else 1
            for rep in range(reps):
                if k == 0:
                    probe_idx = np.array([], dtype=int)
                else:
                    probe_idx = rng.choice(idx_all, size=min(int(k), n), replace=False)
                eval_idx = np.setdiff1d(idx_all, probe_idx)
                if len(eval_idx) == 0:
                    continue
                probe_theta = (np.stack([theta_q[q] for q in qids[probe_idx]])
                              if len(probe_idx) else np.zeros((0, len(prior_a))))
                a_star, b_star = fit_probe(
                    probe_theta, y[probe_idx], prior_a=prior_a, prior_b=prior_b, ridge=ridge)
                eval_theta = np.stack([theta_q[q] for q in qids[eval_idx]])
                p = 1.0 / (1.0 + np.exp(-(eval_theta @ a_star - b_star)))
                m = prediction_metrics(y[eval_idx], p)
                rows.append({"model_id": mid, "k": int(k), "repeat": int(rep),
                            "n_probe": int(len(probe_idx)), "n_eval": int(len(eval_idx)), **m})
    return pd.DataFrame(rows)
