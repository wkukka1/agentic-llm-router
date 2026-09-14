"""P7f: few-shot cold-start -- warm-start a NEW model's ``(a_m*, b_m*)`` from a
small labeled probe set, regularized toward the zero-shot profile-text prior.

The existing zero-shot cold-start path (:func:`evaluation.nirt.evaluate.cold_start_eval`
-- project a new model's profile embedding straight through the trained
model's own ``a_head``/``b_head``) is a documented negative result on its own:
worse than the trivial ``warm_mean`` baseline. This module treats that
projection as a PRIOR only, not the final answer, and refines it with a
handful of labeled ``(query, correctness)`` observations for the new model --
a MAP fit of ``(a_m*, b_m*)`` against the FROZEN, already-trained ``theta_q``,
shrunk toward the prior by ``ridge``.

Kept in ``training`` (not ``router``, which must never gain a training
dependency, and not ``evaluation``, which is for label-needing analysis, not
parameter fitting) -- this is a fitting operation, same category as
``training.nirt.train.fit``, just for one new model's two parameters instead
of the whole model.
"""

from __future__ import annotations

import numpy as np


def fit_probe(
    theta_q: np.ndarray,
    y: np.ndarray,
    *,
    prior_a: np.ndarray,
    prior_b: float,
    ridge: float = 1.0,
    lr: float = 0.1,
    epochs: int = 300,
) -> tuple[np.ndarray, float]:
    """MAP-fit ``(a_m*, b_m*)`` for one new model from its probe observations.

    ``theta_q`` is ``[k, K]`` -- the FROZEN trained query head's output for the
    probe queries (``model.latent_query(e_q_probe)``, not a training input
    itself, just a value being reused). ``y`` is ``[k]`` observed correctness
    for the new model on those same queries.

    ``prior_a`` (``[K]``) / ``prior_b`` (scalar) are the zero-shot
    profile-projection estimate -- the fit is regularized toward them via a
    Gaussian-prior penalty ``ridge * (||a - prior_a||^2 + (b - prior_b)^2)``:

    * ``k=0`` probe observations -> returns the prior unchanged (no update
      possible; falls straight back to the documented zero-shot behaviour).
    * As ``k`` grows and/or ``ridge`` shrinks, the estimate converges toward
      the pure-probe MLE, independent of the (possibly poor) prior.

    Logistic BCE has no closed-form MAP estimate, so this runs a small Adam
    fit -- cheap: ``K`` is a handful of dimensions and ``epochs`` a few hundred.
    """
    import torch

    prior_a = np.asarray(prior_a, dtype=np.float64)
    if len(y) == 0:
        return prior_a.copy(), float(prior_b)

    theta_t = torch.as_tensor(np.asarray(theta_q, dtype=np.float64))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64))
    pa = torch.as_tensor(prior_a)
    pb = torch.tensor(float(prior_b), dtype=torch.float64)

    a = pa.clone().requires_grad_(True)
    b = pb.clone().requires_grad_(True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logit = theta_t @ a - b
        nll = torch.nn.functional.binary_cross_entropy_with_logits(logit, y_t)
        reg = ridge * ((a - pa).pow(2).sum() + (b - pb).pow(2))
        (nll + reg).backward()
        opt.step()
    return a.detach().numpy(), float(b.detach())
