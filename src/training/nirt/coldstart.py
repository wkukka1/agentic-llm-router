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
    fixed-strength Gaussian prior: the objective is the SUMMED probe NLL plus
    ``ridge * (||a - prior_a||^2 + (b - prior_b)^2)``:

    * ``k=0`` probe observations -> returns the prior unchanged (no update
      possible; falls straight back to the documented zero-shot behaviour).
    * As ``k`` grows (the data term grows with ``k``, the prior does not)
      and/or ``ridge`` shrinks, the estimate converges toward the pure-probe
      MLE, independent of the (possibly poor) prior.

    Score function: the plain scalar-difficulty bilinear form
    ``logit = theta_q . a - b``. ``prior_a`` must therefore be the *effective*
    discrimination the model uses -- ``model.model_parameters(...)`` output,
    i.e. post-softplus / post-centring, not raw ``a_head`` activations -- and
    the fitted ``(a*, b*)`` are meant to be scored with that same plain form
    (as :func:`evaluation.nirt.coldstart_eval.lomo_eval` does). Models with
    ``difficulty: vector`` or an interaction residual are not supported.

    Logistic BCE has no closed-form MAP estimate, so this runs a small Adam
    fit -- cheap: ``K`` is a handful of dimensions and ``epochs`` a few hundred.
    """
    import torch

    prior_a = np.asarray(prior_a, dtype=np.float64)
    if prior_a.ndim != 1:
        raise ValueError(f"prior_a must be a [K] vector (scalar difficulty only), got {prior_a.shape}")
    if np.ndim(prior_b) != 0:
        raise ValueError("prior_b must be a scalar -- fit_probe supports difficulty='scalar' only")
    if len(y) == 0:
        return prior_a.copy(), float(prior_b)
    if np.asarray(theta_q).shape[1] != prior_a.shape[0]:
        raise ValueError(f"theta_q has K={np.asarray(theta_q).shape[1]} but prior_a has "
                         f"K={prior_a.shape[0]}")

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
        # summed (not mean) NLL: the prior keeps a fixed strength as k grows
        nll = torch.nn.functional.binary_cross_entropy_with_logits(logit, y_t, reduction="sum")
        reg = ridge * ((a - pa).pow(2).sum() + (b - pb).pow(2))
        (nll + reg).backward()
        opt.step()
    return a.detach().numpy(), float(b.detach())
