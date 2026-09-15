"""Beta response head (Phase 2B).

Mean / concentration parameterisation:

    mu_qm    = sigmoid(z_qm)                          in (0, 1)   -> E[y] = mu
    kappa_qm = softplus( f_kappa(e_q, theta_m) + k0 ) + eps       concentration
    alpha    = mu * kappa ,   beta = (1 - mu) * kappa

``mu`` is clamped to ``[mu_min, 1-mu_min]`` and ``kappa`` to
``[min_concentration, max_concentration]`` (all config); ``kappa`` is then raised
to at least ``min_concentration / min(mu, 1-mu)`` so ``alpha, beta >=
min_concentration`` exactly -- no NaN/inf can enter training, and the density's
mean is still ``mu``.

The Beta density has no mass at exact 0 or 1, so the Beta model is trained and
evaluated on **interior** observations only (``0 < y < 1``); the boundary
fraction is reported separately (see ``response_boundary_statistics.json``). We
do NOT jitter exact 0/1 to fake interior support.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from router.nirt.components import mlp

from .response_head import ResponseHead, ResponseOutput


class BetaResponseHead(ResponseHead):
    key = "beta"
    target = "soft"
    interior_only = True

    def __init__(self, *, feature_dim: int, hidden: Optional[int] = 64, cfg: Optional[dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.min_conc = float(cfg.get("min_concentration", 1e-4))
        self.max_conc = float(cfg.get("max_concentration", 1e4))
        self.mu_min = float(cfg.get("mu_min", 1e-4))
        self.eps = float(cfg.get("epsilon", 1e-6))
        self.f_kappa = mlp(feature_dim, 1, hidden)
        self.log_kappa0 = nn.Parameter(torch.zeros(()))

    def _params(self, z, features):
        mu = torch.sigmoid(z).clamp(self.mu_min, 1 - self.mu_min)
        raw = self.log_kappa0 if features is None else self.f_kappa(features).squeeze(-1) + self.log_kappa0
        kappa = (torch.nn.functional.softplus(raw) + self.eps).clamp(self.min_conc, self.max_conc)
        kappa = kappa.expand_as(mu)
        # floor alpha = mu*kappa and beta = (1-mu)*kappa at min_concentration through
        # kappa itself, so the density's mean stays mu and gradients keep flowing
        return mu, torch.maximum(kappa, self.min_conc / torch.minimum(mu, 1 - mu))

    def forward(self, z, features=None):
        mu, kappa = self._params(z, features)
        return ResponseOutput(z=z, params={"mu": mu, "kappa": kappa,
                                           "alpha": mu * kappa, "beta": (1 - mu) * kappa})

    def _dist(self, out):
        # alpha, beta >= min_concentration by construction (see _params)
        return torch.distributions.Beta(out["alpha"], out["beta"])

    def nll(self, y, out):
        y = y.to(out["mu"].dtype).clamp(1e-6, 1 - 1e-6)
        return -self._dist(out).log_prob(y)

    def mean(self, out):
        return out["mu"]

    def variance(self, out):
        return out["mu"] * (1 - out["mu"]) / (out["kappa"] + 1.0)

    def as_probability(self, out):
        return out["mu"]

    def _sample(self, out, n: int = 512):
        return self._dist(out).sample((n,))
