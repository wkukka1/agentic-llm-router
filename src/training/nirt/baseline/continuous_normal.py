"""Heteroskedastic Normal response head (Phase 2A).

    y_qm ~ Normal(mu_qm, sigma_qm)
    mu_qm    = z_qm                                   (the Phase 1 latent score, unchanged)
    sigma_qm = softplus( f_sigma(e_q, theta_m) + s0 ) + epsilon

``sigma`` is input-dependent (heteroskedastic), not one global constant.
``epsilon`` (config) keeps it strictly positive. The Normal has support outside
[0, 1]; the likelihood uses the true Normal (no clipping). For metrics that need
a probability we report ``P(Y >= 0.5) = Phi((mu - 0.5) / sigma)`` and, as the
expected score, ``clamp(mu, 0, 1)`` -- both documented, neither feeds the loss.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from router.nirt.components import mlp

from .response_head import ResponseHead, ResponseOutput

_SQRT2 = math.sqrt(2.0)


class NormalResponseHead(ResponseHead):
    key = "normal"
    target = "soft"

    def __init__(self, *, feature_dim: int, hidden: Optional[int] = 64, cfg: Optional[dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.eps = float(cfg.get("epsilon", 1e-6))
        self.f_sigma = mlp(feature_dim, 1, hidden)
        self.log_sigma0 = nn.Parameter(torch.zeros(()))

    def forward(self, z, features=None):
        raw = self.log_sigma0 if features is None else self.f_sigma(features).squeeze(-1) + self.log_sigma0
        sigma = torch.nn.functional.softplus(raw) + self.eps
        return ResponseOutput(z=z, params={"mu": z, "sigma": sigma.expand_as(z)})

    def _dist(self, out: ResponseOutput):
        return torch.distributions.Normal(out["mu"], out["sigma"])

    def nll(self, y, out):
        return -self._dist(out).log_prob(y.to(out["mu"].dtype))

    def mean(self, out):
        return out["mu"]

    def variance(self, out):
        return out["sigma"] ** 2

    def as_probability(self, out):
        # P(Y >= 0.5) under the predictive Normal
        return 0.5 * torch.erfc((0.5 - out["mu"]) / (out["sigma"] * _SQRT2))

    def expected_score(self, out):
        return out["mu"].clamp(0.0, 1.0)

    # analytic intervals (exact, no sampling)
    def interval(self, out, level: float = 0.9):
        d = self._dist(out)
        return d.icdf(torch.full_like(out["mu"], (1 - level) / 2)), \
            d.icdf(torch.full_like(out["mu"], (1 + level) / 2))

    def lower_bound(self, out, level: float = 0.9):
        return self._dist(out).icdf(torch.full_like(out["mu"], 1 - level))

    def _sample(self, out, n: int = 512):
        return self._dist(out).sample((n,))
