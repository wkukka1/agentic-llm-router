"""Zero-One-Inflated Beta response head (Phase 2C).

    P(Y = 0)      = pi_0
    P(Y = 1)      = pi_1
    P(0 < Y < 1)  = pi_c ,   Y | (0<Y<1) ~ Beta(alpha, beta)

    (pi_0, pi_1, pi_c) = softmax( f_pi(e_q, theta_m) )        (guaranteed simplex)
    mu = sigmoid(z_qm) ,   kappa = softplus(.) + eps  (clamped)
    alpha = mu * kappa ,   beta = (1 - mu) * kappa

Log-likelihood (stable, log-space):

    y == 0        -> log pi_0
    y == 1        -> log pi_1
    0 < y < 1     -> log pi_c + log Beta(y; alpha, beta)

**pi_0 / pi_1 are literal probability mass at the boundaries -- NOT a 3PL
guessing parameter.** A 3PL lower asymptote is an item-response mechanism on
P(correct); ZOIB's boundary masses are just the empirical spikes of the graded
score at 0 and 1, with a continuous Beta over the middle.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from router.nirt.components import mlp

from .response_head import ResponseHead, ResponseOutput


class ZOIBResponseHead(ResponseHead):
    key = "zoib"
    target = "soft"

    def __init__(self, *, feature_dim: int, hidden: Optional[int] = 64, cfg: Optional[dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.min_conc = float(cfg.get("min_concentration", 1e-4))
        self.max_conc = float(cfg.get("max_concentration", 1e4))
        self.mu_min = float(cfg.get("mu_min", 1e-4))
        self.eps = float(cfg.get("epsilon", 1e-6))
        self.f_pi = mlp(feature_dim, 3, hidden)
        self.pi_bias = nn.Parameter(torch.zeros(3))
        self.f_kappa = mlp(feature_dim, 1, hidden)
        self.log_kappa0 = nn.Parameter(torch.zeros(()))

    def forward(self, z, features=None):
        if features is None:
            log_pi = F.log_softmax(self.pi_bias, dim=-1).expand(z.shape[0], 3)
            raw_k = self.log_kappa0.expand(z.shape[0])
        else:
            log_pi = F.log_softmax(self.f_pi(features) + self.pi_bias, dim=-1)
            raw_k = self.f_kappa(features).squeeze(-1) + self.log_kappa0
        mu = torch.sigmoid(z).clamp(self.mu_min, 1 - self.mu_min)
        kappa = (F.softplus(raw_k) + self.eps).clamp(self.min_conc, self.max_conc)
        pi = log_pi.exp()
        return ResponseOutput(z=z, params={
            "mu": mu, "kappa": kappa, "alpha": mu * kappa, "beta": (1 - mu) * kappa,
            "log_pi0": log_pi[:, 0], "log_pi1": log_pi[:, 1], "log_pic": log_pi[:, 2],
            "pi0": pi[:, 0], "pi1": pi[:, 1], "pic": pi[:, 2],
        })

    def _beta(self, out):
        return torch.distributions.Beta(out["alpha"].clamp_min(self.min_conc),
                                        out["beta"].clamp_min(self.min_conc))

    def nll(self, y, out):
        y = y.to(out["mu"].dtype)
        is0, is1 = y <= 0.0, y >= 1.0
        beta_lp = self._beta(out).log_prob(y.clamp(1e-6, 1 - 1e-6))
        interior = out["log_pic"] + beta_lp
        ll = torch.where(is0, out["log_pi0"], torch.where(is1, out["log_pi1"], interior))
        return -ll

    def mean(self, out):
        return out["pi1"] + out["pic"] * out["mu"]

    def variance(self, out):
        var_beta = out["mu"] * (1 - out["mu"]) / (out["kappa"] + 1.0)
        e_y2 = out["pi1"] + out["pic"] * (var_beta + out["mu"] ** 2)
        return (e_y2 - self.mean(out) ** 2).clamp_min(0.0)

    def as_probability(self, out):
        return self.mean(out).clamp(0.0, 1.0)

    def boundary_probs(self, out):
        return out["pi0"], out["pi1"]

    def _sample(self, out, n: int = 512):
        probs = torch.stack([out["pi0"], out["pi1"], out["pic"]], dim=-1)     # (B, 3)
        comp = torch.distributions.Categorical(probs=probs).sample((n,))      # (n, B)
        beta_s = self._beta(out).sample((n,))
        return torch.where(comp == 0, torch.zeros_like(beta_s),
                           torch.where(comp == 1, torch.ones_like(beta_s), beta_s))
