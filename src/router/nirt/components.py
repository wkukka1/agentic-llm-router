"""Building blocks for :class:`router.nirt.baseline.BaselineNIRT`.

    e_q --warm-up--> e_q'
    a_q = softplus(f_a(e_q')) * sigmoid(W_r r_q)      >= 0   discrimination
    b_q = f_b(e_q')                                          difficulty (larger = harder)
    logit = a_q . theta_m - b_q  [+ gamma * interaction(...)]

Each optional term is initialised so that *enabling* it starts from the bare IRT
score (relevance gate ~ 1, interaction gamma = 0).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def mlp(in_dim: int, out_dim: int, hidden: Optional[int]) -> nn.Module:
    """Linear, or a 1-hidden-layer ReLU MLP when ``hidden`` is truthy."""
    if hidden:
        return nn.Sequential(nn.Linear(in_dim, int(hidden)), nn.ReLU(),
                             nn.Linear(int(hidden), out_dim))
    return nn.Linear(in_dim, out_dim)


class WarmupBlender(nn.Module):
    """``e_q' = (1-alpha) e_q + alpha * neighbour_mean``. Pass-through when disabled,
    when no neighbour mean is given, or for a row whose neighbour mean is all-zero
    (no neighbours found)."""

    def __init__(self, enabled: bool = False, alpha: float = 0.5, learnable: bool = False):
        super().__init__()
        self.enabled = bool(enabled)
        self.learnable = bool(learnable)
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"warm-up alpha must be in [0, 1], got {alpha}")
        if learnable:
            a = min(max(float(alpha), 1e-3), 1 - 1e-3)   # logit needs the open interval
            self._alpha_logit = nn.Parameter(torch.logit(torch.tensor(a)))
        else:
            self.register_buffer("_alpha", torch.tensor(float(alpha)))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_logit) if self.learnable else self._alpha

    def forward(self, e_q: torch.Tensor, neighbor_mean: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.enabled or neighbor_mean is None:
            return e_q
        a = self.alpha.to(e_q.dtype)
        has = (neighbor_mean.abs().sum(-1, keepdim=True) > 0).to(e_q.dtype)
        return has * ((1 - a) * e_q + a * neighbor_mean) + (1 - has) * e_q


class DiscriminationHead(nn.Module):
    """``a_q = softplus(f_a(e_q)) * gate`` with ``gate = sigmoid(W_r r_q)`` in ``R^K``
    (the relevance mask; ~1 at init). ``W_r`` maps ``R^C -> R^K``."""

    def __init__(self, query_dim, dim, hidden: Optional[int] = 64, relevance_dim: int = 0,
                 use_relevance: bool = False, constrain: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.use_relevance = bool(use_relevance and relevance_dim > 0)
        self.constrain = bool(constrain)
        self.f_a = mlp(query_dim, dim, hidden)
        if self.use_relevance:
            self.w_r = nn.Linear(relevance_dim, dim)
            nn.init.zeros_(self.w_r.weight)
            nn.init.constant_(self.w_r.bias, 2.0)   # sigmoid(2) ~ 0.88 -> near pass-through

    def forward(self, e_q, r_q: Optional[torch.Tensor] = None):
        a = self.f_a(e_q)
        a = F.softplus(a) if self.constrain else a
        gate = None
        if self.use_relevance and r_q is not None:
            gate = torch.sigmoid(self.w_r(r_q))
            a = a * gate
        return a, gate


class DifficultyHead(nn.Module):
    """``b_q`` scalar (larger = harder; it is subtracted in the response)."""

    def __init__(self, query_dim, hidden: Optional[int] = 64):
        super().__init__()
        self.f_b = mlp(query_dim, 1, hidden)

    def forward(self, e_q) -> torch.Tensor:
        return self.f_b(e_q).squeeze(-1)


class InteractionLayer(nn.Module):
    """``logit = base_irt + gamma * MLP([a_q, theta_m, a_q*theta_m, r_q])``, ``gamma``
    init 0. Always returns ``(logit, base_irt)`` so diagnostics see the pure IRT term."""

    def __init__(self, dim, relevance_dim: int = 0, hidden: int = 32, enabled: bool = False):
        super().__init__()
        self.enabled = bool(enabled)
        self.relevance_dim = int(relevance_dim)
        self.net = mlp(3 * dim + self.relevance_dim, 1, hidden)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, a_q, theta_m, b_q, r_q: Optional[torch.Tensor] = None):
        base = (a_q * theta_m).sum(-1) - b_q
        if not self.enabled:
            return base, base
        feats = [a_q, theta_m, a_q * theta_m]
        if self.relevance_dim and r_q is not None:
            feats.append(r_q)
        return base + self.gamma * self.net(torch.cat(feats, dim=-1)).squeeze(-1), base


class LengthHead(nn.Module):
    """Independent ``f_len(e_q) -> scalar``. Part of the control architecture Phase 3
    couples; NOT connected to the correctness logit and its loss is disabled (Phase 0
    has no length labels) -- present only so the interface is stable."""

    def __init__(self, query_dim, hidden: Optional[int] = 64, enabled: bool = False):
        super().__init__()
        self.enabled = bool(enabled)
        self.net = mlp(query_dim, 1, hidden)

    def forward(self, e_q) -> torch.Tensor:
        return self.net(e_q).squeeze(-1)
