"""Building blocks for :class:`router.nirt.baseline.BaselineNIRT`.

    e_q --warm-up--> e_q'
    a_q = softplus(f_a(e_q')) * sigmoid(W_r r_q)      >= 0   discrimination
    b_q = f_b(e_q')                                          difficulty (larger = harder)
    logit = a_q . theta_m - b_q  [+ gamma * interaction(...)]

Each optional term is initialised so that *enabling* it starts from the bare IRT
score (relevance gate ~ 1, interaction gamma = 0).
"""

from __future__ import annotations

import math
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


class CostHead(nn.Module):
    """Expected-execution-cost probe: ``feats -> (cost_mu, cost_log_sigma)``, a
    Gaussian over ``log USD``. Estimates ``E[C | q, m]`` ex ante (output length is
    unknown before generation), via the lognormal mean ``exp(mu + sigma^2 / 2)``.
    The caller detaches ``feats``, so this head never shapes the quality trunk."""

    LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 3.0
    _HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)

    def __init__(self, in_dim: int, hidden: Optional[int], *, init_mu: float):
        super().__init__()
        self.net = mlp(in_dim, 2, hidden)
        # input standardisation (identity until set_standardization): the IRT
        # latent is only defined up to scale, so the probe should not depend on it
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        last = self.net[-1] if isinstance(self.net, nn.Sequential) else self.net
        with torch.no_grad():   # mu starts at exactly init_mu for every input
            last.weight[0].zero_()
            last.bias[0] = float(init_mu)

    @torch.no_grad()
    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Fix the input mean / std (computed from TRAIN-split features)."""
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-6))

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net((feats - self.feat_mean) / self.feat_std)
        return out[:, 0], out[:, 1].clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX)

    def nll(self, log_c: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Per-row Gaussian NLL of ``log_c`` (constant included, so it is a true NLL)."""
        return (log_sigma + 0.5 * (log_c - mu) ** 2 * torch.exp(-2 * log_sigma)
                + self._HALF_LOG_2PI)

    def train_loss(self, log_c: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
                   beta: float = 0.5) -> torch.Tensor:
        """Per-row beta-NLL (Seitzer et al. 2022): the NLL weighted by a detached
        ``sigma^(2 beta)``. Plain NLL lets a large learned sigma starve the mean
        of gradient; ``beta=0`` recovers it. Report :meth:`nll`, train on this."""
        return self.nll(log_c, mu, log_sigma) * torch.exp(2 * beta * log_sigma).detach()

    @staticmethod
    def expected_cost(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        return torch.exp(mu + 0.5 * torch.exp(2 * log_sigma))


class OutputLengthHead(nn.Module):
    """Output-length classifier over fixed token-count buckets:
    ``logits = log_prior_m + f(feats)``. ``log_prior_m`` is the model's train-split
    bucket distribution (passed per row); ``f``'s last layer starts at zero, so the
    head begins as exactly that per-model prior and learns only the query-specific
    correction. ``E[L_out] = softmax(logits) . bucket_values`` -- and because cost
    is linear in tokens, ``p_in * L_in + p_out * E[L_out]`` is the EXACT expected
    cost under the predicted distribution. ``feats`` are detached by the caller."""

    def __init__(self, in_dim: int, hidden: Optional[int], bucket_values):
        super().__init__()
        values = torch.as_tensor(bucket_values, dtype=torch.float32)
        self.n_buckets = int(values.numel())
        self.net = mlp(in_dim, self.n_buckets, hidden)
        self.register_buffer("bucket_values", values)
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        last = self.net[-1] if isinstance(self.net, nn.Sequential) else self.net
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()

    @torch.no_grad()
    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-6))

    def forward(self, feats: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
        return log_prior.to(feats.dtype) + self.net((feats - self.feat_mean) / self.feat_std)

    def nll(self, labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Per-row cross-entropy of the observed bucket."""
        return F.cross_entropy(logits, labels.long(), reduction="none")

    def expected_length(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1) @ self.bucket_values

