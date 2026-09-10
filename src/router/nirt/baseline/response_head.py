"""Response-head interface (Phase 1 Bernoulli + Phase 2 continuous).

The NIRT core produces one latent correctness location per (query, model):

    z_qm = a_q . theta_m - b_q  [+ gamma * interaction]     (the Phase 1 logit)

A :class:`ResponseHead` turns ``z`` (plus a per-example feature vector
``features = concat(e_q, theta_m)`` for the scale / concentration nets) into a
predictive distribution over the graded score ``y in [0, 1]``. The core does not
know which distribution is used.

Common API (all heads):

    out  = head(z, features)          -> ResponseOutput
    loss = head.loss(y, out)          -> scalar   (mean NLL; the training objective)
    head.nll(y, out)                  -> (B,)     per-sample NLL
    head.mean(out) / head.variance(out) / head.stddev(out)
    head.interval(out, level)         -> (lower, upper)   central `level` interval
    head.lower_bound(out, level)      -> (B,)     one-sided lower quantile
    head.as_probability(out)          -> (B,)     E[Y] clamped to [0, 1] (BCE diagnostic)
    head.param_summary(out)           -> {name: (B,) tensor}   extra params for diagnostics
    head.boundary_probs(out)          -> (pi0, pi1) or None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from router.config import require_choice


@dataclass
class ResponseOutput:
    z: torch.Tensor                                   # latent correctness location (B,)
    params: dict = field(default_factory=dict)        # head-specific tensors

    def __getitem__(self, k):
        return self.params[k]

    def get(self, k, default=None):
        return self.params.get(k, default)


class ResponseHead(nn.Module):
    """Base class. Subclasses set ``key`` / ``target`` and implement ``forward`` +
    ``nll``; the rest have sensible defaults."""

    key: str = "base"
    target: str = "soft"          # which target column the trainer feeds: {soft, binary}

    # -- distribution ------------------------------------------------------
    def forward(self, z: torch.Tensor, features: Optional[torch.Tensor] = None) -> ResponseOutput:
        raise NotImplementedError

    def nll(self, y: torch.Tensor, out: ResponseOutput) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, y: torch.Tensor, out: ResponseOutput) -> torch.Tensor:
        return self.nll(y, out).mean()

    # -- point / spread --------------------------------------------------
    def mean(self, out: ResponseOutput) -> torch.Tensor:
        raise NotImplementedError

    def variance(self, out: ResponseOutput) -> torch.Tensor:
        raise NotImplementedError

    def stddev(self, out: ResponseOutput) -> torch.Tensor:
        return self.variance(out).clamp_min(0).sqrt()

    def as_probability(self, out: ResponseOutput) -> torch.Tensor:
        return self.mean(out).clamp(0.0, 1.0)

    # -- intervals ------------------------------------------------------
    def _sample(self, out: ResponseOutput, n: int = 512) -> torch.Tensor:
        raise NotImplementedError

    def quantiles(self, out: ResponseOutput, qs, n: int = 512) -> torch.Tensor:
        s = self._sample(out, n)                       # (n, B)
        q = torch.as_tensor(qs, dtype=s.dtype, device=s.device)
        return torch.quantile(s, q, dim=0)            # (len(qs), B)

    def interval(self, out: ResponseOutput, level: float = 0.9):
        lo, hi = (1 - level) / 2, (1 + level) / 2
        q = self.quantiles(out, [lo, hi])
        return q[0], q[1]

    def lower_bound(self, out: ResponseOutput, level: float = 0.9) -> torch.Tensor:
        """One-sided lower quantile at ``1 - level`` (e.g. level 0.9 -> 10th pct)."""
        return self.quantiles(out, [1 - level])[0]

    # -- diagnostics ---------------------------------------------------
    def param_summary(self, out: ResponseOutput) -> dict:
        return {k: v for k, v in out.params.items() if torch.is_tensor(v) and v.dim() <= 1}

    def boundary_probs(self, out: ResponseOutput):
        return None


class BernoulliResponseHead(ResponseHead):
    """Phase 1 control arm. ``p = sigmoid(z)``; NLL == BCE-with-logits.
    Parameter-free -- all capacity is in the IRT core."""

    key = "bernoulli"
    target = "binary"
    distribution = "bernoulli"

    def forward(self, z: torch.Tensor, features: Optional[torch.Tensor] = None) -> ResponseOutput:
        return ResponseOutput(z=z, params={"logit": z, "p": torch.sigmoid(z)})

    def nll(self, y, out):
        import torch.nn.functional as F

        return F.binary_cross_entropy_with_logits(out["logit"], y.to(out["logit"].dtype),
                                                  reduction="none")

    def mean(self, out):
        return out["p"]

    def variance(self, out):
        p = out["p"]
        return p * (1 - p)

    def as_probability(self, out):
        return out["p"]

    def _sample(self, out, n: int = 512):
        p = out["p"].unsqueeze(0).expand(n, -1)
        return torch.bernoulli(p)


# --------------------------------------------------------------------------- #
# factory                                                                     #
# --------------------------------------------------------------------------- #
_REGISTRY: Optional[dict] = None


def _registry() -> dict:
    """``{name: ResponseHead subclass}``, built once. The continuous heads are
    imported lazily here to break the ``response_head <-> continuous_*`` cycle."""
    global _REGISTRY
    if _REGISTRY is None:
        from .continuous_beta import BetaResponseHead
        from .continuous_normal import NormalResponseHead
        from .continuous_zoib import ZOIBResponseHead

        _REGISTRY = {"bernoulli": BernoulliResponseHead, "normal": NormalResponseHead,
                     "beta": BetaResponseHead, "zoib": ZOIBResponseHead}
    return _REGISTRY


RESPONSE_MODELS = ("bernoulli", "normal", "beta", "zoib")


def build_response_head(name: str, cfg: Optional[dict] = None, *, feature_dim: int,
                        hidden: Optional[int] = 64) -> ResponseHead:
    name = (name or "bernoulli").lower()
    reg = _registry()
    require_choice(name, reg, field="response model")
    if name == "bernoulli":
        return BernoulliResponseHead()
    return reg[name](feature_dim=feature_dim, hidden=hidden, cfg=cfg or {})
