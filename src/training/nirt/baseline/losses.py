"""Phase 1 loss = BCE-with-logits + IRT identifiability regularisers.

``a . theta - b`` is invariant under a shift/scale of the latent space; the
centring (``theta_center_l2``) + mild L2 terms fix a stable, comparable
convention across runs. They are documented and NOT tuned to a validation number.
The ``theta`` terms are applied by the trainer to the *fitted* ability rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F


@dataclass
class RegConfig:
    theta_l2: float = 1e-4           # shrink ability scale
    theta_center_l2: float = 1e-3    # pin mean_m theta_m ~ 0
    discrimination_l2: float = 0.0
    difficulty_l2: float = 1e-4

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RegConfig":
        d = d or {}
        return cls(**{f: float(d.get(f, getattr(cls, f))) for f in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def bce_loss(logit: torch.Tensor, y: torch.Tensor, *,
             pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean BCE-with-logits. ``y`` may be soft (in [0, 1])."""
    return F.binary_cross_entropy_with_logits(logit, y.to(logit.dtype), pos_weight=pos_weight)


def regularization(reg: RegConfig, *, theta_all=None, a_q=None, b_q=None) -> torch.Tensor:
    terms = []
    if theta_all is not None:
        if reg.theta_l2:
            terms.append(reg.theta_l2 * theta_all.pow(2).mean())
        if reg.theta_center_l2:
            terms.append(reg.theta_center_l2 * theta_all.mean(dim=0).pow(2).sum())
    if a_q is not None and reg.discrimination_l2:
        terms.append(reg.discrimination_l2 * a_q.pow(2).mean())
    if b_q is not None and reg.difficulty_l2:
        terms.append(reg.difficulty_l2 * b_q.pow(2).mean())
    return torch.stack(terms).sum() if terms else torch.zeros(())


def class_balance_pos_weight(y: torch.Tensor) -> torch.Tensor:
    """``#neg / #pos`` for ``BCEWithLogitsLoss(pos_weight=...)``."""
    pos = (y >= 0.5).float().sum().clamp(min=1.0)
    return ((y.numel() - pos).clamp(min=1.0) / pos).detach()
