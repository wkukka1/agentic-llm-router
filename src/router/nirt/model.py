"""The NIRT response model (Phase 2).

Two **orientations** of the same bilinear IRT core, selected by
``model.orientation`` (:func:`build_model`):

``query_latent`` -- :class:`NIRTModel` (ours, the default) ::

    theta_q = f_theta(e_q)                 # latent ability on the QUERY
    (a_m, b_m) on the model                # free params, or linear heads on e_m
    logit   = a_m . theta_q - b_m

``model_latent`` -- :class:`IRTRouterModel` (the IRT-Router paper, Song et al. ACL'25) ::

    theta_m = [sigmoid] W_theta e_m        # latent ability on the LLM
    a_q = W_a e_q ,  b_q = W_b e_q         # discrimination / difficulty on the QUERY
    logit   = a_q . theta_m - b_q

The paper's orientation is the classical-IRT reading (examinee = LLM has ability,
item = query has difficulty) and yields reusable per-LLM ability vectors; ours
amortises the latent vector from the query text instead. We keep both and compare
(``scripts/nirt/ab_orientation.py``).

Both classes share one interface so the trainer is orientation-agnostic:
``forward(e_q, model_ref)`` where ``model_ref`` is a ``(B, profile_dim)`` float
tensor of profile embeddings (``model_params="projected"``) or a ``(B,)`` long
tensor of model indices (``model_params="free"``); plus ``predict_proba``,
``.dim``, ``.model_params``, ``.orientation``.

``dim`` (K) is the only knob between the 1-D and multidimensional model.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..config import coerce_auto_bool, coerce_hidden

MODEL_PARAM_MODES = ("projected", "free")
ORIENTATIONS = ("query_latent", "model_latent")
DIFFICULTY_MODES = ("scalar", "vector")


def _query_head(in_dim: int, out_dim: int, hidden: Optional[int]) -> nn.Module:
    """Linear ``in_dim -> out_dim``, or a 1-hidden-layer MLP when ``hidden`` is set."""
    if hidden:
        return nn.Sequential(
            nn.Linear(in_dim, int(hidden)), nn.ReLU(), nn.Linear(int(hidden), out_dim)
        )
    return nn.Linear(in_dim, out_dim)


class NIRTModel(nn.Module):
    """``query_latent`` orientation with ``theta_q = f_theta(e_q)`` and
    ``(a_m, b_m)`` free per model or projected from the profile embedding.

    ``difficulty="scalar"`` (default): ``logit = a_m . theta_q - b_m``, ``b_m`` a
    scalar per model (classical compensatory MIRT intercept).

    ``difficulty="vector"``: ``b_m`` is a ``K``-vector and
    ``logit = sum_k a_mk (theta_qk - b_mk)`` -- a per-latent-dimension difficulty,
    i.e. each ability axis has its own threshold before it contributes.
    """

    orientation = "query_latent"

    def __init__(
        self,
        *,
        query_dim: int = 768,
        dim: int = 1,
        n_models: int = 1,
        model_params: str = "projected",
        query_hidden: Optional[int] = 64,
        constrain_discrimination: bool = True,
        profile_dim: int = 768,
        difficulty: str = "scalar",
    ):
        super().__init__()
        if model_params not in MODEL_PARAM_MODES:
            raise ValueError(
                f"model_params must be one of {MODEL_PARAM_MODES}, got {model_params!r}"
            )
        if difficulty not in DIFFICULTY_MODES:
            raise ValueError(
                f"difficulty must be one of {DIFFICULTY_MODES}, got {difficulty!r}"
            )
        self.query_dim = int(query_dim)
        self.dim = int(dim)
        self.n_models = int(n_models)
        self.model_params = model_params
        self.query_hidden = query_hidden
        self.constrain_discrimination = bool(constrain_discrimination)
        self.profile_dim = int(profile_dim)
        self.difficulty = difficulty
        b_dim = self.dim if difficulty == "vector" else 1

        # -- query head: e_q -> theta_q in R^K -------------------------------
        self.query_head = _query_head(self.query_dim, self.dim, query_hidden)

        # -- model side: (a_m, b_m) ----------------------------------------
        if model_params == "projected":
            self.a_head = nn.Linear(self.profile_dim, self.dim)
            self.b_head = nn.Linear(self.profile_dim, b_dim)
        else:  # free
            self.a = nn.Embedding(self.n_models, self.dim)
            self.b = nn.Embedding(self.n_models, b_dim)
            nn.init.normal_(self.a.weight, std=0.1)
            nn.init.zeros_(self.b.weight)

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        model_cfg: dict,
        *,
        n_models: int,
        query_dim: int = 768,
        profile_dim: int = 768,
    ) -> "NIRTModel":
        dim = int(model_cfg.get("dim", 1))
        return cls(
            query_dim=query_dim,
            dim=dim,
            n_models=n_models,
            model_params=model_cfg.get("model_params", "projected"),
            query_hidden=coerce_hidden(model_cfg.get("query_hidden", 64)),
            constrain_discrimination=coerce_auto_bool(
                model_cfg.get("constrain_discrimination", "auto"), dim == 1
            ),
            profile_dim=profile_dim,
            difficulty=model_cfg.get("difficulty", "scalar") or "scalar",
        )

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def latent_query(self, e_q: torch.Tensor) -> torch.Tensor:
        """``theta_q`` for a batch of query embeddings -> ``(B, K)``."""
        return self.query_head(e_q)

    def model_parameters(
        self, model_ref: Union[torch.Tensor, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(a_m, b_m)`` for a batch.

        ``model_ref`` is a ``(B, profile_dim)`` float tensor of profile
        embeddings when ``model_params="projected"``, or a ``(B,)`` long tensor
        of model indices when ``model_params="free"``.
        """
        if self.model_params == "projected":
            a = self.a_head(model_ref)
            b = self.b_head(model_ref)
        else:
            if not torch.is_tensor(model_ref):
                model_ref = torch.as_tensor(model_ref, dtype=torch.long)
            a = self.a(model_ref)
            b = self.b(model_ref)
        if self.difficulty == "scalar":
            b = b.squeeze(-1)                       # (B,)
        if self.constrain_discrimination:
            a = F.softplus(a)
        return a, b

    def forward(
        self,
        e_q: torch.Tensor,
        model_ref: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Logit of ``P(correct)`` -> ``(B,)``. Apply ``sigmoid`` for a probability."""
        theta = self.latent_query(e_q)              # (B, K)
        a, b = self.model_parameters(model_ref)     # (B, K), (B,) or (B, K)
        if self.difficulty == "vector":
            return (a * (theta - b)).sum(-1)
        return (a * theta).sum(-1) - b

    @torch.no_grad()
    def predict_proba(
        self, e_q: torch.Tensor, model_ref: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(e_q, model_ref))


class IRTRouterModel(nn.Module):
    """``model_latent`` orientation (IRT-Router, Song et al. ACL'25):

        theta_m = [sigmoid] W_theta e_m        # ability on the LLM
        a_q = head_a(e_q) ,  b_q = head_b(e_q) # discrimination / difficulty on the query
        logit = a_q . theta_m - b_q
    """

    orientation = "model_latent"

    def __init__(
        self,
        *,
        query_dim: int = 768,
        dim: int = 1,
        n_models: int = 1,
        model_params: str = "projected",
        query_hidden: Optional[int] = 64,
        constrain_discrimination: bool = True,
        bound_ability: bool = True,
        profile_dim: int = 768,
    ):
        super().__init__()
        if model_params not in MODEL_PARAM_MODES:
            raise ValueError(
                f"model_params must be one of {MODEL_PARAM_MODES}, got {model_params!r}"
            )
        self.query_dim = int(query_dim)
        self.dim = int(dim)
        self.n_models = int(n_models)
        self.model_params = model_params
        self.query_hidden = query_hidden
        self.constrain_discrimination = bool(constrain_discrimination)
        self.bound_ability = bool(bound_ability)
        self.profile_dim = int(profile_dim)

        # -- query side: e_q -> (a_q discrimination, b_q difficulty) --------
        self.a_head = _query_head(self.query_dim, self.dim, query_hidden)
        self.b_head = _query_head(self.query_dim, 1, query_hidden)

        # -- model side: theta_m ability ---------------------------------
        if model_params == "projected":
            self.theta_head = nn.Linear(self.profile_dim, self.dim)
        else:  # free  (== the paper's M-IRT)
            self.theta = nn.Embedding(self.n_models, self.dim)
            nn.init.normal_(self.theta.weight, std=0.1)

    @classmethod
    def from_config(cls, model_cfg: dict, *, n_models: int,
                    query_dim: int = 768, profile_dim: int = 768) -> "IRTRouterModel":
        dim = int(model_cfg.get("dim", 1))
        return cls(
            query_dim=query_dim,
            dim=dim,
            n_models=n_models,
            model_params=model_cfg.get("model_params", "projected"),
            query_hidden=coerce_hidden(model_cfg.get("query_hidden", 64)),
            constrain_discrimination=coerce_auto_bool(
                model_cfg.get("constrain_discrimination", "auto"), dim == 1
            ),
            bound_ability=bool(model_cfg.get("bound_ability", True)),
            profile_dim=profile_dim,
        )

    def latent_ability(self, model_ref: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """``theta_m`` for a batch -> ``(B, K)``."""
        if self.model_params == "projected":
            theta = self.theta_head(model_ref)
            return torch.sigmoid(theta) if self.bound_ability else theta
        if not torch.is_tensor(model_ref):
            model_ref = torch.as_tensor(model_ref, dtype=torch.long)
        return self.theta(model_ref)

    def query_parameters(self, e_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(a_q, b_q)`` -> ``(B, K)``, ``(B,)``."""
        a = self.a_head(e_q)
        b = self.b_head(e_q).squeeze(-1)
        if self.constrain_discrimination:
            a = F.softplus(a)
        return a, b

    def forward(
        self, e_q: torch.Tensor, model_ref: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        theta_m = self.latent_ability(model_ref)     # (B, K)
        a_q, b_q = self.query_parameters(e_q)        # (B, K), (B,)
        return (a_q * theta_m).sum(-1) - b_q

    @torch.no_grad()
    def predict_proba(
        self, e_q: torch.Tensor, model_ref: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(e_q, model_ref))


_MODEL_CLASSES = {"query_latent": NIRTModel, "model_latent": IRTRouterModel}


def build_model(
    model_cfg: dict, *, n_models: int, query_dim: int = 768, profile_dim: int = 768
):
    """Instantiate the model for ``model_cfg['orientation']`` (default ``query_latent``)."""
    orientation = model_cfg.get("orientation") or "query_latent"
    if orientation not in _MODEL_CLASSES:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}")
    return _MODEL_CLASSES[orientation].from_config(
        model_cfg, n_models=n_models, query_dim=query_dim, profile_dim=profile_dim
    )
