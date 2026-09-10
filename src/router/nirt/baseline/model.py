"""``BaselineNIRT`` -- the Phase 1 plain Bernoulli / BCE NIRT baseline.

``model_latent`` orientation: ``theta_m`` = per-LLM ability (free Embedding, or
``W_theta @ e_m`` for cold-start); ``a_q``/``b_q`` = query discrimination /
difficulty (see :mod:`router.nirt.components`).

    base_irt = a_q . theta_m - b_q
    logit    = base_irt [+ gamma * interaction(...)]     ;   p = sigmoid(logit)

``forward`` returns an :class:`Output` exposing every intermediate for
diagnostics. This is an interpretable control arm -- Phase 2 replaces the
response head, not this core; do not add capacity here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from router.config import coerce_auto_bool, coerce_hidden
from router.nirt.components import (
    DifficultyHead,
    DiscriminationHead,
    InteractionLayer,
    LengthHead,
    WarmupBlender,
)

from .response_head import ResponseOutput, build_response_head

MODEL_PARAM_MODES = ("free", "projected")


@dataclass
class Output:
    logit: torch.Tensor            # (B,)  latent correctness location z_qm (= Phase 1 logit)
    base_irt: torch.Tensor         # (B,)  a_q . theta_m - b_q  (no interaction residual)
    a_q: torch.Tensor              # (B, K)
    b_q: torch.Tensor              # (B,)
    theta_m: torch.Tensor          # (B, K)
    relevance_gate: Optional[torch.Tensor] = None
    length_pred: Optional[torch.Tensor] = None
    response: Optional[ResponseOutput] = None    # response-head distribution params

    def proba(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)


class BaselineNIRT(nn.Module):
    orientation = "model_latent"
    arch = "baseline"

    def __init__(
        self, *, query_dim: int = 768, dim: int = 8, n_models: int = 1, profile_dim: int = 768,
        relevance_dim: int = 0, model_params: str = "free", query_hidden: Optional[int] = 64,
        constrain_discrimination: bool = True, bound_ability: bool = True,
        use_relevance: bool = True, use_interaction: bool = True, use_warmup: bool = False,
        warmup_alpha: float = 0.5, warmup_learnable: bool = False, use_length_head: bool = True,
        interaction_hidden: int = 32, response_model: str = "bernoulli",
        response_cfg: Optional[dict] = None,
    ):
        super().__init__()
        if model_params not in MODEL_PARAM_MODES:
            raise ValueError(f"model_params must be one of {MODEL_PARAM_MODES}, got {model_params!r}")
        self.query_dim, self.dim, self.n_models = int(query_dim), int(dim), int(n_models)
        self.profile_dim, self.relevance_dim = int(profile_dim), int(relevance_dim)
        self.model_params = model_params
        self.bound_ability = bool(bound_ability)
        self.use_relevance = bool(use_relevance and relevance_dim > 0)
        self.use_interaction = bool(use_interaction)
        self.use_warmup = bool(use_warmup)
        self.use_length_head = bool(use_length_head)

        self.warmup = WarmupBlender(self.use_warmup, warmup_alpha, warmup_learnable)
        self.disc_head = DiscriminationHead(self.query_dim, self.dim, query_hidden, self.relevance_dim,
                                            use_relevance=self.use_relevance,
                                            constrain=bool(constrain_discrimination))
        self.diff_head = DifficultyHead(self.query_dim, query_hidden)
        self.interaction = InteractionLayer(self.dim, self.relevance_dim if self.use_relevance else 0,
                                            interaction_hidden, enabled=self.use_interaction)
        self.length_head = LengthHead(self.query_dim, query_hidden, enabled=self.use_length_head)
        self.response_model = (response_model or "bernoulli").lower()
        self.response_head = build_response_head(self.response_model, response_cfg,
                                                 feature_dim=self.query_dim + self.dim,
                                                 hidden=query_hidden)

        if model_params == "free":
            self.theta = nn.Embedding(self.n_models, self.dim)
            nn.init.normal_(self.theta.weight, std=0.1)
        else:
            self.theta_proj = nn.Linear(self.profile_dim, self.dim)

    @classmethod
    def from_config(cls, model_cfg: dict, *, n_models: int, query_dim: int = 768,
                    profile_dim: int = 768, relevance_dim: int = 0) -> "BaselineNIRT":
        m = dict(model_cfg or {})
        abl, wu = dict(m.get("ablation", {}) or {}), dict(m.get("warmup", {}) or {})
        dim = int(m.get("dim", m.get("theta_dim", 8)))
        cd = coerce_auto_bool(m.get("constrain_discrimination", "auto"), dim == 1)
        qh = coerce_hidden(m.get("query_hidden", 64))
        pick = lambda k, d: abl.get(k, m.get(k, d))  # noqa: E731
        return cls(
            query_dim=query_dim, dim=dim, n_models=n_models, profile_dim=profile_dim,
            relevance_dim=relevance_dim, model_params=m.get("model_params", "free"),
            query_hidden=qh, constrain_discrimination=cd,
            bound_ability=bool(m.get("bound_ability", True)),
            use_relevance=bool(pick("use_relevance", True)),
            use_interaction=bool(pick("use_interaction", True)),
            use_warmup=bool(pick("use_warmup", wu.get("enabled", False))),
            warmup_alpha=float(m.get("warmup_alpha", wu.get("alpha", 0.5))),
            warmup_learnable=bool(wu.get("learnable", False)),
            use_length_head=bool(m.get("use_length_head", True)),
            interaction_hidden=int(m.get("interaction_hidden", 32)),
            response_model=m.get("response_model", "bernoulli"),
            response_cfg=m.get("response_cfg", {}),
        )

    def latent_ability(self, model_ref) -> torch.Tensor:
        if not torch.is_tensor(model_ref):
            model_ref = torch.as_tensor(model_ref)
        if self.model_params == "free":
            return self.theta(model_ref.long())
        theta = self.theta_proj(model_ref.float())
        return torch.sigmoid(theta) if self.bound_ability else theta

    def theta_table(self, profile_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full ``(M, K)`` ability matrix for diagnostics."""
        if self.model_params == "free":
            return self.theta.weight.detach()
        if profile_matrix is None:
            raise ValueError("projected theta needs the profile embedding matrix")
        with torch.no_grad():
            return self.latent_ability(profile_matrix)

    def forward(self, e_q, model_ref, r_q=None, neighbor_mean=None) -> Output:
        e = self.warmup(e_q, neighbor_mean)
        if self.use_relevance and r_q is None:
            r_q = e_q.new_zeros(e_q.shape[0], self.relevance_dim)
        a_q, gate = self.disc_head(e, r_q if self.use_relevance else None)
        b_q = self.diff_head(e)
        theta_m = self.latent_ability(model_ref)
        logit, base = self.interaction(a_q, theta_m, b_q, r_q if self.use_relevance else None)
        features = torch.cat([e, theta_m], dim=-1) if self.response_model != "bernoulli" else None
        return Output(logit=logit, base_irt=base, a_q=a_q, b_q=b_q, theta_m=theta_m,
                      relevance_gate=gate, response=self.response_head(logit, features),
                      length_pred=self.length_head(e) if self.use_length_head else None)

    @torch.no_grad()
    def predict_proba(self, e_q, model_ref, r_q=None, neighbor_mean=None) -> torch.Tensor:
        out = self.forward(e_q, model_ref, r_q, neighbor_mean)
        return self.response_head.as_probability(out.response)


def build_baseline_model(model_cfg: dict, *, n_models: int, query_dim: int = 768,
                         profile_dim: int = 768, relevance_dim: int = 0) -> BaselineNIRT:
    return BaselineNIRT.from_config(model_cfg, n_models=n_models, query_dim=query_dim,
                                    profile_dim=profile_dim, relevance_dim=relevance_dim)
