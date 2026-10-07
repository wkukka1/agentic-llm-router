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

from router.config import coerce_auto_bool, coerce_hidden, require_choice
from router.nirt.components import (
    CostHead,
    DifficultyHead,
    OutputLengthHead,
    DiscriminationHead,
    InteractionLayer,
    LengthHead,
    WarmupBlender,
)

from .response_head import ResponseOutput, build_response_head

MODEL_PARAM_MODES = ("free", "projected")
COST_MODES = ("log_cost", "length")


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
    response_head: Optional[nn.Module] = None     # the head that produced `response`
    cost_mu: Optional[torch.Tensor] = None        # (B,)  expected-cost probe, log-USD location
    cost_log_sigma: Optional[torch.Tensor] = None  # (B,)  its log scale
    length_logits: Optional[torch.Tensor] = None  # (B, n_buckets)  output-length classifier

    def proba(self) -> torch.Tensor:
        """The response head's probability (``sigmoid(logit)`` for Bernoulli; for a
        continuous head ``sigmoid(logit)`` is not a predictive probability)."""
        if self.response_head is not None and self.response is not None:
            return self.response_head.as_probability(self.response)
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
        response_cfg: Optional[dict] = None, use_cost_head: bool = False,
        cost_hidden: Optional[int] = 64, cost_init_mu: Optional[float] = None,
        cost_price_dim: int = 0, cost_residual: bool = False, cost_mode: str = "log_cost",
        length_bucket_values=None,
    ):
        super().__init__()
        require_choice(model_params, MODEL_PARAM_MODES, field="model_params")
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

        # built LAST: constructing it earlier would shift the RNG stream every
        # quality layer is initialised from, breaking quality-path bit identity
        self.use_cost_head = bool(use_cost_head)
        self.cost_price_dim = int(cost_price_dim)
        # residual mode: cost_mu = log C(m) (train per-model offset, passed per row
        # as `cost_offset`) + head output, so the head learns only the query-level
        # deviation from each model's typical cost
        self.cost_residual = bool(cost_residual)
        # cost_mode "log_cost": regress log USD directly (CostHead).
        # cost_mode "length": classify output length into token buckets
        # (OutputLengthHead); expected cost = p_in * L_in + p_out * E[L_out].
        require_choice(cost_mode, COST_MODES, field="cost_mode")
        self.cost_mode = cost_mode
        self.cost_head: Optional[nn.Module] = None
        if self.use_cost_head and cost_mode == "length":
            if length_bucket_values is None:
                raise ValueError("cost_mode='length' needs length_bucket_values (train-split "
                                 "mean output length per bucket); the training harness supplies it")
            self.cost_head = OutputLengthHead(self.query_dim + self.dim + self.cost_price_dim,
                                              cost_hidden, length_bucket_values)
        elif self.use_cost_head:
            if cost_init_mu is None:
                raise ValueError("use_cost_head=True needs cost_init_mu (the train-split mean "
                                 "log cost); the training harness supplies it")
            self.cost_head = CostHead(self.query_dim + self.dim + self.cost_price_dim, cost_hidden,
                                      init_mu=float(cost_init_mu))

    @classmethod
    def from_config(cls, model_cfg: dict, *, n_models: int, query_dim: int = 768,
                    profile_dim: int = 768, relevance_dim: int = 0) -> "BaselineNIRT":
        m = dict(model_cfg or {})
        abl, wu = dict(m.get("ablation", {}) or {}), dict(m.get("warmup", {}) or {})
        dim = int(m.get("dim", m.get("theta_dim", 8)))
        cd = coerce_auto_bool(m.get("constrain_discrimination", "auto"), dim == 1)
        qh = coerce_hidden(m.get("query_hidden", 64))

        def pick(k, d):
            return abl.get(k, m.get(k, d))

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
            use_cost_head=bool(m.get("use_cost_head", False)),
            cost_hidden=coerce_hidden(m.get("cost_hidden", 64)),
            cost_init_mu=m.get("cost_init_mu"),
            cost_price_dim=int(m.get("cost_price_dim", 0) or 0),
            cost_residual=bool(m.get("cost_residual", False)),
            cost_mode=m.get("cost_mode", "log_cost") or "log_cost",
            length_bucket_values=(m.get("length_buckets") or {}).get("values"),
        )

    def cost_parameters(self) -> list:
        return list(self.cost_head.parameters()) if self.cost_head is not None else []

    def quality_parameters(self) -> list:
        """Every parameter except the cost probe's, matched by identity."""
        cost = {id(p) for p in self.cost_parameters()}
        return [p for p in self.parameters() if id(p) not in cost]

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

    def forward(self, e_q, model_ref, r_q=None, neighbor_mean=None, price_feats=None,
                cost_offset=None, length_prior=None) -> Output:
        e = self.warmup(e_q, neighbor_mean)
        if self.use_relevance and r_q is None:
            # same fill the data path uses for an unclustered query: uniform 1/C
            r_q = e_q.new_full((e_q.shape[0], self.relevance_dim), 1.0 / self.relevance_dim)
        a_q, gate = self.disc_head(e, r_q if self.use_relevance else None)
        b_q = self.diff_head(e)
        theta_m = self.latent_ability(model_ref)
        logit, base = self.interaction(a_q, theta_m, b_q, r_q if self.use_relevance else None)
        features = torch.cat([e, theta_m], dim=-1) if self.response_model != "bernoulli" else None
        # the length head feeds no loss (no length labels) -- not run on every
        # forward; call predict_length() explicitly
        out = Output(logit=logit, base_irt=base, a_q=a_q, b_q=b_q, theta_m=theta_m,
                     relevance_gate=gate, response=self.response_head(logit, features),
                     response_head=self.response_head)
        if self.cost_head is not None and self.cost_mode == "length":
            if length_prior is None:
                raise ValueError("length cost head needs length_prior (per-row log bucket prior; "
                                 "see training.nirt.baseline.length.attach_length_inputs)")
            out.length_logits = self.cost_head(self._cost_input(e, theta_m, price_feats),
                                               length_prior)
        elif self.cost_head is not None:
            out.cost_mu, out.cost_log_sigma = self.cost_head(
                self._cost_input(e, theta_m, price_feats))
            if self.cost_residual:
                if cost_offset is None:
                    raise ValueError("residual cost head needs cost_offset (per-row log C(m); "
                                     "see training.nirt.baseline.data.attach_cost_offset)")
                out.cost_mu = out.cost_mu + cost_offset.to(out.cost_mu.dtype)
        return out

    def _cost_input(self, e, theta_m, price_feats=None) -> torch.Tensor:
        """The probe's input: the quality representation, DETACHED (no gradient
        path back into the trunk), plus optional fixed model-price features."""
        feats = [e.detach(), theta_m.detach()]
        if self.cost_price_dim:
            if price_feats is None:
                raise ValueError("this cost head was trained with model price features; "
                                 "pass price_feats")
            feats.append(price_feats.to(e.dtype))
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def cost_features(self, e_q, model_ref, neighbor_mean=None, price_feats=None) -> torch.Tensor:
        """The frozen probe input for a batch, without running the quality heads."""
        return self._cost_input(self.warmup(e_q, neighbor_mean), self.latent_ability(model_ref),
                                price_feats)

    def predict_length(self, e_q, neighbor_mean=None) -> Optional[torch.Tensor]:
        return self.length_head(self.warmup(e_q, neighbor_mean)) if self.use_length_head else None

    @torch.no_grad()
    def predict_cost(self, e_q, model_ref, r_q=None, neighbor_mean=None, price_feats=None,
                     cost_offset=None):
        """Expected USD cost ``E[C | q, m]`` -> ``(B,)``."""
        if self.cost_head is None:
            raise ValueError("model has no cost head (train with cost.enabled: true)")
        if self.cost_mode == "length":
            raise ValueError("a length-mode head predicts tokens, not USD: use "
                             "batched_forward(fields=('cost',)) with in_tokens / token_price")
        out = self.forward(e_q, model_ref, r_q, neighbor_mean, price_feats, cost_offset)
        return CostHead.expected_cost(out.cost_mu, out.cost_log_sigma)

    @torch.no_grad()
    def predict_proba(self, e_q, model_ref, r_q=None, neighbor_mean=None) -> torch.Tensor:
        out = self.forward(e_q, model_ref, r_q, neighbor_mean)
        return self.response_head.as_probability(out.response)


def build_baseline_model(model_cfg: dict, *, n_models: int, query_dim: int = 768,
                         profile_dim: int = 768, relevance_dim: int = 0) -> BaselineNIRT:
    return BaselineNIRT.from_config(model_cfg, n_models=n_models, query_dim=query_dim,
                                    profile_dim=profile_dim, relevance_dim=relevance_dim)
