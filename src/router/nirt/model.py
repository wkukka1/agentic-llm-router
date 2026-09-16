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

import warnings
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..config import coerce_auto_bool, coerce_hidden, require_choice

MODEL_PARAM_MODES = ("projected", "free")
ORIENTATIONS = ("query_latent", "model_latent")
DIFFICULTY_MODES = ("scalar", "vector")
HEAD_NORMS = ("none", "layernorm")
HEAD_ACTIVATIONS = ("relu", "gelu")

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU}

# NIRTModel keys IRTRouterModel does not read -> "value means the default".
_NIRT_ONLY_KNOBS = {
    "difficulty": lambda v: v in (None, "", "scalar"),
    "model_hidden": lambda v: coerce_hidden(v) is None,
    "interaction": lambda v: not v,
    "interaction_hidden": lambda v: True,        # only meaningful with interaction
    "center_discrimination": lambda v: not v,
    "query_head_batchnorm": lambda v: not v,
}


class _MLPHead(nn.Module):
    """``in_dim -> hidden -> ... -> out_dim`` with configurable depth / norm /
    activation / dropout / residual. Only instantiated when at least one of those
    is non-default; the default single-hidden-layer ReLU head stays a bare
    :class:`nn.Sequential` (see :func:`_query_head`) so old checkpoints load."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, *, layers: int,
                 norm: str, activation: str, dropout: float, residual: bool):
        super().__init__()
        act = _ACT[activation]

        def block(d_in: int, d_out: int) -> nn.Sequential:
            seq: list[nn.Module] = [nn.Linear(d_in, d_out)]
            if norm == "layernorm":
                seq.append(nn.LayerNorm(d_out))
            seq.append(act())
            if dropout > 0:
                seq.append(nn.Dropout(dropout))
            return nn.Sequential(*seq)

        self.blocks = nn.ModuleList(
            [block(in_dim, hidden)] + [block(hidden, hidden) for _ in range(layers - 1)]
        )
        self.head = nn.Linear(hidden, out_dim)
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks[0](x)
        for blk in self.blocks[1:]:
            h = h + blk(h) if self.residual else blk(h)
        return self.head(h)


def _query_head(
    in_dim: int,
    out_dim: int,
    hidden: Optional[int],
    *,
    layers: int = 1,
    norm: str = "none",
    activation: str = "relu",
    dropout: float = 0.0,
    residual: bool = False,
) -> nn.Module:
    """``e -> R^out``. Linear when ``hidden`` is falsy; the legacy
    ``Linear -> ReLU -> Linear`` :class:`nn.Sequential` when every extra knob is at
    its default (so existing ``state_dict``s load unchanged); otherwise an
    :class:`_MLPHead`."""
    if not hidden:
        return nn.Linear(in_dim, out_dim)
    if layers == 1 and norm == "none" and activation == "relu" and dropout == 0.0 and not residual:
        return nn.Sequential(
            nn.Linear(in_dim, int(hidden)), nn.ReLU(), nn.Linear(int(hidden), out_dim)
        )
    return _MLPHead(in_dim, out_dim, int(hidden), layers=int(layers), norm=norm,
                    activation=activation, dropout=float(dropout), residual=bool(residual))


def _head_kwargs(model_cfg: dict) -> dict:
    """Pull the P2a query-head knobs out of ``model_cfg`` (all default to the
    legacy single-hidden-layer ReLU head)."""
    norm = str(model_cfg.get("query_head_norm", "none") or "none")
    act = str(model_cfg.get("query_head_activation", "relu") or "relu")
    if norm not in HEAD_NORMS:
        raise ValueError(f"query_head_norm must be one of {HEAD_NORMS}, got {norm!r}")
    if act not in HEAD_ACTIVATIONS:
        raise ValueError(f"query_head_activation must be one of {HEAD_ACTIVATIONS}, got {act!r}")
    return {
        "layers": int(model_cfg.get("query_head_layers", 1) or 1),
        "norm": norm,
        "activation": act,
        "dropout": float(model_cfg.get("query_head_dropout", 0.0) or 0.0),
        "residual": bool(model_cfg.get("query_head_residual", False)),
    }


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
        head_kwargs: Optional[dict] = None,
        model_hidden: Optional[int] = None,
        interaction: bool = False,
        interaction_hidden: int = 32,
        center_discrimination: bool = False,
        query_head_batchnorm: bool = False,
    ):
        super().__init__()
        require_choice(model_params, MODEL_PARAM_MODES, field="model_params")
        require_choice(difficulty, DIFFICULTY_MODES, field="difficulty")
        self.query_dim = int(query_dim)
        self.dim = int(dim)
        self.n_models = int(n_models)
        self.model_params = model_params
        self.query_hidden = query_hidden
        self.constrain_discrimination = bool(constrain_discrimination)
        self.profile_dim = int(profile_dim)
        self.difficulty = difficulty
        self.head_kwargs = dict(head_kwargs or {})
        self.model_hidden = model_hidden
        self.interaction = bool(interaction)
        self.interaction_hidden = int(interaction_hidden)
        self.center_discrimination = bool(center_discrimination)
        self.query_head_batchnorm = bool(query_head_batchnorm)
        # projected-mode centring reference set at inference
        # (set_discrimination_reference); a plain attribute, not a buffer, so
        # state_dicts are unchanged
        self._a_ref: Optional[torch.Tensor] = None
        b_dim = self.dim if difficulty == "vector" else 1

        # -- query head: e_q -> theta_q in R^K -------------------------------
        self.query_head = _query_head(self.query_dim, self.dim, query_hidden, **self.head_kwargs)

        # -- optional whitening of theta_q (P7b) -----------------------------
        # Counter-pressure against the query head collapsing onto a low-rank
        # subspace: BatchNorm forces Cov(theta_q) ~= I_K over a batch, whose
        # effective rank ((sum lambda)^2 / sum lambda^2) is exactly K. Only
        # constructed when enabled, so a disabled model's state_dict is
        # unchanged. ``affine=False`` -- no learnable scale/shift, so it can
        # only decorrelate/rescale, never re-introduce a shared shift.
        if self.query_head_batchnorm:
            self.query_bn = nn.BatchNorm1d(self.dim, affine=False)

        # -- model side: (a_m, b_m) ----------------------------------------
        # ``model_hidden`` (P2b): a 1-hidden-layer MLP on the profile embedding
        # instead of a bare linear projection (``projected`` mode only). ``None``
        # keeps the legacy ``nn.Linear`` heads, so old checkpoints load.
        if model_params == "projected":
            from .components import mlp

            self.a_head = mlp(self.profile_dim, self.dim, model_hidden)
            self.b_head = mlp(self.profile_dim, b_dim, model_hidden)
        else:  # free
            self.a = nn.Embedding(self.n_models, self.dim)
            self.b = nn.Embedding(self.n_models, b_dim)
            nn.init.normal_(self.a.weight, std=0.1)
            nn.init.zeros_(self.b.weight)

        # -- optional non-additive interaction residual (P2c) --------------
        # ``logit = bilinear_base + gamma * net([theta_q, a_m, theta_q * a_m])``.
        # ``gamma`` starts at 0 so switching it on is a no-op at init; the module
        # is only created when enabled, so a disabled model's state_dict is
        # byte-identical to the pre-P2c architecture.
        if self.interaction:
            from .components import mlp

            self.interaction_net = mlp(3 * self.dim, 1, self.interaction_hidden)
            self.interaction_gamma = nn.Parameter(torch.zeros(()))

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
            head_kwargs=_head_kwargs(model_cfg),
            model_hidden=coerce_hidden(model_cfg.get("model_hidden")),
            interaction=bool(model_cfg.get("interaction", False)),
            interaction_hidden=int(model_cfg.get("interaction_hidden", 32) or 32),
            center_discrimination=bool(model_cfg.get("center_discrimination", False)),
            query_head_batchnorm=bool(model_cfg.get("query_head_batchnorm", False)),
        )

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def latent_query(self, e_q: torch.Tensor) -> torch.Tensor:
        """``theta_q`` for a batch of query embeddings -> ``(B, K)``."""
        theta = self.query_head(e_q)
        if self.query_head_batchnorm:
            if self.training and theta.shape[0] == 1:
                # batch statistics are undefined for one row (BatchNorm1d raises);
                # normalise with the running statistics instead
                bn = self.query_bn
                theta = F.batch_norm(theta, bn.running_mean, bn.running_var,
                                     training=False, eps=bn.eps)
            else:
                theta = self.query_bn(theta)
        return theta

    @torch.no_grad()
    def set_discrimination_reference(
        self, profile_embeddings: Optional[Union[torch.Tensor, np.ndarray]]
    ) -> None:
        """Fix the projected-mode centring reference ``r`` to the mean
        discrimination over a *candidate pool* (``(M, profile_dim)`` profile
        embeddings), or clear it with ``None``.

        Without it, projected mode centres on the mean of whatever rows share
        the forward call, so inference depends on batch boundaries (and a
        one-model batch centres every ``a_m`` to zero). Only
        ``center_discrimination`` + ``projected`` models use it; training
        leaves it unset (per-query batches are exact there)."""
        if profile_embeddings is None or self.model_params != "projected":
            self._a_ref = None
            return
        w = next(self.a_head.parameters())
        ref = torch.as_tensor(np.asarray(profile_embeddings), dtype=w.dtype, device=w.device)
        a = self.a_head(ref)
        if self.constrain_discrimination:
            a = F.softplus(a)
        self._a_ref = a.mean(0, keepdim=True)

    def _center_discrimination(self, a: torch.Tensor) -> torch.Tensor:
        """Subtract a reference vector ``r`` from every ``a_m`` before ranking.

        Exactly lossless for within-query ranking: for any two candidates
        ``m, n`` scored against the same ``theta_q``, ``(a_m - r).theta_q -
        (a_n - r).theta_q == a_m.theta_q - a_n.theta_q`` for ANY fixed ``r``
        -- the ``r.theta_q`` term cancels. ``r`` only needs to be the same
        constant across every candidate compared for one routing decision.

        ``free`` mode: ``r`` = mean over the FULL model pool
        (``self.a.weight``, post-constraint), independent of which models
        appear in this particular forward call -- required so two separate
        calls scoring different candidates still use the same ``r``.
        ``projected`` mode: the pool reference from
        :meth:`set_discrimination_reference` when one is set (inference);
        otherwise ``r`` = mean over the CURRENT batch. Exact when the batch is
        one query's full candidate set (``train.sampler: query``); a converged
        approximation otherwise (see docs on the P7b centering ablation).
        """
        if self.model_params == "free":
            raw = self.a.weight
            r = (F.softplus(raw) if self.constrain_discrimination else raw).mean(0, keepdim=True)
        elif self._a_ref is not None:
            r = self._a_ref
        else:
            r = a.mean(0, keepdim=True)
        return a - r

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
        if self.center_discrimination:
            a = self._center_discrimination(a)
        return a, b

    def score(self, theta: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Logit from an already-computed ``(theta_q, a_m, b_m)`` triple -> ``(B,)``.

        The bilinear base (scalar- or vector-difficulty) plus the interaction
        residual when ``self.interaction`` is set -- the one place this scoring
        formula is defined. ``forward`` calls it after ``model_parameters``;
        ``training.nirt.pairwise.pairwise_logit_diff`` calls it twice (once per
        battle side) against one shared ``theta`` (XD-09)."""
        if self.difficulty == "vector":
            base = (a * (theta - b)).sum(-1)
        else:
            base = (a * theta).sum(-1) - b
        if self.interaction:
            feats = torch.cat([theta, a, theta * a], dim=-1)      # (B, 3K)
            base = base + self.interaction_gamma * self.interaction_net(feats).squeeze(-1)
        return base

    def forward(
        self,
        e_q: torch.Tensor,
        model_ref: Union[torch.Tensor, np.ndarray],
        *,
        theta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Logit of ``P(correct)`` -> ``(B,)``. Apply ``sigmoid`` for a probability.

        ``theta`` (an already-computed ``latent_query(e_q)``) lets a trainer that
        also needs ``theta_q`` run the query head once -- one BatchNorm update and
        one dropout mask per step."""
        if theta is None:
            theta = self.latent_query(e_q)          # (B, K)
        a, b = self.model_parameters(model_ref)     # (B, K), (B,) or (B, K)
        return self.score(theta, a, b)

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
        head_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        require_choice(model_params, MODEL_PARAM_MODES, field="model_params")
        self.query_dim = int(query_dim)
        self.dim = int(dim)
        self.n_models = int(n_models)
        self.model_params = model_params
        self.query_hidden = query_hidden
        self.constrain_discrimination = bool(constrain_discrimination)
        self.bound_ability = bool(bound_ability)
        self.profile_dim = int(profile_dim)
        self.head_kwargs = dict(head_kwargs or {})

        # -- query side: e_q -> (a_q discrimination, b_q difficulty) --------
        self.a_head = _query_head(self.query_dim, self.dim, query_hidden, **self.head_kwargs)
        self.b_head = _query_head(self.query_dim, 1, query_hidden, **self.head_kwargs)

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
        ignored = [k for k, is_default in _NIRT_ONLY_KNOBS.items()
                   if k in model_cfg and not is_default(model_cfg[k])]
        if ignored:
            warnings.warn(
                f"orientation=model_latent ignores NIRT-only model keys {ignored}; "
                "this run trains the plain IRT-Router model",
                stacklevel=2,
            )
        if model_cfg.get("model_params", "projected") == "free" and "bound_ability" in model_cfg:
            warnings.warn(
                "bound_ability has no effect with model_params=free (free-mode "
                "ability is an unbounded embedding, as in the paper's M-IRT)",
                stacklevel=2,
            )
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
            head_kwargs=_head_kwargs(model_cfg),
        )

    def latent_ability(self, model_ref: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """``theta_m`` for a batch -> ``(B, K)``.

        ``bound_ability`` (sigmoid) applies to ``projected`` mode only; ``free``
        mode keeps the unbounded embedding (paper M-IRT) whatever the flag says,
        so saved free-mode runs score unchanged."""
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
    require_choice(orientation, _MODEL_CLASSES, field="orientation")
    return _MODEL_CLASSES[orientation].from_config(
        model_cfg, n_models=n_models, query_dim=query_dim, profile_dim=profile_dim
    )
