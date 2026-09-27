"""A fitted router's identity: enough to resolve a live decision back to an
exact checkpoint, training dataset version, and hyperparameters.

Today's checkpoints (``runs_dir/<name>/{model.pt,run.json}``, loaded by
:func:`router.nirt.checkpoint.load_run`) carry most of this informally inside
``run.json``. :class:`~router.models.store.LocalArtifactStore` is what
actually constructs a :class:`RouterModelArtifact` from one, wrapping
``load_run`` rather than reinventing the on-disk format -- this is the typed
version that :class:`~router.tracing.traces.ExecutionTrace` and
:class:`~router.decision.RoutingDecision` reference by ``artifact_id``.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any


class ArtifactFormat(enum.Enum):
    NPZ = "npz"
    SAFETENSORS = "safetensors"
    JSON = "json"
    PICKLE = "pickle"


class ArtifactPayload(abc.ABC):
    """Opaque to serving: a trainer and a :class:`~router.routing.base.RouterModel`
    registered under the same ``router_model_name`` agree on the payload shape
    (:class:`~router.models.registry.RouterModelFactory` rejects a mismatch).
    Just enough structure -- :meth:`to_state`/:meth:`from_state` -- for
    :class:`~router.models.store.ArtifactStore` to (de)serialize any payload
    generically without knowing its shape."""

    @abc.abstractmethod
    def to_state(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_state(cls, state: dict[str, Any]) -> "ArtifactPayload":
        raise NotImplementedError


@dataclass
class NIRTArtifactPayload(ArtifactPayload):
    """A lossless wrapper around the ``model.pt`` dict
    ``training.nirt.train.fit`` already writes -- not a new format."""

    state_dict: dict[str, Any]
    config: dict[str, Any]
    n_models: int
    model_index: dict[str, int]
    query_dim: int
    profile_dim: int

    def to_state(self) -> dict[str, Any]:
        return {
            "format": "nirt",
            "state_dict": self.state_dict,
            "config": self.config,
            "n_models": self.n_models,
            "model_index": self.model_index,
            "query_dim": self.query_dim,
            "profile_dim": self.profile_dim,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "NIRTArtifactPayload":
        return cls(
            state_dict=state["state_dict"],
            config=state["config"],
            n_models=state["n_models"],
            model_index=state["model_index"],
            query_dim=state["query_dim"],
            profile_dim=state["profile_dim"],
        )


@dataclass
class BaselineArtifactPayload(ArtifactPayload):
    """A lossless wrapper around the ``model.pt`` dict
    ``training.nirt.baseline.checkpoint.save_checkpoint`` already writes --
    not a new format. Unlike :class:`NIRTArtifactPayload`, there is no
    serving-side loader for this format yet (``training.nirt.baseline.checkpoint``
    lives under ``training``, which ``router`` must not import), so
    :class:`~router.models.store.LocalArtifactStore` does not support this
    payload -- :class:`~training.nirt.baseline.trainer.BaselineNIRTTrainer`
    builds it directly from the in-memory ``fit()`` result instead."""

    state_dict: dict[str, Any]
    model_cfg: dict[str, Any]
    model_index: dict[str, int]
    query_dim: int
    profile_dim: int
    relevance_dim: int

    def to_state(self) -> dict[str, Any]:
        return {
            "format": "baseline",
            "state_dict": self.state_dict,
            "model_cfg": self.model_cfg,
            "model_index": self.model_index,
            "query_dim": self.query_dim,
            "profile_dim": self.profile_dim,
            "relevance_dim": self.relevance_dim,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "BaselineArtifactPayload":
        return cls(
            state_dict=state["state_dict"],
            model_cfg=state["model_cfg"],
            model_index=state["model_index"],
            query_dim=state["query_dim"],
            profile_dim=state["profile_dim"],
            relevance_dim=state["relevance_dim"],
        )


@dataclass
class MLPArtifactPayload(ArtifactPayload):
    """The IRT-free ``e_q -> R^M`` MLP router (:func:`router.nirt.baselines_infer.build_mlp_router`).

    Unlike NIRT/baseline, ``training.trainers.mlp_router.fit_mlp_router``
    persisted nothing before :class:`~training.trainers.mlp_router.MLPRouterTrainer`
    -- there is no prior on-disk convention this wraps, this *is* the
    convention. Rebuild with ``build_mlp_router(in_dim, len(model_ids), hidden,
    dropout)`` + ``load_state_dict``.

    ``pathway``/``query_pathway``/``query_features`` mirror the NIRT
    artifact's ``config["data"]`` keys (MLPROUTERTRAINER-001) -- without them
    ``MLPRouter.from_artifact`` can't know which embedding space the model
    was trained on and silently serves the wrong one (MLPROUTER-001).
    Defaulted in :meth:`from_state` so an artifact saved before this field
    existed still loads, at the same ``pathway="irt"`` default the router
    constructor used before either bug was fixed."""

    state_dict: dict[str, Any]
    model_ids: list[str]
    in_dim: int
    hidden: int
    dropout: float
    pathway: str = "irt"
    query_pathway: Any = None
    query_features: Any = None

    def to_state(self) -> dict[str, Any]:
        return {
            "format": "mlp",
            "state_dict": self.state_dict,
            "model_ids": self.model_ids,
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "pathway": self.pathway,
            "query_pathway": self.query_pathway,
            "query_features": self.query_features,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "MLPArtifactPayload":
        return cls(
            state_dict=state["state_dict"],
            model_ids=state["model_ids"],
            in_dim=state["in_dim"],
            hidden=state["hidden"],
            dropout=state["dropout"],
            pathway=state.get("pathway", "irt"),
            query_pathway=state.get("query_pathway"),
            query_features=state.get("query_features"),
        )


@dataclass
class RouterModelArtifact:
    artifact_id: str
    payload: ArtifactPayload
    content_hash: str
    format: ArtifactFormat
    schema_version: str = ""
    supported_model_ids: list[str] = field(default_factory=list)
    router_model_name: str = ""
    router_model_version: str = ""
    training_dataset_version: str = ""
    training_run_id: str = ""
    created_at: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
