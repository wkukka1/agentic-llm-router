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
