"""A fitted router's identity: enough to resolve a live decision back to an
exact checkpoint, training dataset version, and hyperparameters.

Today's checkpoints (``runs_dir/<name>/{model.pt,run.json}``, loaded by
:func:`router.nirt.checkpoint.load_run`) carry most of this informally inside
``run.json``. Nothing constructs a :class:`RouterModelArtifact` yet -- this is
the typed version that :class:`~router.tracing.traces.ExecutionTrace` and
:class:`~router.decision.RoutingDecision` reference by ``artifact_id``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ArtifactFormat(enum.Enum):
    NPZ = "npz"
    SAFETENSORS = "safetensors"
    JSON = "json"
    PICKLE = "pickle"


@dataclass
class RouterModelArtifact:
    artifact_id: str
    path: str
    content_hash: str
    format: ArtifactFormat
    supported_model_ids: list[str] = field(default_factory=list)
    router_model_name: str = ""
    router_model_version: str = ""
    training_dataset_version: str = ""
    created_at: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
