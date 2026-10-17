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
from typing import Any, Optional


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

    def save(self, path, *, encoder: dict[str, Any], model_costs: Optional[list[float]] = None,
             meta: Optional[dict[str, Any]] = None) -> None:
        """One ``.npz`` the gateway serves with numpy alone: the two layers'
        weights (``Linear -> ReLU -> Dropout -> Linear``; dropout is off at
        inference) and a JSON header. No training prompts or scores inside."""
        import json

        import numpy as np

        def arr(key):
            v = self.state_dict[key]
            return np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float32)

        header = {"format": "mlp", "model_ids": self.model_ids, "in_dim": self.in_dim, "hidden": self.hidden,
                  "dropout": self.dropout, "pathway": self.pathway, "encoder": encoder,
                  "model_costs": model_costs, **(meta or {})}
        np.savez_compressed(path, w1=arr("0.weight"), b1=arr("0.bias"), w2=arr("3.weight"), b2=arr("3.bias"),
                            header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8))

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
class KNNArtifactPayload(ArtifactPayload):
    """A kNN router as plain data: the training queries' embeddings and their
    per-model scores. A prompt's predicted quality per model is the mean score
    of its ``k`` nearest training queries (cosine) -- what
    :class:`~router.routing.routers.KNNRouter` computes from the full
    training data, without needing that data.

    This is what makes the gateway installable on its own: LLMRouterBench's
    8,602 training queries are ~7 MB as an ``.npz`` (embeddings in float16),
    against gigabytes for the dataset, and ``encoder`` names a model the
    ``fastembed`` backend serves without PyTorch.

    ``scores`` are column-mean imputed (a query a model never saw gets that
    model's mean), as in :func:`router.nirt.baselines_infer._fit_knn`.
    ``model_costs`` is each model's mean training cost, for
    ``RouterModel.default_model_costs``."""

    embeddings: Any                 # [N, d] unit vectors (float16 on disk)
    scores: Any                     # [N, M] float32
    model_ids: list[str]
    k: int
    encoder: dict[str, Any]         # EncoderConfig fields: backend, model_name, normalize, max_seq_length
    model_costs: Optional[list[float]] = None

    def to_state(self) -> dict[str, Any]:
        return {"format": "knn", "embeddings": self.embeddings, "scores": self.scores,
                "model_ids": self.model_ids, "k": self.k, "encoder": self.encoder,
                "model_costs": self.model_costs}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "KNNArtifactPayload":
        return cls(embeddings=state["embeddings"], scores=state["scores"], model_ids=list(state["model_ids"]),
                   k=int(state["k"]), encoder=dict(state["encoder"]), model_costs=state.get("model_costs"))

    def save(self, path, *, meta: Optional[dict[str, Any]] = None) -> None:
        """One compressed ``.npz``: the two arrays plus a JSON header."""
        import json

        import numpy as np

        header = {"format": "knn", "model_ids": self.model_ids, "k": self.k, "encoder": self.encoder,
                  "model_costs": self.model_costs, **(meta or {})}
        np.savez_compressed(path, embeddings=np.asarray(self.embeddings, dtype=np.float16),
                            scores=np.asarray(self.scores, dtype=np.float32),
                            header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8))

    @classmethod
    def load(cls, path) -> tuple["KNNArtifactPayload", dict[str, Any]]:
        """``(payload, header)`` from a file :meth:`save` wrote."""
        import json

        import numpy as np

        with np.load(path, allow_pickle=False) as z:
            header = json.loads(z["header"].tobytes().decode())
            if header.get("format") != "knn":
                raise ValueError(f"{path} is not a kNN router file (format={header.get('format')!r})")
            payload = cls(embeddings=z["embeddings"].astype(np.float32), scores=z["scores"].astype(np.float32),
                          model_ids=list(header["model_ids"]), k=int(header["k"]),
                          encoder=dict(header["encoder"]), model_costs=header.get("model_costs"))
        return payload, header


def load_router_file(path) -> tuple[ArtifactPayload, dict[str, Any]]:
    """``(payload, header)`` from a router ``.npz`` (``aar-gateway setup``):
    a :class:`KNNArtifactPayload` or an :class:`MLPArtifactPayload` whose
    ``state_dict`` holds numpy arrays, by the header's ``format``."""
    import json

    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(z["header"].tobytes().decode())
        kind = header.get("format")
        if kind == "knn":
            return KNNArtifactPayload.load(path)
        if kind == "mlp":
            state = {"0.weight": z["w1"], "0.bias": z["b1"], "3.weight": z["w2"], "3.bias": z["b2"]}
            payload = MLPArtifactPayload(state_dict=state, model_ids=list(header["model_ids"]),
                                         in_dim=int(header["in_dim"]), hidden=int(header["hidden"]),
                                         dropout=float(header.get("dropout", 0.0)),
                                         pathway=header.get("pathway", "retrieval"))
            return payload, header
    raise ValueError(f"{path} is not a router file this version can load (format={kind!r})")


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
