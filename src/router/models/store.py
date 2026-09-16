"""``ArtifactStore``: the store owns file locations, the artifact does not
(docs/architecture.md).

:class:`LocalArtifactStore` wraps the existing ``runs_dir/<name>/{model.pt,
run.json}`` checkpoint convention rather than reinventing it: :meth:`load`
delegates straight to :func:`router.nirt.checkpoint.load_run`, so every
checkpoint already on disk keeps loading unchanged through both the old
function and this new store -- ``load_run`` never reads ``run.json`` (only
``model.pt``), so this store's own leaner ``run.json`` sidecar (written by
:meth:`save`, carrying just the artifact's typed fields) never conflicts with
the richer one ``training.nirt.train.fit`` writes for its own runs (
``history``/``baselines``/``git_sha``  -- that richer provenance is
``TrainingRun`` territory, not ``RouterModelArtifact``'s job).
"""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any

from .artifacts import ArtifactFormat, NIRTArtifactPayload, RouterModelArtifact


class ArtifactStore(abc.ABC):
    @abc.abstractmethod
    def save(self, artifact: RouterModelArtifact) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, artifact_id: str) -> RouterModelArtifact:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, artifact_id: str) -> bool:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    """``root_path`` is a NIRT ``runs_dir``: artifacts live at
    ``root_path/<artifact_id>/{model.pt,run.json}``.

    Only :class:`~router.models.artifacts.NIRTArtifactPayload` is supported
    today -- the only format with an existing on-disk convention to wrap.
    Baseline/MLP payloads are Phase C's job, once those trainers exist as
    ``RouterModelTrainer``s.
    """

    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)

    def exists(self, artifact_id: str) -> bool:
        return (self.root_path / artifact_id / "model.pt").exists()

    def load(self, artifact_id: str) -> RouterModelArtifact:
        from ..nirt.checkpoint import load_run

        model, config, model_index = load_run(artifact_id, runs_dir=self.root_path)
        payload = NIRTArtifactPayload(
            state_dict=model.state_dict(),
            config=config,
            n_models=len(model_index),
            model_index=model_index,
            query_dim=model.query_dim,
            profile_dim=model.profile_dim,
        )
        meta = self._read_run_json(artifact_id)
        return RouterModelArtifact(
            artifact_id=artifact_id,
            payload=payload,
            content_hash=meta.get("content_hash", ""),
            # torch.save's zip/pickle container is the closest of the four
            # target formats; NPZ/SAFETENSORS/JSON don't apply to a NIRT run.
            format=ArtifactFormat.PICKLE,
            schema_version=meta.get("schema_version", "0"),
            supported_model_ids=list(model_index),
            router_model_name="nirt",
            router_model_version=meta.get("router_model_version", ""),
            training_dataset_version=meta.get("training_dataset_version", ""),
            training_run_id=meta.get("training_run_id", artifact_id),
            created_at=meta.get("created_at", ""),
            hyperparameters=config,
        )

    def save(self, artifact: RouterModelArtifact) -> str:
        if not isinstance(artifact.payload, NIRTArtifactPayload):
            raise NotImplementedError(
                f"LocalArtifactStore.save only supports NIRTArtifactPayload today "
                f"(got {type(artifact.payload).__name__})"
            )
        import torch

        out = self.root_path / artifact.artifact_id
        out.mkdir(parents=True, exist_ok=True)
        torch.save(artifact.payload.to_state(), out / "model.pt")
        (out / "run.json").write_text(
            json.dumps(
                {
                    "artifact_id": artifact.artifact_id,
                    "content_hash": artifact.content_hash,
                    "schema_version": artifact.schema_version,
                    "router_model_name": artifact.router_model_name,
                    "router_model_version": artifact.router_model_version,
                    "training_dataset_version": artifact.training_dataset_version,
                    "training_run_id": artifact.training_run_id,
                    "created_at": artifact.created_at,
                    "hyperparameters": artifact.hyperparameters,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(out)

    def _read_run_json(self, artifact_id: str) -> dict[str, Any]:
        path = self.root_path / artifact_id / "run.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
