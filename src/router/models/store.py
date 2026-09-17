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
import os
import threading
from pathlib import Path
from typing import Any, Callable

from .artifacts import ArtifactFormat, MLPArtifactPayload, NIRTArtifactPayload, RouterModelArtifact


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

    Supports :class:`~router.models.artifacts.NIRTArtifactPayload` (wraps the
    existing ``training.nirt.train.fit`` convention) and
    :class:`~router.models.artifacts.MLPArtifactPayload` (a new convention --
    ``fit_mlp_router`` persisted nothing before this). **Not**
    :class:`~router.models.artifacts.BaselineArtifactPayload`: that format's
    loader lives under ``training`` (``training.nirt.baseline.checkpoint``),
    which this ``router``-side module must not import -- see
    :class:`~training.nirt.baseline.trainer.BaselineNIRTTrainer`, which
    builds that payload in-memory instead of through this store.
    """

    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)

    def exists(self, artifact_id: str) -> bool:
        return (self.root_path / artifact_id / "model.pt").exists()

    def load(self, artifact_id: str) -> RouterModelArtifact:
        import torch

        blob_path = self.root_path / artifact_id / "model.pt"
        fmt = torch.load(blob_path, map_location="cpu", weights_only=True).get("format")
        if fmt == "mlp":
            return self._load_mlp(artifact_id)
        return self._load_nirt(artifact_id)

    def _load_nirt(self, artifact_id: str) -> RouterModelArtifact:
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

    def _load_mlp(self, artifact_id: str) -> RouterModelArtifact:
        import torch

        blob = torch.load(self.root_path / artifact_id / "model.pt",
                          map_location="cpu", weights_only=True)
        payload = MLPArtifactPayload.from_state(blob)
        meta = self._read_run_json(artifact_id)
        return RouterModelArtifact(
            artifact_id=artifact_id,
            payload=payload,
            content_hash=meta.get("content_hash", ""),
            format=ArtifactFormat.PICKLE,
            schema_version=meta.get("schema_version", "0"),
            supported_model_ids=list(payload.model_ids),
            router_model_name="mlp",
            training_run_id=meta.get("training_run_id", artifact_id),
            created_at=meta.get("created_at", ""),
            hyperparameters=meta.get("hyperparameters", {}),
        )

    @staticmethod
    def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
        """Write via a unique same-directory temp file, then :func:`os.replace`
        (atomic on POSIX and Windows) into ``path``. A reader only ever sees
        ``path`` fully absent or fully written -- never a half-written file
        from a killed process, and never a good file clobbered mid-overwrite
        by a bad one (RM-01). ``write(tmp_path)`` performs the actual write."""
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            write(tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def save(self, artifact: RouterModelArtifact) -> str:
        if not isinstance(artifact.payload, (NIRTArtifactPayload, MLPArtifactPayload)):
            raise NotImplementedError(
                f"LocalArtifactStore.save does not support {type(artifact.payload).__name__}"
            )
        import torch

        out = self.root_path / artifact.artifact_id
        out.mkdir(parents=True, exist_ok=True)
        self._atomic_write(out / "model.pt", lambda tmp: torch.save(artifact.payload.to_state(), tmp))
        run_json_text = json.dumps(
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
        )
        # written last: a reader never sees a complete model.pt paired with a
        # missing/stale run.json from an interrupted write (RM-01)
        self._atomic_write(out / "run.json", lambda tmp: tmp.write_text(run_json_text, encoding="utf-8"))
        return str(out)

    def _read_run_json(self, artifact_id: str) -> dict[str, Any]:
        path = self.root_path / artifact_id / "run.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
