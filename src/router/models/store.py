"""``ArtifactStore``: the store owns file locations, the artifact does not
(docs/architecture.md).

:class:`LocalArtifactStore` wraps the existing ``runs_dir/<name>/{model.pt,
run.json}`` checkpoint convention rather than reinventing it: :meth:`load`
delegates straight to :func:`router.nirt.checkpoint.load_run` to build the
model, so every checkpoint already on disk keeps loading unchanged through
both the old function and this new store. The artifact's own typed fields
(``content_hash``, ``router_model_version``, ...) are the authority
:meth:`save` writes into ``model.pt`` itself, alongside the weights, so one
atomic replace covers both; ``run.json`` is written second as a
human-readable mirror, *merged* onto whatever's already there (e.g. the
richer ``history``/``baselines``/``git_sha`` provenance
``training.nirt.train.fit`` writes for its own runs -- that's ``TrainingRun``
territory, not ``RouterModelArtifact``'s job) rather than replacing it.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

from .artifacts import ArtifactFormat, MLPArtifactPayload, NIRTArtifactPayload, RouterModelArtifact

logger = logging.getLogger(__name__)


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

    def _resolve(self, artifact_id: str) -> Path:
        """``root_path / artifact_id``, rejecting an id that would resolve
        outside ``root_path`` (an absolute id, or one containing ``..``) --
        LOCALARTIFACTSTORE-003."""
        candidate = (self.root_path / artifact_id).resolve()
        root = self.root_path.resolve()
        if candidate != root and not candidate.is_relative_to(root):
            raise ValueError(f"artifact_id {artifact_id!r} escapes the store root {root}")
        return self.root_path / artifact_id

    def exists(self, artifact_id: str) -> bool:
        return (self._resolve(artifact_id) / "model.pt").exists()

    def load(self, artifact_id: str) -> RouterModelArtifact:
        import torch

        blob_path = self._resolve(artifact_id) / "model.pt"
        # read the blob once and hand it to whichever loader below needs it,
        # instead of each of {this dispatch, _load_mlp, load_run} separately
        # torch.load-ing the same file (LOCALARTIFACTSTORE-004)
        blob = torch.load(blob_path, map_location="cpu", weights_only=True)
        if blob.get("format") == "mlp":
            return self._load_mlp(artifact_id, blob)
        return self._load_nirt(artifact_id, blob)

    def _meta_for(self, artifact_id: str, blob: dict[str, Any]) -> dict[str, Any]:
        """Metadata embedded in ``model.pt`` by this store's own :meth:`save`
        (authoritative -- see the atomicity note there), falling back to
        ``run.json`` for a checkpoint this store did not write (e.g. one
        ``training.nirt.train.fit`` wrote directly)."""
        embedded = blob.get("_artifact_meta")
        if isinstance(embedded, dict):
            return embedded
        return self._read_run_json(artifact_id)

    @staticmethod
    def _artifact_from_meta(
        *,
        artifact_id: str,
        payload: NIRTArtifactPayload | MLPArtifactPayload,
        meta: dict[str, Any],
        router_model_name: str,
        supported_model_ids: list[str],
        hyperparameters: dict[str, Any],
    ) -> RouterModelArtifact:
        """Shared by both loaders so they cannot drift on which meta fields
        they read back (LOCALARTIFACTSTORE-001)."""
        return RouterModelArtifact(
            artifact_id=artifact_id,
            payload=payload,
            content_hash=meta.get("content_hash", ""),
            # torch.save's zip/pickle container is the closest of the four
            # target formats; NPZ/SAFETENSORS/JSON don't apply to a NIRT run.
            format=ArtifactFormat.PICKLE,
            schema_version=meta.get("schema_version", "0"),
            supported_model_ids=supported_model_ids,
            router_model_name=router_model_name,
            router_model_version=meta.get("router_model_version", ""),
            training_dataset_version=meta.get("training_dataset_version", ""),
            training_run_id=meta.get("training_run_id", artifact_id),
            created_at=meta.get("created_at", ""),
            hyperparameters=hyperparameters,
        )

    def _load_nirt(self, artifact_id: str, blob: dict[str, Any]) -> RouterModelArtifact:
        from ..nirt.checkpoint import load_run

        model, config, model_index = load_run(artifact_id, runs_dir=self.root_path, blob=blob)
        payload = NIRTArtifactPayload(
            state_dict=model.state_dict(),
            config=config,
            n_models=len(model_index),
            model_index=model_index,
            query_dim=model.query_dim,
            profile_dim=model.profile_dim,
        )
        meta = self._meta_for(artifact_id, blob)
        return self._artifact_from_meta(
            artifact_id=artifact_id,
            payload=payload,
            meta=meta,
            router_model_name="nirt",
            supported_model_ids=list(model_index),
            hyperparameters=config,
        )

    def _load_mlp(self, artifact_id: str, blob: dict[str, Any]) -> RouterModelArtifact:
        payload = MLPArtifactPayload.from_state(blob)
        meta = self._meta_for(artifact_id, blob)
        return self._artifact_from_meta(
            artifact_id=artifact_id,
            payload=payload,
            meta=meta,
            router_model_name="mlp",
            supported_model_ids=list(payload.model_ids),
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

        out = self._resolve(artifact.artifact_id)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "artifact_id": artifact.artifact_id,
            "content_hash": artifact.content_hash,
            "schema_version": artifact.schema_version,
            "router_model_name": artifact.router_model_name,
            "router_model_version": artifact.router_model_version,
            "training_dataset_version": artifact.training_dataset_version,
            "training_run_id": artifact.training_run_id,
            "created_at": artifact.created_at,
            "hyperparameters": artifact.hyperparameters,
        }
        # meta rides inside the same atomic model.pt write as the weights, so
        # a load() racing a re-save can never pair new weights with stale
        # metadata the way two independently-replaced files could
        # (LOCALARTIFACTSTORE-005); run.json below is a best-effort,
        # human-readable mirror, not the field load() actually trusts.
        state = dict(artifact.payload.to_state())
        state["_artifact_meta"] = meta
        self._atomic_write(out / "model.pt", lambda tmp: torch.save(state, tmp))
        # merged onto whatever's already there so a richer run.json a training
        # run wrote directly (history/val_metrics/git_sha/...) survives a
        # later store.save() for the same id instead of being replaced
        # wholesale (LOCALARTIFACTSTORE-002)
        merged = {**self._read_run_json(artifact.artifact_id), **meta}
        run_json_text = json.dumps(merged, indent=2)
        self._atomic_write(out / "run.json", lambda tmp: tmp.write_text(run_json_text, encoding="utf-8"))
        return str(out)

    def _read_run_json(self, artifact_id: str) -> dict[str, Any]:
        path = self.root_path / artifact_id / "run.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # distinguishable from "never had metadata" only in the log --
            # returning {} either way keeps load() working, but this makes a
            # corrupt run.json visible instead of silently reading back as
            # clean-but-sparse (LOCALARTIFACTSTORE-006)
            logger.warning("run.json for artifact %r at %s is unreadable: %s", artifact_id, path, e)
            return {}
