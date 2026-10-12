"""Package a kNN router as one small file the gateway serves on its own.

A :class:`~router.routing.routers.KNNRouter` normally reads the whole training
dataset to find a prompt's nearest training queries. Everything it actually
uses is two arrays -- the training queries' embeddings and their per-model
scores -- so :func:`export_knn_artifact` writes just those
(:class:`~router.models.artifacts.KNNArtifactPayload`): ~7 MB for
LLMRouterBench instead of gigabytes, served with numpy and an encoder that
needs no PyTorch.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np

from router.embeddings.encoder import load_encoder
from router.models.artifacts import KNNArtifactPayload
from router.nirt.baselines_infer import train_correctness_matrix
from router.routing.routers import _mean_train_costs


def export_knn_artifact(data, *, k: int = 25, pathway: str = "retrieval",
                        dataset: str = "") -> tuple[KNNArtifactPayload, dict]:
    """``(payload, header)`` for ``data``'s train split -- the same neighbours
    and scores ``KNNRouter(data, k=k, pathway=pathway)`` routes on."""
    C, model_ids, query_ids = train_correctness_matrix(data, pathway=pathway)
    scores = np.where(np.isnan(C), np.nanmean(C, axis=0), C).astype(np.float32)
    embeddings = np.asarray(data.query_embeddings(pathway).gather(list(query_ids)), dtype=np.float32)
    enc = load_encoder(data.cfg, pathway).cfg
    costs = _mean_train_costs(data, model_ids)
    payload = KNNArtifactPayload(
        embeddings=embeddings, scores=scores, model_ids=[str(m) for m in model_ids], k=int(k),
        encoder={"backend": enc.backend, "model_name": enc.model_name, "normalize": bool(enc.normalize),
                 "max_seq_length": int(enc.max_seq_length or 0)},
        model_costs=[float(c) for c in costs] if costs is not None else None)
    digest = hashlib.sha256(embeddings.astype(np.float16).tobytes() + scores.tobytes()).hexdigest()[:16]
    header = {"artifact_id": f"knn-{dataset or 'pool'}-k{k}-{digest}", "content_hash": digest,
              "dataset": dataset, "n_queries": int(len(query_ids)), "pathway": pathway,
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return payload, header
