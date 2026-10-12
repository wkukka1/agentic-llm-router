"""Repros for LocalArtifactStore / RouterModelArtifact. Run from repo root."""
import sys, json, tempfile, pathlib
sys.path.insert(0, "src")
import torch
from router.models.artifacts import (ArtifactFormat, MLPArtifactPayload,
                                     NIRTArtifactPayload, RouterModelArtifact)
from router.models.store import LocalArtifactStore

root = pathlib.Path(tempfile.mkdtemp()) / "runs"
store = LocalArtifactStore(root)

print("== LOCALARTIFACTSTORE-001: MLP round trip drops fields ==")
p = MLPArtifactPayload(state_dict={"w": torch.zeros(1)}, model_ids=["a", "b"], in_dim=3, hidden=4, dropout=0.1)
art = RouterModelArtifact(
    "mlp1", p, "abc123", ArtifactFormat.PICKLE, schema_version="3", router_model_name="mlp",
    router_model_version="v2", training_dataset_version="ds-7", training_run_id="run-9",
    created_at="2026-01-01T00:00:00+00:00", hyperparameters={"pathway": "bert"})
store.save(art)
back = store.load("mlp1")
for f in ("content_hash", "schema_version", "router_model_version", "training_dataset_version",
          "training_run_id", "created_at", "hyperparameters"):
    a, b = getattr(art, f), getattr(back, f)
    print(f"{f:26s} saved={a!r:34} loaded={b!r:34} {'OK' if a == b else 'LOST'}")

print("== LOCALARTIFACTSTORE-002: save clobbers fit()'s richer run.json ==")
d = root / "nirt1"; d.mkdir(parents=True)
(d / "run.json").write_text(json.dumps({"git_sha": "deadbeef", "best_epoch": 7,
                                        "history": [{"epoch": 0}], "val_metrics": {"bce": 0.5}}))
nirt = NIRTArtifactPayload(state_dict={}, config={}, n_models=1, model_index={"a": 0}, query_dim=2, profile_dim=2)
store.save(RouterModelArtifact("nirt1", nirt, "", ArtifactFormat.PICKLE, router_model_name="nirt"))
print("run.json keys after save:", sorted(json.loads((d / "run.json").read_text())))

print("== LOCALARTIFACTSTORE-003: artifact_id escapes root_path ==")
store.save(RouterModelArtifact("../escaped", p, "", ArtifactFormat.PICKLE, router_model_name="mlp"))
print("outside root:", (root.parent / "escaped" / "model.pt").exists())
