# LOCALARTIFACTSTORE-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Regression test `tests/router/models/test_store.py::test_mlp_roundtrip_keeps_router_model_version_and_training_dataset_version`

Both loaders now build their `RouterModelArtifact` through one shared
`_artifact_from_meta` helper ([store.py](../../../../src/router/models/store.py)),
so `_load_mlp` reads every meta field `_load_nirt` does, including
`router_model_version` and `training_dataset_version`. Landed together with
LOCALARTIFACTSTORE-002/-004/-005/-006 in the same pass over `store.py`.

## Fix suggested when the bug was filed

Read the two fields from `meta` in `_load_mlp`, as `_load_nirt` does. Better, share one
`_artifact_from(meta, payload, ...)` helper so the two loaders cannot drift again.

## Regression tests

`tests/router/models/test_store.py:175` `test_mlp_roundtrip_keeps_router_model_version_and_training_dataset_version`

```python
def test_mlp_roundtrip_keeps_router_model_version_and_training_dataset_version(tmp_path):
    """LOCALARTIFACTSTORE-001: these two provenance fields must survive an
    MLP save/load round trip, not come back as ''."""
    store = LocalArtifactStore(tmp_path)
    store.save(_mlp_artifact_versioned("mlp-run"))
    reloaded = store.load("mlp-run")
    assert reloaded.router_model_version == "v2"
    assert reloaded.training_dataset_version == "ds-7"
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/models/store.py`

```diff
--- a/src/router/models/store.py
+++ b/src/router/models/store.py
@@ -57,22 +62,74 @@ class LocalArtifactStore(ArtifactStore):
     def __init__(self, root_path: str | Path):
         self.root_path = Path(root_path)
 
+    def _resolve(self, artifact_id: str) -> Path:
+        """``root_path / artifact_id``, rejecting an id that would resolve
+        outside ``root_path`` (an absolute id, or one containing ``..``) --
+        LOCALARTIFACTSTORE-003."""
+        candidate = (self.root_path / artifact_id).resolve()
+        root = self.root_path.resolve()
+        if candidate != root and not candidate.is_relative_to(root):
+            raise ValueError(f"artifact_id {artifact_id!r} escapes the store root {root}")
+        return self.root_path / artifact_id
+
     def exists(self, artifact_id: str) -> bool:
-        return (self.root_path / artifact_id / "model.pt").exists()
+        return (self._resolve(artifact_id) / "model.pt").exists()
 
     def load(self, artifact_id: str) -> RouterModelArtifact:
         import torch
 
-        blob_path = self.root_path / artifact_id / "model.pt"
-        fmt = torch.load(blob_path, map_location="cpu", weights_only=True).get("format")
-        if fmt == "mlp":
-            return self._load_mlp(artifact_id)
-        return self._load_nirt(artifact_id)
+        blob_path = self._resolve(artifact_id) / "model.pt"
+        # read the blob once and hand it to whichever loader below needs it,
+        # instead of each of {this dispatch, _load_mlp, load_run} separately
+        # torch.load-ing the same file (LOCALARTIFACTSTORE-004)
+        blob = torch.load(blob_path, map_location="cpu", weights_only=True)
+        if blob.get("format") == "mlp":
+            return self._load_mlp(artifact_id, blob)
+        return self._load_nirt(artifact_id, blob)
+
+    def _meta_for(self, artifact_id: str, blob: dict[str, Any]) -> dict[str, Any]:
+        """Metadata embedded in ``model.pt`` by this store's own :meth:`save`
+        (authoritative -- see the atomicity note there), falling back to
+        ``run.json`` for a checkpoint this store did not write (e.g. one
+        ``training.nirt.train.fit`` wrote directly)."""
+        embedded = blob.get("_artifact_meta")
+        if isinstance(embedded, dict):
+            return embedded
+        return self._read_run_json(artifact_id)
+
+    @staticmethod
+    def _artifact_from_meta(
+        *,
+        artifact_id: str,
+        payload: NIRTArtifactPayload | MLPArtifactPayload,
+        meta: dict[str, Any],
+        router_model_name: str,
+        supported_model_ids: list[str],
+        hyperparameters: dict[str, Any],
+    ) -> RouterModelArtifact:
+        """Shared by both loaders so they cannot drift on which meta fields
+        they read back (LOCALARTIFACTSTORE-001)."""
+        return RouterModelArtifact(
+            artifact_id=artifact_id,
+            payload=payload,
+            content_hash=meta.get("content_hash", ""),
+            # torch.save's zip/pickle container is the closest of the four
+            # target formats; NPZ/SAFETENSORS/JSON don't apply to a NIRT run.
+            format=ArtifactFormat.PICKLE,
+            schema_version=meta.get("schema_version", "0"),
+            supported_model_ids=supported_model_ids,
+            router_model_name=router_model_name,
+            router_model_version=meta.get("router_model_version", ""),
+            training_dataset_version=meta.get("training_dataset_version", ""),
+            training_run_id=meta.get("training_run_id", artifact_id),
+            created_at=meta.get("created_at", ""),
+            hyperparameters=hyperparameters,
+        )
 
-    def _load_nirt(self, artifact_id: str) -> RouterModelArtifact:
+    def _load_nirt(self, artifact_id: str, blob: dict[str, Any]) -> RouterModelArtifact:
         from ..nirt.checkpoint import load_run
 
-        model, config, model_index = load_run(artifact_id, runs_dir=self.root_path)
+        model, config, model_index = load_run(artifact_id, runs_dir=self.root_path, blob=blob)
         payload = NIRTArtifactPayload(
             state_dict=model.state_dict(),
             config=config,
```
