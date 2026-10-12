# LOCALARTIFACTSTORE-004 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Regression test `tests/router/models/test_store.py::test_load_reads_model_pt_exactly_once`

`load()` now calls `torch.load` exactly once and passes the resulting `blob`
into `_load_mlp(artifact_id, blob)` directly, and into
`_load_nirt(artifact_id, blob)`, which forwards it to a new optional `blob`
parameter on `router.nirt.checkpoint.load_run` (defaults to `None`, so every
other caller of `load_run` is unaffected and still does its own
`torch.load`).

## Fix suggested when the bug was filed

Load the blob once, pass it to `_load_mlp` / a `_load_nirt` variant that accepts a blob
(or split `load_run` into read + build).

## Regression tests

`tests/router/models/test_store.py:215` `test_load_reads_model_pt_exactly_once`

```python
def test_load_reads_model_pt_exactly_once(tmp_path, monkeypatch):
    """LOCALARTIFACTSTORE-004: load() must not deserialize the same model.pt
    more than once."""
    store = LocalArtifactStore(tmp_path)
    store.save(_mlp_artifact_versioned("once-run"))

    calls = []
    real_load = torch.load

    def counting_load(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    store.load("once-run")
    assert len(calls) == 1
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
@@ -81,41 +138,25 @@ class LocalArtifactStore(ArtifactStore):
             query_dim=model.query_dim,
             profile_dim=model.profile_dim,
         )
-        meta = self._read_run_json(artifact_id)
-        return RouterModelArtifact(
+        meta = self._meta_for(artifact_id, blob)
+        return self._artifact_from_meta(
             artifact_id=artifact_id,
             payload=payload,
-            content_hash=meta.get("content_hash", ""),
-            # torch.save's zip/pickle container is the closest of the four
-            # target formats; NPZ/SAFETENSORS/JSON don't apply to a NIRT run.
-            format=ArtifactFormat.PICKLE,
-            schema_version=meta.get("schema_version", "0"),
-            supported_model_ids=list(model_index),
+            meta=meta,
             router_model_name="nirt",
-            router_model_version=meta.get("router_model_version", ""),
-            training_dataset_version=meta.get("training_dataset_version", ""),
-            training_run_id=meta.get("training_run_id", artifact_id),
-            created_at=meta.get("created_at", ""),
+            supported_model_ids=list(model_index),
             hyperparameters=config,
         )
 
-    def _load_mlp(self, artifact_id: str) -> RouterModelArtifact:
-        import torch
-
-        blob = torch.load(self.root_path / artifact_id / "model.pt",
-                          map_location="cpu", weights_only=True)
+    def _load_mlp(self, artifact_id: str, blob: dict[str, Any]) -> RouterModelArtifact:
         payload = MLPArtifactPayload.from_state(blob)
-        meta = self._read_run_json(artifact_id)
-        return RouterModelArtifact(
+        meta = self._meta_for(artifact_id, blob)
+        return self._artifact_from_meta(
             artifact_id=artifact_id,
             payload=payload,
-            content_hash=meta.get("content_hash", ""),
-            format=ArtifactFormat.PICKLE,
-            schema_version=meta.get("schema_version", "0"),
-            supported_model_ids=list(payload.model_ids),
+            meta=meta,
             router_model_name="mlp",
-            training_run_id=meta.get("training_run_id", artifact_id),
-            created_at=meta.get("created_at", ""),
+            supported_model_ids=list(payload.model_ids),
             hyperparameters=meta.get("hyperparameters", {}),
         )
 
```

`src/router/nirt/checkpoint.py`

```diff
--- a/src/router/nirt/checkpoint.py
+++ b/src/router/nirt/checkpoint.py
@@ -25,8 +25,14 @@ __all__ = ["load_run"]
 FORMAT = "nirt"
 
 
-def load_run(name: str, runs_dir: Optional[os.PathLike] = None):
-    """Return ``(model, config, model_index)`` for a saved run."""
+def load_run(name: str, runs_dir: Optional[os.PathLike] = None, blob: Optional[dict] = None):
+    """Return ``(model, config, model_index)`` for a saved run.
+
+    ``blob`` lets a caller that already deserialized ``model.pt`` (e.g.
+    :class:`~router.models.store.LocalArtifactStore`, which reads it once to
+    dispatch on ``format``) pass it straight through instead of paying for a
+    second ``torch.load`` of the same file.
+    """
     import torch
 
     if runs_dir is not None:
@@ -36,9 +42,11 @@ def load_run(name: str, runs_dir: Optional[os.PathLike] = None):
 
         base = REPO_ROOT / DEFAULT_NIRT_RUNS_DIR
     path = base / name / "model.pt"
-    # weights_only: the blob is tensors + plain dicts/ints, and a run directory
-    # may be shared/downloaded -- never unpickle arbitrary objects on the serving path
-    blob = torch.load(path, map_location="cpu", weights_only=True)
+    if blob is None:
+        # weights_only: the blob is tensors + plain dicts/ints, and a run
+        # directory may be shared/downloaded -- never unpickle arbitrary
+        # objects on the serving path
+        blob = torch.load(path, map_location="cpu", weights_only=True)
     fmt = blob.get("format")
     if fmt is not None and fmt != FORMAT:
         raise ValueError(
```
