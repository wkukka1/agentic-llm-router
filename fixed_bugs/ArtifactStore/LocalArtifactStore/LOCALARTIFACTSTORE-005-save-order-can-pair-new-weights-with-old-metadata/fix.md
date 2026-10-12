# LOCALARTIFACTSTORE-005 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Regression test `tests/router/models/test_store.py::test_reload_after_resave_never_sees_new_weights_with_stale_metadata`

Took the second option: `save()` now embeds the artifact's typed metadata
(`content_hash`, `router_model_version`, `training_dataset_version`, ...)
inside the `model.pt` blob itself, under a reserved `_artifact_meta` key,
before the single atomic `torch.save`/`os.replace` for that file. `load()`
reads `_artifact_meta` back out of the blob it already deserialized and only
falls back to `_read_run_json` when that key is absent (a checkpoint this
store never saved, e.g. one `training.nirt.train.fit` wrote directly).
`run.json` is still written afterward, but purely as a best-effort,
human-readable mirror -- since `load()` no longer trusts it for correctness,
the re-save race this bug describes can no longer produce a mismatched pair:
one atomic replace now carries both the weights and their metadata.

## Fix suggested when the bug was filed

Write each save into a new versioned subdirectory and switch a pointer (or rename the
directory) once both files are complete, or store the metadata inside `model.pt` so one
atomic replace covers both.

## Regression tests

`tests/router/models/test_store.py:233` `test_reload_after_resave_never_sees_new_weights_with_stale_metadata`

```python
def test_reload_after_resave_never_sees_new_weights_with_stale_metadata(tmp_path):
    """LOCALARTIFACTSTORE-005: metadata rides inside the same atomic model.pt
    write as the weights, so a load() right after a re-save must reflect the
    new save's metadata even if run.json is deleted/stale."""
    store = LocalArtifactStore(tmp_path)
    store.save(_mlp_artifact_versioned("resave-run"))

    resaved = RouterModelArtifact(
        artifact_id="resave-run",
        payload=MLPArtifactPayload(
            state_dict={"w": torch.zeros(2)}, model_ids=["a", "b"], in_dim=2, hidden=2, dropout=0.0,
        ),
        content_hash="h2", format=ArtifactFormat.PICKLE, router_model_name="mlp",
        router_model_version="v3", training_dataset_version="ds-8",
    )
    store.save(resaved)
    (tmp_path / "resave-run" / "run.json").unlink()  # remove the mirror entirely

    reloaded = store.load("resave-run")
    assert reloaded.content_hash == "h2"
    assert reloaded.router_model_version == "v3"
    assert torch.equal(reloaded.payload.state_dict["w"], torch.zeros(2))
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
 
@@ -140,25 +181,33 @@ class LocalArtifactStore(ArtifactStore):
             )
         import torch
 
-        out = self.root_path / artifact.artifact_id
+        out = self._resolve(artifact.artifact_id)
         out.mkdir(parents=True, exist_ok=True)
-        self._atomic_write(out / "model.pt", lambda tmp: torch.save(artifact.payload.to_state(), tmp))
-        run_json_text = json.dumps(
-            {
-                "artifact_id": artifact.artifact_id,
-                "content_hash": artifact.content_hash,
-                "schema_version": artifact.schema_version,
-                "router_model_name": artifact.router_model_name,
-                "router_model_version": artifact.router_model_version,
-                "training_dataset_version": artifact.training_dataset_version,
-                "training_run_id": artifact.training_run_id,
-                "created_at": artifact.created_at,
-                "hyperparameters": artifact.hyperparameters,
-            },
-            indent=2,
-        )
-        # written last: a reader never sees a complete model.pt paired with a
-        # missing/stale run.json from an interrupted write (RM-01)
+        meta = {
+            "artifact_id": artifact.artifact_id,
+            "content_hash": artifact.content_hash,
+            "schema_version": artifact.schema_version,
+            "router_model_name": artifact.router_model_name,
+            "router_model_version": artifact.router_model_version,
+            "training_dataset_version": artifact.training_dataset_version,
+            "training_run_id": artifact.training_run_id,
+            "created_at": artifact.created_at,
+            "hyperparameters": artifact.hyperparameters,
+        }
+        # meta rides inside the same atomic model.pt write as the weights, so
+        # a load() racing a re-save can never pair new weights with stale
+        # metadata the way two independently-replaced files could
+        # (LOCALARTIFACTSTORE-005); run.json below is a best-effort,
+        # human-readable mirror, not the field load() actually trusts.
+        state = dict(artifact.payload.to_state())
+        state["_artifact_meta"] = meta
+        self._atomic_write(out / "model.pt", lambda tmp: torch.save(state, tmp))
+        # merged onto whatever's already there so a richer run.json a training
+        # run wrote directly (history/val_metrics/git_sha/...) survives a
+        # later store.save() for the same id instead of being replaced
+        # wholesale (LOCALARTIFACTSTORE-002)
+        merged = {**self._read_run_json(artifact.artifact_id), **meta}
+        run_json_text = json.dumps(merged, indent=2)
         self._atomic_write(out / "run.json", lambda tmp: tmp.write_text(run_json_text, encoding="utf-8"))
         return str(out)
 
```
