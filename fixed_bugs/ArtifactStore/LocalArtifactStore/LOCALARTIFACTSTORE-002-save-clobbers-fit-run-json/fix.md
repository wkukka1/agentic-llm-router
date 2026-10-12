# LOCALARTIFACTSTORE-002 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Regression test `tests/router/models/test_store.py::test_save_merges_onto_an_existing_fit_style_run_json_instead_of_replacing_it`

`save()` now reads whatever `run.json` already exists (via `_read_run_json`) and
merges the store's own fields onto it (`{**existing, **meta}`) before writing,
instead of replacing the file outright. The store's typed fields have no key
overlap with `fit()`'s (`history`, `val_metrics`, `git_sha`, `best_epoch`,
`config`, ...), so both sets of fields now coexist. `run.json` is also no
longer the authority `load()` depends on for correctness -- see
LOCALARTIFACTSTORE-005 -- so a future field collision would only affect the
human-readable mirror, not loading.

## Fix suggested when the bug was filed

Write the store's sidecar under a different name (`artifact.json`), or merge into the
existing `run.json` instead of replacing it, or refuse to overwrite an existing
`run.json` not written by the store.

## Regression tests

`tests/router/models/test_store.py:185` `test_save_merges_onto_an_existing_fit_style_run_json_instead_of_replacing_it`

```python
def test_save_merges_onto_an_existing_fit_style_run_json_instead_of_replacing_it(tmp_path):
    """LOCALARTIFACTSTORE-002: fit()'s run.json (history/val_metrics/git_sha)
    must survive a later store.save() for the same artifact id."""
    import json

    run_dir = tmp_path / "fit-run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "git_sha": "abc123", "history": [{"epoch": 0, "loss": 1.0}], "best_epoch": 0,
    }))

    store = LocalArtifactStore(tmp_path)
    store.save(_mlp_artifact_versioned("fit-run"))

    on_disk = json.loads((run_dir / "run.json").read_text())
    assert on_disk["git_sha"] == "abc123"
    assert on_disk["history"] == [{"epoch": 0, "loss": 1.0}]
    assert on_disk["router_model_version"] == "v2"  # store's own fields still land
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/models/store.py`

```diff
--- a/src/router/models/store.py
+++ b/src/router/models/store.py
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
