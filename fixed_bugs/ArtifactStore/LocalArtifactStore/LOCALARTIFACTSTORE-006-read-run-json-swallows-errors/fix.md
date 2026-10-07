# LOCALARTIFACTSTORE-006 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Regression test `tests/router/models/test_store.py::test_corrupt_run_json_logs_a_warning_instead_of_reading_as_clean`

`_read_run_json` now logs `logger.warning(...)` with the artifact id, the
path, and the caught exception before returning `{}`. Note `run.json` is no
longer the metadata `load()` trusts for a store-written artifact
(LOCALARTIFACTSTORE-005 embeds it in `model.pt` instead), so a corrupt
`run.json` today only degrades the human-readable mirror for store-written
artifacts -- but it is still the sole metadata source for a checkpoint
`training.nirt.train.fit` wrote directly, where this warning remains load-bearing.

## Fix suggested when the bug was filed

Log a warning (with the artifact id and the exception) before returning
`{}`, so a corrupted `run.json` is at least visible in logs rather than
silently presenting as clean-but-sparse metadata.

## Regression tests

`tests/router/models/test_store.py:257` `test_corrupt_run_json_logs_a_warning_instead_of_reading_as_clean`

```python
def test_corrupt_run_json_logs_a_warning_instead_of_reading_as_clean(tmp_path, caplog):
    """LOCALARTIFACTSTORE-006"""
    run_dir = tmp_path / "corrupt-run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{not valid json")

    store = LocalArtifactStore(tmp_path)
    with caplog.at_level("WARNING"):
        meta = store._read_run_json("corrupt-run")
    assert meta == {}
    assert any("corrupt-run" in r.message for r in caplog.records)
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/models/store.py`

```diff
--- a/src/router/models/store.py
+++ b/src/router/models/store.py
@@ -24,6 +27,8 @@ from typing import Any, Callable
 
 from .artifacts import ArtifactFormat, MLPArtifactPayload, NIRTArtifactPayload, RouterModelArtifact
 
+logger = logging.getLogger(__name__)
+
 
 class ArtifactStore(abc.ABC):
     @abc.abstractmethod
@@ -168,5 +217,10 @@ class LocalArtifactStore(ArtifactStore):
             return {}
         try:
             return json.loads(path.read_text(encoding="utf-8"))
-        except (json.JSONDecodeError, OSError):
+        except (json.JSONDecodeError, OSError) as e:
+            # distinguishable from "never had metadata" only in the log --
+            # returning {} either way keeps load() working, but this makes a
+            # corrupt run.json visible instead of silently reading back as
+            # clean-but-sparse (LOCALARTIFACTSTORE-006)
+            logger.warning("run.json for artifact %r at %s is unreadable: %s", artifact_id, path, e)
             return {}
```
