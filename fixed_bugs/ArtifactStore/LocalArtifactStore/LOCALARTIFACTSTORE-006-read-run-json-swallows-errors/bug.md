# LOCALARTIFACTSTORE-006: A corrupt `run.json` reads back as "no metadata", not an error

- **Status:** fixed 2026-09-23, see [fix.md](fix.md)
- **Severity:** low-medium
- **Class:** `LocalArtifactStore` ([router/models/store.py:165-172](../../../../src/router/models/store.py#L165-L172))
- **Found:** 2026-09-22

## What

```python
try:
    return json.loads(path.read_text(encoding="utf-8"))
except (json.JSONDecodeError, OSError):
    return {}
```

`_read_run_json` catches `JSONDecodeError`/`OSError` and returns `{}` with no
logging, no warning — indistinguishable from a `run.json` that genuinely
never had metadata (which `save()`'s own atomic-write discipline,
[RM-01](../LOCALARTIFACTSTORE-002-save-clobbers-fit-run-json/bug.md)'s fix, is
supposed to make rare, not impossible on e.g. disk corruption or an
out-of-space partial write that predates the atomic-write fix).

## Failure scenario

A truncated or corrupted `run.json` (disk fault, manual edit, an artifact
written before the atomic-write fix landed) silently loads as an artifact
with every provenance field defaulted — `created_at=""`,
`hyperparameters={}`, etc. — with nothing distinguishing "corrupted" from
"never had this data", making the corruption invisible until someone notices
the missing values.
