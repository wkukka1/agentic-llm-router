# MOD-001: `load_synthetic_tasks` raises on a missing file; its "no file" warning is unreachable

- **Status:** reproduced (`known_bugs/_repros/repro_prompt_decomposition_training.py`, section 3)
- **Severity:** low
- **Where:** `load_synthetic_tasks` ([sources.py:83-90](../../../src/training/prompt_decomposition/data/sources.py#L83-L90))
- **Found:** 2026-09-25

## What

```python
spec = importlib.util.spec_from_file_location(name, path)
if spec is None or spec.loader is None:
    log.warning("no synthetic task file at %s", path)
    continue
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)      # <- FileNotFoundError for a missing file
```

`spec_from_file_location` does not check that the path exists; it returns
`None` only for a suffix it has no loader for. A missing
`data/synthetic/task_<x>.py` therefore gets a valid spec and
`exec_module` raises `FileNotFoundError`, so the intended
warn-and-continue branch never runs. Separately, the paths are relative
(`data/synthetic/...`), so the loader only works from the repo root.

## Suggested fix

`if not Path(path).exists(): log.warning(...); continue` before building the
spec, and resolve `SYNTHETIC_DIR` against the repo root.
