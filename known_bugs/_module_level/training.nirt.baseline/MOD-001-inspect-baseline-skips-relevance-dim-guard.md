# MOD-001: `inspect_baseline.py` bypasses the stale-taxonomy guard

- **Status:** confirmed (by reading; no runtime repro)
- **Severity:** low
- **Where:** [scripts/nirt/baseline/inspect_baseline.py:40](../../../scripts/nirt/baseline/inspect_baseline.py#L40)
- **Not a class:** a script calling free functions in `training.nirt.baseline`, so it lives
  under `_module_level/`.
- **Found:** 2026-09-19 (first reported by the earlier diff review; re-checked top-down)

## What

`build_arrays` ([data.py:118](../../../src/training/nirt/baseline/data.py#L118)) got a
`relevance_dim` check that raises a clear `ValueError` when the relevance store was rebuilt
with a different cluster count than the checkpoint was trained on. `eval.py` passes
`relevance_dim=model.relevance_dim` ([eval.py:311](../../../src/training/nirt/baseline/eval.py#L311)).
`inspect_baseline.py` calls `build_arrays(...)` with `use_relevance=model.use_relevance` but
no `relevance_dim`, so the check is skipped.

## Failure scenario

Inspecting a checkpoint trained before the taxonomy was re-clustered (for example 39 vs 154
clusters) hits the bare matmul shape error inside the model, which is what the guard was
added to prevent.

## Suggested fix

Pass `relevance_dim=model.relevance_dim if model.use_relevance else None`.
