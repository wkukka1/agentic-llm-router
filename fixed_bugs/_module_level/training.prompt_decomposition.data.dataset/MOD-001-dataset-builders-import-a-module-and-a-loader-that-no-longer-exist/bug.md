# MOD-001: `build_real_only_dataset` and `build_task_dataset` cannot run (stale import; deleted loader)

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** high (the only path that rebuilds the domain / task head's train / val / test splits)
- **Where:** [dataset.py:217](../../../../src/training/prompt_decomposition/data/dataset.py#L217) and [dataset.py:268](../../../../src/training/prompt_decomposition/data/dataset.py#L268) (import); [dataset.py:270](../../../../src/training/prompt_decomposition/data/dataset.py#L270) (call). Sole callers: `prompt-decomposition build-real` / `build-task` ([scripts/prompt_decomposition_cli.py:59-70](../../../../scripts/prompt_decomposition_cli.py#L59-L70))
- **Found:** 2026-09-25

## What

Two independent defects, both fatal:

1. Both builders do `from decompose.classifiers.prompt_decomposition import sources`.
   The loaders live in `training.prompt_decomposition.data.sources`; no
   `decompose/.../sources.py` exists (a leftover from the "Nest the subsystem"
   / "refactor the heads" moves). Every call raises
   `ImportError: cannot import name 'sources'`.
2. `build_real_only_dataset` then calls `sources.load_handlabelled()`, which
   does not exist. It was deleted in `ea4d209` ("Deslop: delete three dead
   paths") while this caller stayed. `HANDLABELLED_PATH` is still defined at
   [sources.py:33](../../../../src/training/prompt_decomposition/data/sources.py#L33) with
   no reader, which is the tell. So even with (1) fixed, this builder still
   raises `AttributeError`. `build_task_dataset` only needs (1): with the
   import aliased it builds the documented splits (train 1,430 / val 149 /
   test 201; `include_agent` adds 1,201 `agent_labelled` rows; `include_mined`
   adds 240), so the rest of it is healthy.

Nothing catches this: `tests/prompt_decomposition/test_data.py` covers the
helpers (`dedupe`, `split_frame`, `assert_no_leakage`, the frozen eval set) but
no test calls either builder, and the import is function-local, so a plain
`import training.prompt_decomposition.data.dataset` succeeds. A scan of every
`from <first-party> import <name>` in `src/`, `scripts/` and `tests/`
(`scan_stale_imports.py`) finds exactly these two and nothing else, but it
cannot see attribute misses like (2).

## Impact

The shipped domain / task heads cannot be retrained from source: the frozen
splits under `data/processed/*` are the only copy that works. Any change to the
hand-labelled pool (new labels, relabelling) cannot be turned into a train set.

## Related staleness (same root cause)

* [dataset.py:155](../../../../src/training/prompt_decomposition/data/dataset.py#L155) tells the user to run `router build-data`; the console script is `prompt-decomposition` (`pyproject.toml`) and its subcommands are `build-real` / `build-task`, so the command named in the error does not exist (the CLI module's own docstring still lists `build-data` too). The same stale `router ...` name is in [overfit.py:10](../../../../src/evaluation/prompt_decomposition/overfit.py#L10) (`router overfit`) and [settings.py:57](../../../../src/decompose/classifiers/prompt_decomposition/settings.py#L57) (`router describe-data`).
* `sources.py`'s module docstring still describes loaders for LMArena flags, MMLU-Pro, RouterArena and BIG-bench that are gone.
