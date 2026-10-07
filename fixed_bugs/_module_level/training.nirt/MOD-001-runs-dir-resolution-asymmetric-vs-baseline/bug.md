# MOD-001: An explicit relative output dir resolves differently in `nirt.train.fit` than in the baseline's `fit`

- **Status:** fixed 2026-09-25, see [fix.md](fix.md)
- **Severity:** medium
- **Where:** `fit()` ([training/nirt/train.py:744](../../../../src/training/nirt/train.py#L744)) vs. `fit()` ([training/nirt/baseline/train.py:231](../../../../src/training/nirt/baseline/train.py#L231))
- **Found:** 2026-09-22

## What

Baseline's `fit()`, given an explicit relative `out_dir`, joins it against
the phase-0 config's root before writing:

```python
out = Path(out_dir or cfg.get("out_dir", _DEFAULT_DIR))
out = out if out.is_absolute() else p0.root / out   # baseline/train.py:230-231
```

NIRT's `fit()`, given an explicit relative `runs_dir`, uses it as-is with no
such join:

```python
base = Path(runs_dir) if runs_dir is not None else _resolve_runs_dir(nirt_cfg, phase0_cfg)   # train.py:744
```

Only the *default* (`runs_dir=None`) branch of NIRT's `fit()` resolves
against `phase0_cfg` (via `_resolve_runs_dir`); an *explicit* relative
`runs_dir` bypasses that and resolves relative to the process's current
working directory instead.

## Failure scenario

The same conceptual call — "train and write output under this relative
directory, relative to my phase-0 config's root" — lands in different
places depending on which of the two sibling `fit()` functions is called.
A caller who learned the convention from the baseline trainer (explicit
relative path resolved against `p0.root`) and applies it to `NIRTTrainer`/
`training.nirt.train.fit` gets output written relative to whatever the
process's CWD happens to be instead — silently, no error, just a
misplaced/duplicated run directory.
