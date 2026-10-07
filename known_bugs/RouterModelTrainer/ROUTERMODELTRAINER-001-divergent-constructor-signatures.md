# ROUTERMODELTRAINER-001: Concrete trainers don't share a constructor contract

- **Status:** confirmed (traced; code_review finding TT-05)
- **Severity:** low (design consistency, not a runtime crash by itself)
- **Class:** `RouterModelTrainer` ABC (`src/training/trainers/base.py`) and its three concrete trainers
- **Found:** 2026-09-22

## What

`RouterModelTrainer` defines no `__init__` of its own, so nothing constrains
subclass constructors to agree. In practice they don't:

- `NIRTTrainer(runs_dir: Optional[os.PathLike] = None)` ([nirt/trainer.py:29](../../src/training/nirt/trainer.py#L29))
- `BaselineNIRTTrainer(runs_dir: Optional[str] = None)` ([nirt/baseline/trainer.py:35](../../src/training/nirt/baseline/trainer.py#L35))
- `MLPRouterTrainer(runs_dir: os.PathLike)` — **required**, no default ([trainers/mlp_router_trainer.py:32](../../src/training/trainers/mlp_router_trainer.py#L32))

## Failure scenario

Code written generically against `RouterModelTrainer` (e.g. a factory or a
loop that instantiates trainers by name with no arguments) works for two of
the three trainers and raises `TypeError: missing 1 required positional
argument` for `MLPRouterTrainer` alone.

## Suggested fix

Give `RouterModelTrainer` an `__init__(self, runs_dir: Optional[os.PathLike] = None)`
that subclasses call via `super().__init__(runs_dir)`, so all three agree on
an optional `runs_dir` with the same default-resolution behavior.
