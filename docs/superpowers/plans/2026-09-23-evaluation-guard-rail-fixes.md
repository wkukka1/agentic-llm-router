# Evaluation guard-rail fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two open, `med`-severity `known_bugs/` defects in `evaluation/`: `train_quality`'s `metric` argument silently falls back to the `"quality"` branch for anything that isn't the literal string `"accuracy"` (a typo like `"Accuracy"` gets the wrong quantity with no error), and `compare_routers` reindexes a router's predictions to the comparison's full model pool with no guard, the exact all-NaN-column failure mode its sibling `evaluate_router` already guards against.

**Architecture:** Two independent, single-file patches: `evaluation/nirt/routing.py` (`train_quality`) and `evaluation/routing/oracle.py` (`evaluate_router`/`compare_routers`, factored through one new shared helper). No dependency between them.

**Tech Stack:** Python 3.12, pytest, pandas/numpy.

**Spec:** `fixed_bugs/_module_level/evaluation.nirt.routing/MOD-001-train-quality-silently-ignores-unknown-metric/bug.md`, `fixed_bugs/_module_level/evaluation.routing.oracle/MOD-001-compare-routers-no-missing-model-check/bug.md`.

## Global Constraints

- `train_quality`'s two documented values, `"quality"` (the default) and `"accuracy"`, must keep behaving exactly as before — this is a validation-only addition, not a behavior change for any currently-correct caller.
- `evaluate_router`'s existing missing-model error messages/`match=` patterns in `tests/router/routing/test_routing_interface.py::test_evaluate_raises_when_outcomes_lack_a_pool_model` (`match="GPT-B"` and `match="cost_df.*GPT-C"`) must still match after the guard is factored into a shared helper — do not change the wording so much that these regexes stop matching.
- `compare_routers`'s new guard checks each router's *own* `model_ids` against `true_df.columns` (the comparison's fixed pool) -- the opposite direction from `evaluate_router`'s check (which checks `true_df`/`cost_df`'s columns against the router's pool) -- so the shared helper must take "have" and "need" as separate, clearly-named parameters rather than assuming a fixed direction.

## Review Focus

- `metric="Accuracy"` (a capitalization slip, the exact near-miss the bug file calls out) must raise, not silently take the `"quality"` branch. (Task 1)
- `compare_routers` called with all routers' pools fully covering `true_df.columns` (the common, already-passing case, exercised by every existing `compare_routers` test) must be completely unaffected by the new guard. (Task 2)
- `compare_routers`'s duplicate-router-name check (`test_compare_routers_rejects_duplicate_names`) runs before the per-router prediction loop today -- the new missing-model guard must not change which error a caller sees first when both problems are present (duplicate name check stays first, since it's already first in the loop and doesn't need the model list). (Task 2)

---

## Task 1: `train_quality` rejects an unrecognized `metric`

**Files:**
- Modify: `src/evaluation/nirt/routing.py` (`train_quality`, ~L115-125)
- Test: `tests/evaluation/nirt/test_nirt_routing.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `train_quality(data, models, metric="quality") -> dict[str, float]` (unchanged signature) now raises `ValueError` when `metric` isn't `"quality"` or `"accuracy"`, instead of silently treating any other value as `"quality"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/evaluation/nirt/test_nirt_routing.py`:

```python
def test_train_quality_rejects_an_unrecognized_metric():
    """MOD-001 (evaluation.nirt.routing): a metric that isn't "quality" or
    "accuracy" must raise, not silently fall back to the "quality" branch --
    a capitalization slip like "Accuracy" must not compare the wrong quantity
    with no signal that anything was wrong."""
    from evaluation.nirt.routing import train_quality

    d = FakeTrainingData(make_split_obs(seed=11))
    model_ids = sorted(d.observations["model_id"].unique())

    with pytest.raises(ValueError, match="metric"):
        train_quality(d, model_ids, metric="Accuracy")
```

(Check the top of the file for the existing `FakeTrainingData`/`make_split_obs` import from `helpers` and `import pytest` — reuse them, don't re-import.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/evaluation/nirt/test_nirt_routing.py::test_train_quality_rejects_an_unrecognized_metric -v`
Expected: FAIL — `train_quality` returns a dict instead of raising.

- [ ] **Step 3: Implement**

In `src/evaluation/nirt/routing.py`:

```python
_TRAIN_QUALITY_METRICS = frozenset({"quality", "accuracy"})


def train_quality(data, models: list[str], metric: str = "quality") -> dict[str, float]:
    """Per-model mean training correctness -- for the 'best fixed model' baseline.

    ``metric="quality"`` -> mean graded target; ``"accuracy"`` -> mean(target>=0.5).
    """
    if metric not in _TRAIN_QUALITY_METRICS:
        raise ValueError(
            f"train_quality: metric must be one of {sorted(_TRAIN_QUALITY_METRICS)}, got {metric!r}"
        )
    obs = data.nirt_observations()
    obs = obs[(obs["split"] == "train") & (obs["model_id"].isin(models))]
    y = obs["target"].to_numpy(np.float64)
    if metric == "accuracy":
        y = (y >= 0.5).astype(np.float64)
    return pd.Series(y).groupby(obs["model_id"].to_numpy()).mean().to_dict()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/evaluation/nirt/test_nirt_routing.py -v`
Expected: PASS, all tests in the file (including `test_routing_report_and_lambda_tradeoff`, which calls `train_quality(d, model_ids)` with the default `metric="quality"` — unaffected).

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/nirt/routing.py tests/evaluation/nirt/test_nirt_routing.py
git commit -m "fix(evaluation): train_quality rejects an unrecognized metric instead of silently defaulting (MOD-001 evaluation.nirt.routing)"
```

---

## Task 2: `compare_routers` guards against a router pool missing a comparison model

**Files:**
- Modify: `src/evaluation/routing/oracle.py` (new helper near `_TOL`/`evaluate_router` ~L57-93, `compare_routers` ~L669-679)
- Test: `tests/router/routing/test_routing_interface.py`

**Interfaces:**
- Consumes: nothing new (uses `RouterModel.model_ids`, `pd.DataFrame.columns`, already used).
- Produces: a new module-private helper `_require_pool_covered(label: str, have: Sequence[str], need: Sequence[str], *, reason: str) -> None` in `evaluation/routing/oracle.py`, raising `ValueError(f"{label} has no column for {missing}; {reason}")` when `set(need) - set(have)` is non-empty. `evaluate_router` calls it once per frame (replacing its inline loop body); `compare_routers` calls it once per router (new).

- [ ] **Step 1: Write the failing test**

Add to `tests/router/routing/test_routing_interface.py`, near `test_compare_routers_rejects_duplicate_names`:

```python
def test_compare_routers_raises_when_a_routers_pool_is_missing_a_model():
    """MOD-001 (evaluation.routing.oracle): compare_routers must guard the
    same reindex-introduces-an-all-NaN-column failure mode evaluate_router
    already guards (test_evaluate_raises_when_outcomes_lack_a_pool_model)."""
    true_df, cost_df = _mats()  # columns GPT-A, GPT-B, GPT-C
    partial_pool_router = MatrixRouter(true_df[["GPT-A", "GPT-C"]], name="partial")

    with pytest.raises(ValueError, match="GPT-B"):
        compare_routers([partial_pool_router], true_df, cost_df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/router/routing/test_routing_interface.py::test_compare_routers_raises_when_a_routers_pool_is_missing_a_model -v`
Expected: FAIL — no `ValueError` raised; `compare_routers` runs to completion with an all-NaN `"GPT-B"` column for `partial_pool_router` instead.

- [ ] **Step 3: Implement**

In `src/evaluation/routing/oracle.py`, add the shared helper right before `evaluate_router` (~L59, after `_TOL = 1e-9`):

```python
def _require_pool_covered(label: str, have: Sequence[str], need: Sequence[str], *, reason: str) -> None:
    """Raise if ``need`` isn't fully covered by ``have`` -- reindexing to a
    model that's missing would add an all-NaN column and silently turn every
    regret / hit rate into NaN. Shared by :func:`evaluate_router` (checks
    ``true_df``/``cost_df``'s columns against the router's pool) and
    :func:`compare_routers` (checks each router's own pool against the
    comparison's fixed ``model_ids`` -- the opposite direction, MOD-001)."""
    missing = sorted(set(need) - set(have))
    if missing:
        raise ValueError(f"{label} has no column for {missing}; {reason}")
```

Replace `evaluate_router`'s existing inline guard (~L84-93):

```python
    model_ids = list(router.model_ids)
    for name, frame in (("true_df", true_df), ("cost_df", cost_df)):
        if frame is not None:
            _require_pool_covered(
                name, frame.columns, model_ids,
                reason="restrict the router's pool or supply their outcomes",
            )
```

In `compare_routers` (~L669-679), add the guard before the per-router prediction loop uses `mat = r.aligned_scores(...)`:

```python
    model_ids = list(true_df.columns)
    query_ids = list(true_df.index)
    preds: dict[str, np.ndarray] = {}
    for r in routers:
        if r.name in preds:
            raise ValueError(
                f"compare_routers: duplicate router name {r.name!r}; give each "
                "router passed in a distinct .name"
            )
        _require_pool_covered(
            f"router {r.name!r}'s pool", r.model_ids, model_ids,
            reason="restrict true_df/cost_df to the router's pool or give it those models",
        )
        mat = r.aligned_scores(query_ids).reindex(columns=model_ids)
        preds[r.name] = mat.to_numpy(float)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/router/routing/test_routing_interface.py -v -k "evaluate or compare_routers"`
Expected: PASS, all of them, including the pre-existing `test_evaluate_raises_when_outcomes_lack_a_pool_model` (its `match="GPT-B"`/`match="cost_df.*GPT-C"` regexes still match the refactored message) and `test_compare_routers_rejects_duplicate_names` (still raises on the duplicate-name check first, since that check runs before the new guard in the loop).

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/routing/oracle.py tests/router/routing/test_routing_interface.py
git commit -m "fix(evaluation): compare_routers guards a router pool missing a comparison model, like evaluate_router already does (MOD-001 evaluation.routing.oracle)"
```

---

## Final step: update `known_bugs/`

Update status to `fixed` in:
- `fixed_bugs/_module_level/evaluation.nirt.routing/MOD-001-train-quality-silently-ignores-unknown-metric/bug.md`
- `fixed_bugs/_module_level/evaluation.routing.oracle/MOD-001-compare-routers-no-missing-model-check/bug.md`

And their rows in `known_bugs/README.md`'s `## Bugs` table.
