# Bug-hunt fixes (2026-09-24 / 25) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 12 open, not-yet-started defects from the 2026-09-24 / 25 bug hunt (one high, the rest medium to low), each with a failing-first regression test, verify the one fix already in flight, and close out the `known_bugs/` entries.

**Architecture:** Eleven small, mostly independent patches, each confined to one module plus its test file, ordered by severity; Task 3 is verification-only (plus one optional follow-up) because its fix is already in the working tree. Task 1 (dead dataset builders) and Task 10 (two more prompt-decomposition fixes) share files and must run in that order; Task 12 is gated on a decision; Task 13 closes out. No task adds a new abstraction except two tiny helpers (`preserve_rng_state`, `correlation_ceiling`).

**Tech Stack:** Python 3.12, pytest, numpy / pandas, PyTorch and FAISS (each test that needs them uses `pytest.importorskip`), import-linter.

**Spec:** [`docs/bug_hunt_2026-09.md`](../../bug_hunt_2026-09.md) (the report) and the per-bug entries it links under `known_bugs/`. Two of that report's bugs are **already being fixed in the working tree by another change** and are not implemented here: bug 4 (IRT-Router zero-cost rows; `training.data.pricing.impute_missing_usage`, done, uncommitted) and bug 2 (taxonomy fit; see Task 3, done and passing, uncommitted).

## Decisions to make before executing

These change what a task does. Each has a recommendation; the plan is written to the recommendation.

- **D1, Task 3 (taxonomy):** the in-flight fix makes clustering **train-split only** by default. It does **not** create a separate OOD taxonomy, so afterwards the *train-split* queries of the held-out math / code families (about 70% of 19,850) still shape the centroids. A truly OOD-safe taxonomy needs a parallel taxonomy directory plumbed through `relevance` and the NIRT data path, which is a design task of its own (the bug entry lists an `exclude_query_ids` option, like `build_query_bank`'s, as the still-open piece: it is Task 3's optional Step 3 below, and a held-out-family run would also need its own `paths.taxonomy` so it does not overwrite the main taxonomy). *Recommendation: accept the train-only fix now; do Step 3 only when an OOD experiment that depends on `r_q` is actually being run.*
- **D2, Task 12 (pricing):** `tests/training/data/test_pricing.py::test_cost_for_known_tokens` asserts `cost_for("m", None, None, prices) == 0.0` under the comment "missing token counts treated as 0". That is a deliberate contract that directly contradicts the schema's rule (`training/data/schemas.py`: missing usage is null, never 0). The fix flips it, and priced models whose source reports no tokens (e.g. lm-eval-harness rows) then get `cost = NaN` and drop out of dense routing matrices instead of appearing free. *Recommendation: flip it; nothing in the current built data is affected.* If you disagree, skip Task 12 and mark the bug `wontfix` with that reason.
- **D3, after Task 3 (operational, not code):** re-clustering invalidates `data/taxonomy/*`, the `query_relevance__*` store and every NIRT checkpoint trained on the old `r_q` (`load_relevance` detects a centroid mismatch, but nothing detects a *checkpoint* trained on old `r_q`). Rebuilding is a multi-hour job that also changes reported numbers. *Recommendation: land the code fix, then decide separately when to rebuild and retrain; this plan does not run it.*
- **D4, Task 7 (judge retries):** the bug entry suggests `range(max_retries + 1)`, but `test_persistent_server_error_exhausts_retries` pins `max_retries=5` to exactly 5 attempts and the message "after 5 retries". This plan therefore clamps to a minimum of one attempt instead (`max(1, max_retries)`), which fixes the `0` case without changing any existing behaviour. The name stays a misnomer for "attempt budget"; renaming it is out of scope.

## Global Constraints

- **Run tests from the repo root:** `.venv\Scripts\python.exe -m pytest <path> -q` (the suite's autouse fixture `chdir`s to the repo root, so code that reads `data/...` relative to CWD works). `python` is not on the bash PATH; always use the venv interpreter.
- **The working tree is dirty with unrelated, in-flight work.** Stage only the files a task lists, by explicit path (`git add <path> <path>`); never `git add -A`, `git add .` or `git commit -a`. Do **not** edit these files, which carry uncommitted changes from other work: `src/training/data/loaders.py`, `src/training/data/pricing.py` (only Task 12, and only after they are committed), `src/router/routing/routers.py`, `src/training/nirt/train.py`, `src/training/nirt/trainer.py`, `src/training/pipeline.py`, `tests/training/data/test_pricing.py` and `tests/training/data/test_irt_router.py` (same exception for Task 12), and `src/training/taxonomy/clustering.py`, `src/training/taxonomy/relevance.py`, `tests/training/taxonomy/test_taxonomy.py` (the taxonomy fix, Task 3). Do the work on a fresh branch or worktree (`superpowers:using-git-worktrees`).
- **Layering is enforced by import-linter** (`pyproject.toml`): `router` never imports `training` / `evaluation`; `training` never imports `evaluation`; `decompose` imports nothing from the others (except the two `decompose.embedders.* -> router.embeddings.encoder` exceptions). Task 4 adds to `router/determinism.py` and uses it from `router/embeddings/encoder.py` (both in `router`); Task 11 adds to `evaluation/`. Run `.venv\Scripts\lint-imports.exe` in Task 13.
- **Style:** match surrounding code. Files under `src/training/prompt_decomposition/`, `src/evaluation/prompt_decomposition/` and `tests/prompt_decomposition/` are ruff-scoped (line length 110, isort order); `ruff` is not installed in the venv, so keep imports sorted and lines under 110 by hand.
- **Commit trailer** on every commit: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`. Commit steps are for execution time; do not commit while merely reviewing this plan.
- **TDD:** every code task writes the failing test first, runs it to see the stated failure, then implements. The failure text in each "Expected" line is what the code does today, verified while writing the bug entries.

## Review Focus

Failure modes the bugs imply but no single task's headline test would catch. Each is pinned by a test in the named task.

- **Old artifacts on disk built by the buggy code** must still load or fail loudly: a `_progress.json` written before `vector_fingerprint` existed must still resume for an identical config and restart for a changed one (Task 5).
- **Missing optional dependencies:** `preserve_rng_state` must not crash when `torch` is not importable, and must restore the CUDA state only when CUDA exists (Task 4).
- **Nullable dtypes:** `n_choices` arrives as pandas `Int64` with `<NA>`; a naive `>= 2` on it yields `NA`, which silently breaks the `missing` count and the warning (Task 8).
- **A dataset builder must work with the real, current data**, not only a fixture: after Task 1's tests pass, build the real `real_only` split into a temp dir and confirm all 400 frozen rows land in `test` (Task 1, Step 5).
- **A fix that quietly breaks a consumer:** making clustering train-only means `build_relevance`, which defaulted to the *clustered* ids, would build `r_q` for train queries only, leaving validation, test and OOD queries with none (the baseline NIRT then fills a uniform `r_q`, silently). This plan's first draft of Task 3 had exactly that hole; the in-flight fix closes it and pins it with `test_relevance_is_still_built_for_every_query_not_just_the_clustered_ones` (Task 3).
- **A "fix" that only moves the leak:** Task 2 must count the dropped cold-start battles in `n_dropped` (the caller prints it and errors when zero battles survive), not drop them silently (Task 2).

---

## Task 1: Dataset builders import a module and a loader that no longer exist (HIGH)

**Bug:** `fixed_bugs/_module_level/training.prompt_decomposition.data.dataset/MOD-001-dataset-builders-import-a-module-and-a-loader-that-no-longer-exist/bug.md`

**Files:**
- Modify: `src/training/prompt_decomposition/data/dataset.py:155,217,268`
- Modify: `src/training/prompt_decomposition/data/sources.py` (module docstring; add `load_handlabelled`)
- Modify (comments/docstrings only): `src/evaluation/prompt_decomposition/overfit.py:10`, `src/decompose/classifiers/prompt_decomposition/settings.py:57`, `scripts/prompt_decomposition_cli.py:3` (the module docstring)
- Test: `tests/prompt_decomposition/test_data.py`

**Interfaces:**
- Produces: `sources.load_handlabelled(path: str | None = None) -> list[Example]` (reads `path or sources.HANDLABELLED_PATH` **at call time**, so a test can monkeypatch the constant); every row has `source="handlabelled"` and `capability=<domain>`.
- Consumes: `dataset.build_real_only_dataset(*, merge_domains=False, out_dir=PROCESSED_DIR, variant="real_only", seed=...)`, `dataset.build_task_dataset(*, include_synthetic=True, include_agent=False, include_mined=False, out_dir=PROCESSED_DIR, variant="task", seed=...)` (signatures unchanged).

- [ ] **Step 1: Write the failing tests**

Append to `tests/prompt_decomposition/test_data.py` (the file already imports `numpy as np`, `pandas as pd`, `pytest`, `Example`, `assert_no_leakage`):

```python
def _pool(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "prompt": [f"hand labelled prompt number {i} about topic {i % 7}" for i in range(n)],
        "domain": rng.choice(["science_math", "humanities", "language"], n),
    })


def test_load_handlabelled_reads_domain_as_capability(tmp_path):
    from training.prompt_decomposition.data import sources

    pool = _pool(10)
    path = tmp_path / "pool.parquet"
    pool.to_parquet(path, index=False)
    rows = sources.load_handlabelled(str(path))
    assert [r.prompt for r in rows] == pool["prompt"].tolist()
    assert [r.capability for r in rows] == pool["domain"].tolist()
    assert {r.source for r in rows} == {"handlabelled"}


def test_build_real_only_dataset_puts_the_frozen_eval_rows_in_test(tmp_path, monkeypatch):
    """Regression: both builders raised ImportError, so the domain head's splits
    could not be rebuilt from source (known_bugs training.prompt_decomposition.data.dataset MOD-001)."""
    from training.prompt_decomposition.data import dataset, sources

    pool = _pool()
    frozen = pool.iloc[:30]
    pool_path, frozen_path = tmp_path / "pool.parquet", tmp_path / "eval_frozen.parquet"
    pool.to_parquet(pool_path, index=False)
    frozen.to_parquet(frozen_path, index=False)
    monkeypatch.setattr(sources, "HANDLABELLED_PATH", str(pool_path))
    monkeypatch.setattr(dataset, "FROZEN_EVAL_PATH", frozen_path)

    out = tmp_path / "out"
    splits = dataset.build_real_only_dataset(out_dir=out)

    assert_no_leakage(splits)
    eval_rows = splits["test"][splits["test"]["source"] == "handlabelled_eval"]
    assert set(eval_rows["prompt"]) == set(frozen["prompt"])
    assert not set(splits["train"]["prompt"]) & set(frozen["prompt"])
    for name in ("train", "val", "test"):
        assert (out / "real_only" / f"{name}.parquet").exists()


def test_build_task_dataset_pins_synthetic_rows_to_train(tmp_path, monkeypatch):
    from training.prompt_decomposition.data import dataset, sources

    labels = ["answer"] * 60 + ["create"] * 30 + ["classify"] * 20
    real = [Example(prompt=f"real task prompt {i}", source="handlabelled", subset="real_tasks",
                    capability=t) for i, t in enumerate(labels)]
    synthetic = [Example(prompt=f"synthetic prompt {i}", source="synthetic", subset="task_extract",
                         capability="extract") for i in range(10)]
    monkeypatch.setattr(sources, "load_real_tasks", lambda: real)
    monkeypatch.setattr(sources, "load_synthetic_tasks", lambda: synthetic)
    monkeypatch.setattr(sources, "load_generated_tasks", lambda: [])

    splits = dataset.build_task_dataset(out_dir=tmp_path)

    assert (splits["train"]["source"] == "synthetic").sum() == 10
    for name in ("val", "test"):
        assert "synthetic" not in set(splits[name]["source"])
    assert {"answer", "extract"} <= set(splits["train"]["task"])
    assert (tmp_path / "task" / "train.parquet").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/prompt_decomposition/test_data.py -q -k "handlabelled_reads or build_real_only or build_task"`
Expected: 3 FAIL. `load_handlabelled` -> `AttributeError: module ... has no attribute 'load_handlabelled'`; the two builders -> `ImportError: cannot import name 'sources' from 'decompose.classifiers.prompt_decomposition'`.

- [ ] **Step 3: Implement**

In `src/training/prompt_decomposition/data/dataset.py`, replace the two function-local imports (lines 217 and 268):

```python
    from decompose.classifiers.prompt_decomposition import sources
```

with

```python
    from training.prompt_decomposition.data import sources
```

(The import stays function-local: `sources.py` imports `Example` from this module at load time, so a top-level import would be circular.) Then change the error text at line 155 to:

```python
        raise FileNotFoundError(
            f"missing splits {missing} under {target}; run `prompt-decomposition build-real` "
            f"(or `build-task`)")
```

In `src/training/prompt_decomposition/data/sources.py`, add after the path constants (after line 38) and before `load_real_tasks`:

```python
def load_handlabelled(path: str | None = None) -> list[Example]:
    """The hand-labelled real prompts: the domain head's training pool.

    Real user traffic, each prompt read and assigned one of the domain labels by
    hand. ``path`` defaults to :data:`HANDLABELLED_PATH`, resolved at call time.
    """
    frame = pd.read_parquet(path or HANDLABELLED_PATH)
    out = [Example(prompt=row.prompt, source="handlabelled", capability=row.domain)
           for row in frame.itertuples()]
    log.info("hand-labelled: %d rows", len(out))
    return out
```

Replace the module docstring's stale source table (lines 1-20; it lists loaders for LMArena flags, MMLU-Pro, RouterArena and BIG-bench that no longer exist) with:

```python
"""Loaders. Every source yields ``Example`` rows carrying a label in ``capability``.

The central problem this file solves is that **no single source has both real
prompts and reliable labels**, so the mix is deliberate and every row records its
``source``:

===================  ==========================  ==============================
loader               prompts                     labels
===================  ==========================  ==============================
load_handlabelled    real user traffic           domain, read and assigned by hand
load_real_tasks      real user traffic           task type, by hand
load_mined_tasks     real traffic, targeted      task type, by hand
load_agent_tasks     real user traffic           task type, by agent (kappa 0.819)
load_synthetic_tasks hand-written                task type
load_generated_tasks agent-generated             task type
===================  ==========================  ==============================

Training on benchmark exam questions alone produced a classifier that scored 91%
on exams and 47% on real prompts, so evaluation is always reported per source.
"""
```

Fix the stale CLI names (comment / docstring text only): in `src/evaluation/prompt_decomposition/overfit.py:10` change ``router overfit`` to ``prompt-decomposition overfit``; in `src/decompose/classifiers/prompt_decomposition/settings.py:57` change ``router describe-data`` to ``prompt-decomposition describe-data``; in `scripts/prompt_decomposition_cli.py` replace the module-docstring line `    build-data     download RouterArena, dedupe, split, write parquet` with the two lines `    build-real     hand-labelled real prompts only (the split the shipped model uses)` and `    build-task     hand-labelled task splits (synthetic rows into train only)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/prompt_decomposition -q`
Expected: all pass (the 13 existing `test_data.py` tests plus the 3 new ones, and the rest of the directory).

- [ ] **Step 5: Confirm against the real data, not just the fixture**

Run:
```
.venv\Scripts\python.exe -c "import tempfile; from pathlib import Path; from training.prompt_decomposition.data.dataset import build_real_only_dataset as b; s=b(out_dir=Path(tempfile.mkdtemp())); print({k: len(v) for k, v in s.items()}, int((s['test']['source']=='handlabelled_eval').sum()))"
```
Expected: prints three split sizes and `400` (all frozen-eval rows are in `test`). Then run the task builder the same way with `build_task_dataset`; expected sizes `{'train': 1430, 'val': 149, 'test': 201}`.

- [ ] **Step 6: Commit**

```bash
git add src/training/prompt_decomposition/data/dataset.py src/training/prompt_decomposition/data/sources.py src/evaluation/prompt_decomposition/overfit.py src/decompose/classifiers/prompt_decomposition/settings.py scripts/prompt_decomposition_cli.py tests/prompt_decomposition/test_data.py
git commit -m "fix(prompt-decomposition): make the dataset builders runnable again" -m "Both builders imported a sources module that moved and one called load_handlabelled, deleted in ea4d209. Restores the loader, fixes the import, and adds builder smoke tests." -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: Auxiliary pairwise loss trains on cold-start models (MED)

**Bug:** `known_bugs/_module_level/training.nirt.pairwise/MOD-001-auxiliary-battles-include-cold-start-models.md`

**Files:**
- Modify: `src/training/nirt/pairwise.py:56-83` (docstring + the `keep` mask)
- Test: `tests/training/nirt/test_pairwise.py`

**Interfaces:**
- Consumes: `TrainingData.cold_start_model_ids() -> list[str]` (already exists, `training/data/facade.py`).
- Produces: `build_pairwise_arrays(...)` now also drops any battle whose `model_a` or `model_b` is cold-start, counted in `PairwiseArrays.n_dropped`. Signature unchanged.

- [ ] **Step 1: Write the failing test**

In `tests/training/nirt/test_pairwise.py`, extend the stub so it can report cold-start models (existing tests keep working because the new argument defaults to none):

```python
class _StubPairwiseData:
    """Just enough of ``TrainingData`` for ``build_pairwise_arrays``."""

    def __init__(self, battles: pd.DataFrame, q_store, m_store, cold=()):
        self._battles = battles
        self._q = q_store
        self._m = m_store
        self._cold = list(cold)

    def cold_start_model_ids(self):
        return sorted(self._cold)

    # pairwise / query_embeddings / profile_embeddings unchanged
```

Add the test after `test_build_pairwise_arrays_joins_and_drops`:

```python
def test_build_pairwise_arrays_drops_battles_with_cold_start_models(battles, stores):
    """Cold-start models must appear in no training signal (splits.py); the
    battles fixture has mb-vs-mc (q1) and ma-vs-mc (q2) involving cold model mc."""
    q_store, m_store = stores
    data = _StubPairwiseData(battles, q_store, m_store, cold=["mc"])
    arr = build_pairwise_arrays(data, source="chatbot_arena", pathway="irt")
    assert len(arr) == 1                     # only q0 (ma vs mb) survives
    assert arr.n_dropped == 3                # q1, q2 (cold) + q3 (no embedding), counted
    assert arr.y.tolist() == [1.0]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/training/nirt/test_pairwise.py::test_build_pairwise_arrays_drops_battles_with_cold_start_models -q`
Expected: FAIL, `assert 3 == 1` (the cold battles are kept; only the missing-embedding row is dropped).

- [ ] **Step 3: Implement**

In `src/training/nirt/pairwise.py`, replace the `keep` block (lines 76-82):

```python
    # cold-start models are held out of EVERY training signal (data/splits.py);
    # TrainingData.pairwise(models="all") does not filter them, so do it here
    cold = set(data.cold_start_model_ids())
    keep = (
        df["query_id"].isin(q_store._index)
        & df["model_a"].isin(m_store._index)
        & df["model_b"].isin(m_store._index)
        & df["score_a"].notna()
        & ~df["model_a"].isin(cold)
        & ~df["model_b"].isin(cold)
    )
    dropped = int((~keep).sum())
```

and extend the docstring's second paragraph with: "Battles involving a cold-start model are dropped too (counted in ``n_dropped``): those models are held out entirely, and training on their outcomes would leak into the cold-start evaluation."

- [ ] **Step 4: Run the pairwise tests**

Run: `.venv\Scripts\python.exe -m pytest tests/training/nirt/test_pairwise.py -q`
Expected: all pass.

- [ ] **Step 5: Re-run the reproduction against the real data**

Run: `.venv\Scripts\python.exe known_bugs\_repros\repro_pairwise_cold_leak.py`
That script calls `TrainingData.pairwise` directly, so it still shows the facade's `models="all"` count (4617); the point of this step is only that it runs. To confirm the builder-level fix on real data, run:
```
.venv\Scripts\python.exe -c "from router.config import load_config; from training.data.facade import load_training_data; d=load_training_data(load_config()); cold=set(d.cold_start_model_ids()); p=d.pairwise(split='train', models='all'); print('cold battles the builder will now drop:', int((p.model_a.isin(cold)|p.model_b.isin(cold)).sum()))"
```
Expected: `4617`.

- [ ] **Step 6: Commit**

```bash
git add src/training/nirt/pairwise.py tests/training/nirt/test_pairwise.py
git commit -m "fix(nirt): keep cold-start models' battles out of the pairwise auxiliary loss" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: Taxonomy clusters are fit on validation, test and OOD queries (MED): IN FLIGHT, verify only

**Bug:** `fixed_bugs/_module_level/training.taxonomy/MOD-001-clusters-fit-on-all-splits-including-ood/bug.md` (see decisions D1 and D3 above)

**Status:** the fix is already in the working tree, uncommitted (`src/training/taxonomy/clustering.py`, `relevance.py`, `tests/training/taxonomy/test_taxonomy.py`), and its tests pass (8 passed on 2026-09-25). It has the same core as this plan's first draft (`_nirt_query_ids(cfg, *, split=None)`; `build_clusters(..., split="train")`), and one thing the first draft missed: `build_relevance` defaulted to `load_clusters(cfg)["query_id"]`, so a train-only clustering would have left validation / test / OOD queries with no `r_q`. The in-flight change makes `build_relevance` default to every NIRT query instead. **Do not reimplement.** This task is the review that closes it.

**Files:** none modified, except by the optional Steps 3-4 (`src/training/taxonomy/clustering.py`, `tests/training/taxonomy/test_taxonomy.py`, `scripts/taxonomy/cluster_queries.py`), which must wait until the in-flight change is committed.

- [ ] **Step 1: Run the in-flight tests**

Run: `.venv\Scripts\python.exe -m pytest tests/training/taxonomy -q`
Expected: `8 passed`, including the three new ones: `test_nirt_query_ids_can_be_limited_to_one_split`, `test_build_clusters_fits_on_train_queries_only`, `test_relevance_is_still_built_for_every_query_not_just_the_clustered_ones`.

- [ ] **Step 2: Review the diff against this checklist**

Run: `git diff -- src/training/taxonomy tests/training/taxonomy` and confirm:
- `build_clusters` defaults to `split="train"`, and `split=None` restores every split (the old behaviour).
- `build_relevance` defaults to `_nirt_query_ids(cfg)` with **no** split, and `relevance.py` no longer imports `load_clusters` (nothing else in that file used it).
- `clustering.json`'s `query_source` now records the split (`nirt_observations:train`).
- `training/phase0.py` stage 8 still calls `build_clusters(cfg)` then `build_relevance(cfg)` in that order (unchanged, and still right).
- Not yet present, and tracked as still open in the bug entry: an `exclude_query_ids` option for a held-out-family run (optional Step 3). Pre-existing and left alone: the `query_ids or ...` truthiness test in `build_clusters`, which raises for a numpy array (no caller passes one).

- [ ] **Step 3 (optional, only after the in-flight change is committed): `exclude_query_ids` for a held-out-family run**

Add to `tests/training/taxonomy/test_taxonomy.py` (it already imports `numpy as np` and `pandas as pd`; the `cfg` fixture comes from `tests/conftest.py`):

```python
def test_build_clusters_can_exclude_queries(monkeypatch, cfg):
    """A held-out-family run drops those queries before clustering (mirrors build_query_bank)."""
    import training.data.nirt as nirt_data
    import training.taxonomy.clustering as cl
    from router.embeddings import EmbeddingStore

    ids = [f"q{i}" for i in range(6)]
    obs = pd.DataFrame({"query_id": ids,
                        "split": ["train", "train", "validation", "test", "train", "ood"]})
    monkeypatch.setattr(nirt_data, "observations", lambda cfg, **kw: obs)
    store = EmbeddingStore(ids, np.eye(6, dtype=np.float32), {"id_field": "query_id"})
    monkeypatch.setattr(EmbeddingStore, "exists", classmethod(lambda cls, d: True))
    monkeypatch.setattr(EmbeddingStore, "load", classmethod(lambda cls, d, mmap=True: store))

    seen = {}

    def fake_cluster(emb, config, *, ids=None):
        seen["ids"] = list(ids)
        return cl.ClusterResult(labels=np.zeros(len(ids), dtype=int),
                                probabilities=np.ones(len(ids)),
                                centroids=np.zeros((1, 6), dtype=np.float32),
                                config=config, ids=list(ids))

    monkeypatch.setattr(cl, "cluster_embeddings", fake_cluster)

    cl.build_clusters(cfg, config=cl.ClusterConfig(), save=False, exclude_query_ids=["q1"])
    assert seen["ids"] == ["q0", "q4"]                       # train split minus the exclusion

    cl.build_clusters(cfg, config=cl.ClusterConfig(), save=False, split=None,
                      exclude_query_ids=["q5", "q0"])
    assert seen["ids"] == ["q1", "q2", "q3", "q4"]           # every split minus the exclusions

    cl.build_clusters(cfg, config=cl.ClusterConfig(), save=False)
    assert seen["ids"] == ["q0", "q1", "q4"]                 # default unchanged: train only
```

Run it: expected FAIL, `TypeError: build_clusters() got an unexpected keyword argument 'exclude_query_ids'`. Then in `src/training/taxonomy/clustering.py` change the `build_clusters` signature to end `save: bool = True, split: Optional[str] = "train", exclude_query_ids=None) -> ClusterResult:` and replace the `ids = [...]` line with:

```python
    drop = set(map(str, exclude_query_ids or ()))
    ids = [q for q in (query_ids or _nirt_query_ids(cfg, split=split))
           if q in store and str(q) not in drop]
```

Add one sentence to the docstring: "``exclude_query_ids`` are dropped before clustering (mirrors ``build_query_bank``); a held-out-family run should also point ``paths.taxonomy`` elsewhere so it does not overwrite the main taxonomy." Re-run `.venv\Scripts\python.exe -m pytest tests/training/taxonomy -q`; expected all pass. (Validated on top of the in-flight version of `clustering.py`: fails unpatched with the `TypeError` above, passes patched.)

- [ ] **Step 4 (optional): make the new default discoverable in the script**

In `scripts/taxonomy/cluster_queries.py` change the module docstring's first line to `"""UMAP -> HDBSCAN clustering of the TRAIN-split NIRT query embeddings."""`, add `ap.add_argument("--all-splits", action="store_true", help="fit on every split, not just train (the pre-fix behaviour)")`, and call `build_clusters(cfg, config=cc, split=None if args.all_splits else "train")`. No test: it is an argparse pass-through of a parameter the tests above already pin.

- [ ] **Step 5: Decision D3 (operational, not code)**

Re-clustering invalidates `data/taxonomy/*`, the `query_relevance__*` store and every NIRT checkpoint trained on the old `r_q` (the bug entry's dry run: 118 clusters against 154 today, so checkpoints from `artifacts/phase1`, `phase2` and `irt_router` all become stale). Do not rebuild as part of this plan; record the decision in the handoff.

**Commit:** belongs to the owner of the in-flight change. When it lands, Task 13 marks the bug fixed with that commit.

---

## Task 4: `TextEncoder.encode()` reseeds the process-global RNGs (MED-LOW)

**Bug:** `known_bugs/TextEncoder/TEXTENCODER-001-encode-reseeds-global-rngs.md`

**Files:**
- Modify: `src/router/determinism.py` (add `preserve_rng_state`)
- Modify: `src/router/embeddings/encoder.py:33,117-126,150`
- Create: `tests/router/embeddings/test_encoder_rng.py`

**Interfaces:**
- Produces: `router.determinism.preserve_rng_state()`, a context manager that restores `random`, numpy and (if importable) torch / CUDA global RNG states on exit; safe with no torch installed.
- Consumes: `router.determinism.seed_everything(seed, *, deterministic_algorithms=True)` (unchanged).

- [ ] **Step 1: Write the failing tests**

Create `tests/router/embeddings/test_encoder_rng.py`:

```python
"""TextEncoder must not disturb the host process's global RNG streams
(known_bugs TextEncoder TEXTENCODER-001)."""

from __future__ import annotations

import random
import sys

import numpy as np
import pytest

from router.determinism import preserve_rng_state
from router.embeddings.encoder import EncoderConfig, TextEncoder


class _StubEncoder(TextEncoder):
    """bert-backend encoder with the model call stubbed out (no download)."""

    def _encode_bert_batch(self, batch):
        return np.zeros((len(batch), 3), np.float32)


def test_encode_leaves_python_and_numpy_rng_untouched():
    enc = _StubEncoder(EncoderConfig(backend="bert"))
    random.seed(1234)
    np.random.seed(1234)
    expected = (random.random(), float(np.random.rand()))
    random.seed(1234)
    np.random.seed(1234)
    enc.encode(["a routed request"])
    assert (random.random(), float(np.random.rand())) == expected


def test_encode_leaves_torch_rng_untouched():
    torch = pytest.importorskip("torch")
    enc = _StubEncoder(EncoderConfig(backend="bert"))
    torch.manual_seed(7)
    expected = torch.rand(1).item()
    torch.manual_seed(7)
    enc.encode(["a routed request"])
    assert torch.rand(1).item() == expected


def test_repeated_encodes_do_not_pin_the_host_stream():
    """The observed symptom: after the first encode every later draw repeated."""
    enc = _StubEncoder(EncoderConfig(backend="bert"))
    random.seed(99)
    draws = []
    for _ in range(3):
        draws.append(random.random())
        enc.encode(["x"])
    assert len(set(draws)) == 3


def test_model_load_is_seeded_but_restores_the_callers_stream():
    class _Loads(TextEncoder):
        drawn_during_load = None

        def _load_bert(self):
            _Loads.drawn_during_load = random.random()
            return object()

    enc = _Loads(EncoderConfig(backend="bert", seed=42))
    random.seed(42)
    seeded_draw = random.random()
    random.seed(99)
    caller_next = random.random()
    random.seed(99)
    _ = enc.model
    assert _Loads.drawn_during_load == seeded_draw      # the load ran under the encoder's seed
    assert random.random() == caller_next               # ...and the caller's stream is back


def test_preserve_rng_state_restores_all_generators():
    torch = pytest.importorskip("torch")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    expected = (random.random(), float(np.random.rand()), torch.rand(1).item())
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    with preserve_rng_state():
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        random.random(), np.random.rand(), torch.rand(1)
    assert (random.random(), float(np.random.rand()), torch.rand(1).item()) == expected


def test_preserve_rng_state_works_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)      # `import torch` now raises ImportError
    random.seed(3)
    expected = random.random()
    random.seed(3)
    with preserve_rng_state():
        random.seed(8)
    assert random.random() == expected
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/router/embeddings/test_encoder_rng.py -q`
Expected: collection error `ImportError: cannot import name 'preserve_rng_state' from 'router.determinism'` (all 6 tests error until the helper exists). Add the helper (Step 3a) and re-run: the first four then FAIL on the encoder's reseed (e.g. `assert (0.6394..., 0.3745...) == (0.9665..., 0.1915...)`).

- [ ] **Step 3: Implement**

(a) In `src/router/determinism.py`, add `import contextlib` to the imports and append:

```python
@contextlib.contextmanager
def preserve_rng_state():
    """Run a block, then restore the ``random`` / numpy / torch (and CUDA) global RNG states.

    For code that must seed for reproducibility (e.g. a model load) but also runs
    inside someone else's process, where resetting their generators is a side
    effect. Works without torch installed.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = cuda_state = None
    try:
        import torch

        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
    except Exception:  # pragma: no cover - torch is optional here
        pass
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
```

(b) In `src/router/embeddings/encoder.py` change line 33 to `from ..determinism import preserve_rng_state` plus the existing `seed_everything as _seed_everything` import (`from ..determinism import preserve_rng_state, seed_everything as _seed_everything`). Replace the `model` property body (lines 117-126):

```python
    @property
    def model(self):
        if self._model is None:
            # Seeded so the model init is reproducible, but scoped: this also runs on
            # the serving path, where resetting the host's global RNGs is a side effect.
            # (No deterministic-algorithms switch either: that is process-wide too.)
            with preserve_rng_state():
                _seed_everything(self.cfg.seed, deterministic_algorithms=False)
                self._model = (
                    self._load_bert() if self.cfg.backend == "bert"
                    else self._load_sentence_transformer()
                )
        return self._model
```

and **delete** line 150 in `iter_encode` (`_seed_everything(self.cfg.seed, deterministic_algorithms=False)`), replacing it with the comment `# no reseed here: encoding is eval-mode / no_grad with no dropout, and this runs per request`.

- [ ] **Step 4: Run the RNG tests and the existing embedding tests**

Run: `.venv\Scripts\python.exe -m pytest tests/router/embeddings -q`
Expected: all pass (the `encoder`-fixture tests skip if the model is not cached; `test_deterministic_inference` still holds because eval-mode encoding uses no RNG).

- [ ] **Step 5: Re-run the reproduction**

Run: `.venv\Scripts\python.exe known_bugs\_repros\repro_encode_reseeds_globals.py`
Expected: three distinct draws and `draws after the first encode identical -> False`.

- [ ] **Step 6: Commit**

```bash
git add src/router/determinism.py src/router/embeddings/encoder.py tests/router/embeddings/test_encoder_rng.py
git commit -m "fix(embeddings): stop encode() reseeding the process-global RNGs" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 5: `build_store` won't resume a partial build after a device / batch-size change (LOW-MED)

**Bug:** `known_bugs/_module_level/router.embeddings/MOD-001-build-store-resume-ignores-vector-fingerprint.md`

**Files:**
- Modify: `src/router/embeddings/encoder.py:497-507,520-527`
- Create: `tests/router/embeddings/test_build_store_resume.py`

**Interfaces:**
- Produces: `_progress.json` gains a `vector_fingerprint` key; the resume gate compares it when present and falls back to `config_fingerprint` for progress files written before this change.

- [ ] **Step 1: Write the failing tests**

Create `tests/router/embeddings/test_build_store_resume.py`:

```python
"""build_store resumes on vector-affecting settings only (known_bugs router.embeddings MOD-001)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from router.embeddings import build_store
from router.embeddings.encoder import EncoderConfig, TextEncoder

IDS = [f"q{i}" for i in range(40)]
TEXTS = [f"text {i}" for i in IDS]


class _StubEncoder(TextEncoder):
    """Row i of the output is filled with i. Optionally 'dies' after N batches."""

    def __init__(self, cfg, die_after=None):
        super().__init__(cfg)
        self.batches = 0
        self.die_after = die_after

    @property
    def dim(self):
        return 4

    def iter_encode(self, texts, batch_size=None, show_progress=False, _start_offset=0):
        bs = int(batch_size or self.cfg.batch_size)
        for i in range(0, len(texts), bs):
            if self.die_after is not None and self.batches >= self.die_after:
                raise KeyboardInterrupt("simulated kill")
            self.batches += 1
            rows = np.arange(_start_offset + i, _start_offset + i + len(texts[i:i + bs]),
                             dtype=np.float32)
            yield _start_offset + i, np.repeat(rows[:, None], 4, axis=1)


def _interrupted_cpu_build(tmp_path):
    cpu = _StubEncoder(EncoderConfig(device="cpu", batch_size=4), die_after=5)
    with pytest.raises(KeyboardInterrupt):
        build_store(tmp_path, IDS, TEXTS, cpu, id_field="query_id", flush_every=1,
                    show_progress=False)


def _finish(tmp_path, encoder):
    store = build_store(tmp_path, IDS, TEXTS, encoder, id_field="query_id", flush_every=1,
                        show_progress=False)
    rows = np.asarray(store.matrix)[:, 0].copy()
    store.close()
    return rows


def test_partial_build_resumes_after_a_device_change(tmp_path):
    _interrupted_cpu_build(tmp_path)
    gpu = _StubEncoder(EncoderConfig(device="cuda", batch_size=4))
    rows = _finish(tmp_path, gpu)
    assert gpu.batches == 5                                   # only the remaining 20 rows
    np.testing.assert_array_equal(rows, np.arange(40, dtype=np.float32))


def test_partial_build_resumes_after_a_batch_size_change(tmp_path):
    _interrupted_cpu_build(tmp_path)
    other = _StubEncoder(EncoderConfig(device="cpu", batch_size=2))
    rows = _finish(tmp_path, other)
    assert other.batches == 10                                # 20 remaining rows at batch 2
    np.testing.assert_array_equal(rows, np.arange(40, dtype=np.float32))


def test_partial_build_restarts_when_the_vectors_would_differ(tmp_path):
    _interrupted_cpu_build(tmp_path)
    changed = _StubEncoder(EncoderConfig(device="cpu", batch_size=4, normalize=False))
    _finish(tmp_path, changed)
    assert changed.batches == 10                              # full rebuild: 40 rows / batch 4


def test_progress_file_without_vector_fingerprint_still_resumes_identical_config(tmp_path):
    """Old progress files (pre-fix) carry only config_fingerprint."""
    _interrupted_cpu_build(tmp_path)
    progress = tmp_path / "_progress.json"
    prog = json.loads(progress.read_text())
    prog.pop("vector_fingerprint", None)
    progress.write_text(json.dumps(prog))
    same = _StubEncoder(EncoderConfig(device="cpu", batch_size=4))
    _finish(tmp_path, same)
    assert same.batches == 5


def test_progress_file_without_vector_fingerprint_restarts_when_config_differs(tmp_path):
    _interrupted_cpu_build(tmp_path)
    progress = tmp_path / "_progress.json"
    prog = json.loads(progress.read_text())
    prog.pop("vector_fingerprint", None)
    progress.write_text(json.dumps(prog))
    other = _StubEncoder(EncoderConfig(device="cuda", batch_size=4))
    _finish(tmp_path, other)
    assert other.batches == 10                                # legacy file: conservative restart
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/router/embeddings/test_build_store_resume.py -q`
Expected: `test_partial_build_resumes_after_a_device_change` and `..._batch_size_change` FAIL with `assert 10 == 5` / `assert 20 == 10` (restarted from row 0). The other three pass already, which is what pins the no-regression cases.

- [ ] **Step 3: Implement**

In `build_store` (`src/router/embeddings/encoder.py`), replace the resume gate (lines 497-507):

```python
    rows_done = 0
    if resume and not force and ppath.exists() and vpath.exists():
        prog = json.loads(ppath.read_text("utf-8"))
        # vector_fingerprint covers only the fields that change the vectors, so a device or
        # batch-size change resumes; progress files written before it existed carry only
        # config_fingerprint, and fall back to the stricter comparison.
        if "vector_fingerprint" in prog:
            same_vectors = prog["vector_fingerprint"] == vec_fp
        else:
            same_vectors = prog.get("config_fingerprint") == cfg_fp
        if (prog.get("ids_fingerprint") == id_fp
                and same_vectors
                and prog.get("texts_fingerprint", txt_fp) == txt_fp
                and prog.get("dim") == dim):
            rows_done = int(prog.get("rows_done", 0))
            if show_progress:
                print(f"[build_store] resuming {d.name} at row {rows_done}/{n}")
        elif show_progress:
            print(f"[build_store] {d.name}: fingerprint mismatch, restarting")
```

and add `"vector_fingerprint": vec_fp,` to the dict written by `_write_progress` (next to `"config_fingerprint": cfg_fp`).

- [ ] **Step 4: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/router/embeddings -q`
Expected: all pass, including the existing `test_build_store_resumes_from_checkpoint` (its hand-written progress file has no `vector_fingerprint`, so it exercises the legacy fallback).

- [ ] **Step 5: Commit**

```bash
git add src/router/embeddings/encoder.py tests/router/embeddings/test_build_store_resume.py
git commit -m "fix(embeddings): resume a partial build_store on vector_fingerprint, not the full config" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 6: `QueryBank` similarity weighting is inverted on an L2 index (LOW, latent)

**Bug:** `known_bugs/QueryBank/QUERYBANK-001-similarity-weighting-inverted-on-flat-l2.md`

**Files:**
- Modify: `src/training/retrieval/query_bank.py:240-244`
- Test: `tests/training/retrieval/test_retrieval.py`

**Interfaces:**
- Produces: for a bank with `normalized == False` (L2), `neighbor_weighted_mean(weighting="similarity")` weights neighbour `i` by `1 / (1 + d_i)` where `d_i` is the FAISS squared L2 distance; inner-product banks are unchanged (`max(cos, 0)`).

- [ ] **Step 1: Write the failing test**

Append to `tests/training/retrieval/test_retrieval.py`:

```python
def test_similarity_weighting_favours_the_nearer_neighbour_on_an_l2_bank():
    """On an L2 index FAISS scores are squared distances (smaller = closer); weighting by the raw
    score gave the FARTHEST neighbour all the weight (known_bugs QUERYBANK-001)."""
    import numpy as np

    faiss = pytest.importorskip("faiss")
    from router.embeddings import EmbeddingStore
    from training.retrieval.query_bank import QueryBank

    index = faiss.IndexFlatL2(2)
    index.add(np.array([[1, 0], [10, 0]], np.float32))            # near, far
    bank = QueryBank(index, ["near", "far"],
                     {"normalized": False, "index_type": "flat_l2", "default_k": 2})
    store = EmbeddingStore(["near", "far"], np.array([[1, 0], [0, 1]], np.float32),
                           {"id_field": "query_id"})

    vec, found = bank.neighbor_weighted_mean(np.array([[1, 0]], np.float32), store, k=2,
                                             weighting="similarity")

    assert found.tolist() == [2]
    # d_near = 0 -> s = 1;  d_far = 81 -> s = 1/82;  weights 82/83 and 1/83
    np.testing.assert_allclose(vec[0], [82 / 83, 1 / 83], rtol=1e-5)
```

(If `pytest` is not already imported at the top of the file, add `import pytest`.)

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/training/retrieval/test_retrieval.py::test_similarity_weighting_favours_the_nearer_neighbour_on_an_l2_bank -q`
Expected: FAIL, `Mismatched elements ... x: array([0., 1.])` (all weight on the far neighbour).

- [ ] **Step 3: Implement**

In `neighbor_weighted_mean` (`src/training/retrieval/query_bank.py`), replace the two lines under `if weighting == "similarity":`:

```python
        if weighting == "similarity":
            scores = res.scores.astype(np.float64)
            if not self.normalized:
                # flat_l2 scores are squared distances (smaller = closer): turn them into a
                # similarity in (0, 1] before weighting. Padded slots score 0 but are masked by `valid`.
                scores = 1.0 / (1.0 + np.clip(scores, 0.0, None))
            w = np.clip(scores, 0.0, None) * valid
            w = np.where((w.sum(axis=1, keepdims=True) > 1e-12), w, uniform)
```

and change the docstring line ``weighting="similarity"`` weights neighbour ``i`` by ``max(cos_i, 0)`` (the flat_ip score)`` to add: "(for a flat_l2 bank, by ``1 / (1 + d_i)`` of the squared L2 distance)".

- [ ] **Step 4: Run the retrieval tests**

Run: `.venv\Scripts\python.exe -m pytest tests/training/retrieval -q`
Expected: all pass (the flat-IP behaviour and `test_knn_impute.py` are unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/training/retrieval/query_bank.py tests/training/retrieval/test_retrieval.py
git commit -m "fix(retrieval): similarity-weight an L2 QueryBank by closeness, not distance" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 7: `JudgeClient(max_retries=0)` makes no attempt (LOW; see decision D4)

**Bug:** `known_bugs/JudgeClient/JUDGECLIENT-001-max-retries-zero-makes-no-attempt.md`

**Files:**
- Modify: `src/evaluation/judge/client.py:184-196`
- Test: `tests/evaluation/judge/test_judge_client.py`

**Interfaces:**
- Produces: `_call_with_retry` makes `max(1, self.max_retries)` attempts. `max_retries=5` still means 5 attempts and the message still reads "after 5 retries" (both pinned by the existing `test_persistent_server_error_exhausts_retries`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/evaluation/judge/test_judge_client.py` (it already defines `_retry_client`, `_FakeStatusError`, `json`, `pytest`):

```python
def test_zero_max_retries_still_makes_one_attempt(monkeypatch):
    ok = json.dumps({"margin": 2, "reason": "x"})
    c, calls = _retry_client(monkeypatch, [ok])
    c.max_retries = 0
    assert c.judge("q", "a", "b").margin == 2
    assert len(calls) == 1


def test_zero_max_retries_fails_after_a_single_attempt_on_a_transient_error(monkeypatch):
    c, calls = _retry_client(monkeypatch, [_FakeStatusError(503)])
    c.max_retries = 0
    with pytest.raises(RuntimeError):
        c.judge("q", "a", "b")
    assert len(calls) == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/judge/test_judge_client.py -q -k zero_max_retries`
Expected: 2 FAIL. The first raises `RuntimeError: judge call failed after 0 retries` before any call; the second fails `assert 0 == 1`.

- [ ] **Step 3: Implement**

Replace `_call_with_retry` in `src/evaluation/judge/client.py`:

```python
    def _call_with_retry(self, messages: list[dict]) -> str:
        # ``max_retries`` is the attempt budget (5 -> 5 attempts, pinned by the tests);
        # 0 must still try once rather than raise without calling.
        attempts = max(1, self.max_retries)
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            self._throttle()
            try:
                return self._call_once(messages)
            except Exception as exc:
                if not _is_transient(exc, self._connection_errors):
                    raise   # auth / bad request / unknown model: retrying can't help
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"judge call failed after {attempts} retries") from last_exc
```

- [ ] **Step 4: Run the judge tests**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/judge -q`
Expected: all pass, including `test_persistent_server_error_exhausts_retries` (5 calls, "after 5 retries").

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/judge/client.py tests/evaluation/judge/test_judge_client.py
git commit -m "fix(judge): max_retries=0 makes one attempt instead of raising" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 8: Chance correction with `n_choices < 2` gives NaN but is flagged corrected (LOW, latent)

**Bug:** `known_bugs/_module_level/training.data.chance_correction/MOD-001-n-choices-below-two-gives-nan-but-flagged-corrected.md`

**Files:**
- Modify: `src/training/data/chance_correction.py:64-69,86-97`
- Test: `tests/training/data/test_chance_correction.py`

**Interfaces:**
- Produces: `normalized_correction(score, n_choices, clip=True)` raises `ValueError` for `n_choices < 2`; `apply_chance_correction` treats `n_choices < 2` like a missing count (row left uncorrected, `chance_corrected=False`, included in the existing "no n_choices" warning).

- [ ] **Step 1: Write the failing tests**

Append to `tests/training/data/test_chance_correction.py` (imports `warnings`, `pd`, `pytest`, `_CC`, `apply_chance_correction`, `normalized_correction` already exist):

```python
def test_normalized_correction_rejects_fewer_than_two_choices():
    for n in (1, 0):
        with pytest.raises(ValueError, match="n_choices"):
            normalized_correction(1.0, n)


def test_fewer_than_two_choices_is_left_uncorrected_not_nan():
    """n_choices 1 / 0 gave corrected_score NaN with chance_corrected=True, and the facade then
    silently fell back to the raw score (known_bugs chance_correction MOD-001). <NA> counts too."""
    df = pd.DataFrame({
        "score": [1.0, 0.0, 1.0, 1.0],
        "is_multiple_choice": [True] * 4,
        "n_choices": pd.array([1, 0, 4, None], dtype="Int64"),
        "dataset": ["x"] * 4,
    })
    with pytest.warns(UserWarning, match="no n_choices"):
        out = apply_chance_correction(df, _CC(warn=True))
    assert out["chance_corrected"].tolist() == [False, False, True, False]
    assert out["corrected_score"].isna().tolist() == [True, True, False, True]
    assert out["corrected_score"].iloc[2] == pytest.approx(1.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data/test_chance_correction.py -q -k "fewer_than_two"`
Expected: 2 FAIL. `normalized_correction(1.0, 1)` raises `ZeroDivisionError` (not `ValueError`); the frame test shows `chance_corrected == [True, True, True, False]` and `UserWarning` only for the `<NA>` row.

- [ ] **Step 3: Implement**

In `src/training/data/chance_correction.py`, in `normalized_correction` add as its first lines:

```python
    if n_choices < 2:
        raise ValueError(f"n_choices must be >= 2 to have a chance level, got {n_choices}")
```

and in `apply_chance_correction` replace `has_n = out["n_choices"].notna()` (line 87) with:

```python
    # float first: n_choices is a nullable Int64, and `<NA> >= 2` is <NA>, which would break the
    # `missing` count below. A 1- or 0-option "multiple choice" has no chance level either.
    n_choices = pd.to_numeric(out["n_choices"], errors="coerce").astype(float)
    has_n = n_choices >= 2
```

and the warning text (line 94-97) to `f"chance correction: {n_missing} multiple-choice observations have no n_choices (missing or < 2); leaving them unchanged. Tasks: {tasks[:20]}"` (it still contains the substring the existing test looks for, "no n_choices").

- [ ] **Step 4: Run the chance-correction and data tests**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data/test_chance_correction.py tests/training/data/test_data.py tests/training/data/test_facade.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/data/chance_correction.py tests/training/data/test_chance_correction.py
git commit -m "fix(data): treat n_choices < 2 as missing in chance correction" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 9: A unanimous graded row is counted as "discriminative" (LOW)

**Bug:** `known_bugs/_module_level/evaluation.data.stratify/MOD-001-unanimous-graded-rows-counted-discriminative.md`

**Files:**
- Modify: `src/evaluation/data/stratify.py:51`
- Test: `tests/evaluation/data/test_stratify.py`

**Interfaces:**
- Produces: the discriminative stratum is the rows with `row_std > tol` (correctness varies across the pool). Identical to the old rule for 0/1 matrices; different only for graded ones.

- [ ] **Step 1: Write the failing test**

Append to `tests/evaluation/data/test_stratify.py`:

```python
def test_discriminative_excludes_unanimous_graded_rows():
    """Every model at 0.5 is unanimous (std 0) even though the mean is not 0 or 1."""
    R = pd.DataFrame(
        [[0.5, 0.5, 0.5], [0.25, 0.75, 0.5], [1.0, 1.0, 1.0]],
        index=["half", "spread", "right"], columns=["a", "b", "c"],
    )
    out = sample_stratified_queries(R, n_discriminative=10, n_natural=0, seed=0)
    assert set(out.index) == {"spread"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/data/test_stratify.py::test_discriminative_excludes_unanimous_graded_rows -q`
Expected: FAIL, `assert {'half', 'spread'} == {'spread'}`.

- [ ] **Step 3: Implement**

Replace line 51 of `src/evaluation/data/stratify.py`:

```python
    # "correctness varies across the pool" is a std > 0 statement; a mean strictly inside (0, 1)
    # is only equivalent for 0/1 matrices, and lets unanimous graded rows (all 0.5) through
    discriminative_mask = row_std > tol
```

- [ ] **Step 4: Run the stratify tests**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/data -q`
Expected: all pass (the existing binary `test_discriminative_excludes_unanimous_rows` still selects `{"q2", "q3"}`).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/data/stratify.py tests/evaluation/data/test_stratify.py
git commit -m "fix(evaluation): select the discriminative stratum by row std, not row mean" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 10: `run_all` drops failures when nothing succeeds; `load_synthetic_tasks` raises where it means to warn (LOW)

Run after Task 1 (both edit `sources.py` / `tests/prompt_decomposition/`).

**Bugs:** `known_bugs/_module_level/training.prompt_decomposition.experiment/MOD-001-run-all-records-no-failures-when-nothing-succeeds.md` and `known_bugs/_module_level/training.prompt_decomposition.data.sources/MOD-001-missing-synthetic-file-raises-instead-of-warning.md`

**Files:**
- Modify: `src/training/prompt_decomposition/experiment.py:242-252`
- Modify: `src/training/prompt_decomposition/data/sources.py:1-40,80-90`
- Create: `tests/prompt_decomposition/test_experiment.py`
- Test: `tests/prompt_decomposition/test_data.py`

**Interfaces:**
- Produces: `run_all` writes `leaderboard.csv`, `leaderboard.md` and (if any failed) `failures.json` after every config, including when no experiment has succeeded. `load_synthetic_tasks` logs `"no synthetic task file at ..."` and skips a task whose file does not exist.

- [ ] **Step 1: Write the failing tests**

Create `tests/prompt_decomposition/test_experiment.py`:

```python
"""Sweep bookkeeping: failures must be recorded even when nothing succeeds
(known_bugs training.prompt_decomposition.experiment MOD-001)."""

import json

from training.prompt_decomposition.config import DataConfig, ExperimentConfig, ModelConfig
from training.prompt_decomposition.experiment import run_all


def test_run_all_records_failures_even_when_nothing_succeeds(tmp_path):
    bad = ExperimentConfig(name="bad", model=ModelConfig("no_such_model"),
                           data=DataConfig(variant="no_such_variant"))
    frame = run_all([bad], out_dir=tmp_path, continue_on_error=True)

    assert frame.empty
    failures = json.loads((tmp_path / "failures.json").read_text(encoding="utf-8"))
    assert "bad" in failures
    board = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert "_no successful runs_" in board
    assert "bad" in board
```

Append to `tests/prompt_decomposition/test_data.py`:

```python
def test_load_synthetic_tasks_warns_and_skips_a_missing_file(caplog):
    """The 'no synthetic task file' warning was unreachable: a missing path raised
    FileNotFoundError from exec_module (known_bugs ...data.sources MOD-001)."""
    import logging

    from training.prompt_decomposition.data import sources

    with caplog.at_level(logging.WARNING):
        rows = sources.load_synthetic_tasks(["not_a_task"])
    assert rows == []
    assert "no synthetic task file" in caplog.text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/prompt_decomposition/test_experiment.py tests/prompt_decomposition/test_data.py -q -k "records_failures or missing_file"`
Expected: 2 FAIL: `FileNotFoundError: ... failures.json` and `FileNotFoundError: ... task_not_a_task.py`.

- [ ] **Step 3: Implement**

In `src/training/prompt_decomposition/experiment.py`, inside `run_all`, replace the tail of the loop (lines 251-252):

```python
        # always, so a sweep in which every config fails still leaves failures.json and a
        # leaderboard saying so ("no successful runs"); _write_summary handles empty results
        _write_summary(results, failures, out_dir)
```

(remove the `if results:` guard; the `return _summary_frame(results)` after the loop is unchanged).

In `src/training/prompt_decomposition/data/sources.py` add `from pathlib import Path` to the imports (stdlib group, after `import logging`) and, in `load_synthetic_tasks`, insert before `spec = importlib.util.spec_from_file_location(...)`:

```python
        if not Path(path).exists():
            log.warning("no synthetic task file at %s", path)
            continue
```

- [ ] **Step 4: Run the prompt-decomposition tests**

Run: `.venv\Scripts\python.exe -m pytest tests/prompt_decomposition -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/prompt_decomposition/experiment.py src/training/prompt_decomposition/data/sources.py tests/prompt_decomposition/test_experiment.py tests/prompt_decomposition/test_data.py
git commit -m "fix(prompt-decomposition): record sweep failures when none succeed; skip missing synthetic files" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 11: The length head's "ceiling" is a reliability, compared against a correlation (LOW-MED; docs + one helper)

**Bug:** `known_bugs/_module_level/evaluation.prompt_decomposition.length_audit/MOD-001-target-ceiling-is-a-reliability-not-a-correlation-ceiling.md`

**Files:**
- Modify: `src/evaluation/prompt_decomposition/length_audit.py:16-22,157-168,193-195`
- Modify (comment only): `src/training/prompt_decomposition/data/arena_corpus.py:19-21`
- Modify (docs): `docs/prompt_signals.md` (lines ~66, ~200-202, ~252-263, ~278), `docs/prompt_decomposition.md` (~130-137), `docs/prompt_decomposition_experiments.md` (~522-528)
- Test: `tests/prompt_decomposition/test_length.py`

**Interfaces:**
- Produces: `correlation_ceiling(tokens_a, tokens_b, *, averaged=True) -> float` = `sqrt(max(target_ceiling(...), 0))`. `target_ceiling` is **unchanged** (still the variance-share reliability, still the bar for R-squared) and keeps its existing tests.

- [ ] **Step 1: Write the failing tests**

In `tests/prompt_decomposition/test_length.py`, inside the class that already holds the two `target_ceiling` tests (after `test_perfect_agreement_leaves_the_ceiling_at_one`):

```python
    def test_correlation_ceiling_is_the_square_root_of_the_reliability(self):
        """target_ceiling is a variance share (bar for R-squared); a Spearman/Pearson rho is bounded
        by its square root (known_bugs length_audit MOD-001)."""
        from evaluation.prompt_decomposition.length_audit import correlation_ceiling, target_ceiling

        rng = np.random.default_rng(0)
        signal = rng.normal(0, 1, 4000)
        a = np.expm1(signal + rng.normal(0, 1, 4000))
        b = np.expm1(signal + rng.normal(0, 1, 4000))
        assert correlation_ceiling(a, b) == pytest.approx(np.sqrt(target_ceiling(a, b)))
        assert correlation_ceiling(a, b) > target_ceiling(a, b)
        assert correlation_ceiling(a, b, averaged=False) == pytest.approx(
            np.sqrt(target_ceiling(a, b, averaged=False)))

    def test_correlation_ceiling_is_zero_not_nan_when_the_reliability_is_negative(self, monkeypatch):
        from evaluation.prompt_decomposition import length_audit

        monkeypatch.setattr(length_audit, "target_ceiling", lambda *a, **k: -0.4)
        assert length_audit.correlation_ceiling(np.arange(5), np.arange(5)) == 0.0

    def test_perfect_agreement_leaves_the_correlation_ceiling_at_one(self):
        from evaluation.prompt_decomposition.length_audit import correlation_ceiling

        tokens = np.arange(1, 500)
        assert correlation_ceiling(tokens, tokens) == pytest.approx(1.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/prompt_decomposition/test_length.py -q -k correlation_ceiling`
Expected: 3 FAIL, `ImportError: cannot import name 'correlation_ceiling'`.

- [ ] **Step 3: Implement the helper and fix the code-side wording**

In `src/evaluation/prompt_decomposition/length_audit.py`, add after `target_ceiling`:

```python
def correlation_ceiling(tokens_a: np.ndarray, tokens_b: np.ndarray,
                        *, averaged: bool = True) -> float:
    """The highest correlation (Pearson or Spearman rho) a prompt-only predictor can reach.

    :func:`target_ceiling` returns a *reliability*: the share of the target's variance that is
    signal, which is the bar for an R-squared. A correlation is bounded by its square root
    (0.590 -> 0.768 against one model; 0.742 -> 0.861 against the two-model average). Assumes the
    two models are roughly parallel measures of one prompt-driven signal, so read it as an upper
    bound. Floored at 0 when the models disagree (a negative reliability has no square root).
    """
    return float(np.sqrt(max(target_ceiling(tokens_a, tokens_b, averaged=averaged), 0.0)))
```

Extend `target_ceiling`'s docstring first sentence to: "How reproducible the target is, as a **variance share** (the bar for R-squared; use :func:`correlation_ceiling` for a correlation), and so how high a score can go." In `audit_feature_sets` (lines 193-195) change the log to:

```python
    ceiling = target_ceiling(splits["test"]["tokens_a"].to_numpy(),
                             splits["test"]["tokens_b"].to_numpy())
    log.info("ceiling for the averaged target: %.3f as R-squared, %.3f as a correlation",
             ceiling, float(np.sqrt(max(ceiling, 0.0))))
```

In the module docstring (lines 16-22) replace the last two sentences ("so scores against the averaged target belong against 0.742 and scores against a single model against 0.590. Reading one against the other overstates how finished the head is.") with: "so R-squared scores against the averaged target belong against 0.742 and against a single model against 0.590; a *correlation* is bounded by the square root of those (0.861 and 0.768, see :func:`correlation_ceiling`). Reading a correlation against a variance share overstates how finished the head is." In `arena_corpus.py` lines 20-21 change "and that is the ceiling for scores against it" to "and that is the ceiling for an R-squared against it (a correlation is bounded by its square root, ~0.861)".

- [ ] **Step 4: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/prompt_decomposition/test_length.py -q`
Expected: all pass (the two original `target_ceiling` tests are untouched and still pass).

- [ ] **Step 5: Fix the five doc passages**

The wrong claim is "Spearman 0.573 is ~77% of a ≈0.742 ceiling". The right ones are, with the head's numbers 0.510 / 0.525 (one model) and 0.573 (average): **0.664 / 0.684 of 0.768** and **0.573 / 0.861 = 0.665**, and on the variance scale **R-squared 0.399 / 0.742 = 0.54**. Apply these edits (find each passage by its first words; leave the rest of each file alone):

- `docs/prompt_signals.md`, table row for "expected output length": change `ceiling ≈0.742` to `correlation ceiling ≈0.861 (0.742 as a variance share)`.
- `docs/prompt_signals.md`, "**2. Expected output length.**" paragraph: change `against a\nceiling of ≈0.742.` to `against a\ncorrelation ceiling of ≈0.861 (≈0.742 as a variance share).`
- `docs/prompt_signals.md`, from `But the target here is the *average* of the two` through `not 0.03.`, replace with:

```
But the target here is the *average* of the two, which averages away part of
that model-specific noise and is therefore easier to predict than either model
alone. Spearman-Brown puts the *reliability* of a two-item average at **≈0.742**:
the share of the target's variance that is signal. That is the bar for an
R-squared. A correlation is bounded by its square root, **≈0.861** (and by
√0.590 ≈ 0.768 against one model), and the reported Spearman figures are held
against that:

| predicting | correlation ceiling | this head | share |
|---|---|---|---|
| one model's length | √0.590 ≈ 0.768 | 0.510 / 0.525 | ~66% / ~68% |
| the two-model average (reported) | √0.742 ≈ 0.861 | 0.573 | **~67%** |

On the variance scale the same head's R² of 0.399 is ~54% of 0.742. Earlier
versions of this file compared 0.573 with 0.590 (calling it 94%) and then with
0.742 (77%); both divided a correlation by a *variance share*, which is not the
same quantity. This assumes the two arena models are roughly parallel measures
of one prompt-driven signal, so treat the ceilings as upper bounds.
```

- `docs/prompt_signals.md`, the sentence containing `against a ≈0.742 ceiling** for this target`: change to `against a ≈0.861 correlation ceiling** for this target`.
- `docs/prompt_decomposition.md`, the paragraph starting `**There is a ceiling and it is not 1.0.**`: replace with:

```
**There is a ceiling and it is not 1.0.** The two arena models, answering the
same prompt, agree with *each other* at Spearman 0.590: length is substantially a
property of who answers, which no prompt-only feature can reach. The target
averages both models, so its own reliability is higher -- Spearman-Brown gives
≈0.742 as a share of variance -- and a correlation is bounded by the square root
of that, ≈0.861. The shipped head reaches 0.573 with the 1024-d encoder (~67% of
that correlation ceiling; its R² of 0.399 is ~54% of 0.742) and 0.514 with a
384-d one, against 0.353 for the free surface features alone. Roughly 0.29 of
Spearman remains available.
```

- `docs/prompt_decomposition_experiments.md`, the paragraph starting `**The ceiling is ≈0.742, and getting that wrong flattered the result.**` through `hid about 0.17 of remaining headroom.`, replace with:

```
**The ceiling is a correlation of ≈0.861, and getting that wrong flattered the result twice.**
The two models, answering the same prompt, agree with *each other* at Spearman
0.590 — length is substantially a property of who answers. The target is the
average of the two, which averages away part of that noise, so its reliability is
higher: Spearman-Brown gives ≈0.742 for a two-item average. Those are shares of
variance (the bar for an R²); a *correlation* is bounded by their square roots,
√0.590 ≈ 0.768 against one model and √0.742 ≈ 0.861 against the average. Against
one model's length this head scores 0.510/0.525 (~66–68% of 0.768); against the
average it scores 0.573 (~67% of 0.861), and its R² of 0.399 is ~54% of 0.742.
An earlier draft compared 0.573 to 0.590 and claimed 94%; the correction to 0.742
(77%) still divided a correlation by a variance share. About 0.29 of Spearman
remains available, not 0.17.
```

Verify no stale claim remains: `git grep -n "77%\|94%\|≈0.742 ceiling\|0.17 of" -- docs/prompt_signals.md docs/prompt_decomposition.md docs/prompt_decomposition_experiments.md`. Expected: only the sentences quoting the earlier drafts inside the new passages.

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/prompt_decomposition/length_audit.py src/training/prompt_decomposition/data/arena_corpus.py tests/prompt_decomposition/test_length.py docs/prompt_signals.md docs/prompt_decomposition.md docs/prompt_decomposition_experiments.md
git commit -m "fix(evaluation): a correlation ceiling is the square root of the target's reliability" -m "Adds correlation_ceiling and corrects the 'head is at ~77% of the ceiling' claim (~67% as a correlation, ~54% as R-squared)." -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 12: `fill_costs` / `cost_for` price unknown token counts at $0 (LOW-MED; GATED on decision D2)

**Do not start** until (a) decision D2 is made and (b) the in-flight `impute_missing_usage` change to `pricing.py` and `test_pricing.py` is committed by its owner; this task edits the same two files. If D2 is "keep the contract", mark the bug `wontfix` in Task 13 and skip.

**Bug:** `known_bugs/_module_level/training.data.pricing/MOD-001-missing-tokens-priced-at-zero.md`

**Files:**
- Modify: `src/training/data/pricing.py` (`cost_for` and the token handling in `fill_costs`)
- Test: `tests/training/data/test_pricing.py`

**Interfaces:**
- Produces: `cost_for(model_id, input_tokens, output_tokens, prices) -> float` is `NaN` when either token count is `None` / NaN / `pd.NA`; `fill_costs` leaves `cost` NaN for a row with either count unknown.

- [ ] **Step 1: Write the failing tests**

In `tests/training/data/test_pricing.py`, change the last two lines of `test_cost_for_known_tokens` (currently `# missing token counts treated as 0` / `assert cost_for("m", None, None, prices) == 0.0`) to:

```python
    # missing token counts are UNKNOWN, not zero (training/data/schemas.py: missing usage is null)
    assert np.isnan(cost_for("m", None, None, prices))
    assert np.isnan(cost_for("m", 100, None, prices))
    assert np.isnan(cost_for("m", pd.NA, 100, prices))
```

and add:

```python
def test_fill_costs_leaves_unknown_token_counts_nan_not_zero():
    df = pd.DataFrame({
        "model_id": ["llama-3.1-8b-instruct"] * 3,
        "source": ["lm_eval_harness"] * 3,
        "input_tokens": [np.nan, 1000, np.nan],
        "output_tokens": [np.nan, np.nan, 500],
        "cost": [np.nan] * 3,
    })
    out = fill_costs(df, load_config())
    assert out["cost"].isna().all()          # was 0.0, 3e-6 and 3e-6
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data/test_pricing.py -q -k "known_tokens or unknown_token_counts"`
Expected: 2 FAIL. `cost_for("m", None, None, prices)` returns `0.0`; `fill_costs` returns `[0.0, ~3e-06, ~3e-06]`.

- [ ] **Step 3: Implement**

In `src/training/data/pricing.py` add, above `cost_for`:

```python
def _tokens(x) -> float:
    """A token count as a float; unknown (``None`` / NaN / ``pd.NA``) stays NaN, never 0."""
    return float("nan") if x is None or pd.isna(x) else float(x)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").astype(float)
```

Replace the token lines in `cost_for` with:

```python
    it, ot = _tokens(input_tokens), _tokens(output_tokens)
    return it * float(p["input_per_1m"]) / 1e6 + ot * float(p["output_per_1m"]) / 1e6
```

and in `fill_costs` replace the two `it = ... .fillna(0.0)` / `ot = ...` lines with:

```python
    # NaN propagates: a row with an unknown token count keeps an unknown cost, never $0
    it, ot = _column(sub, "input_tokens"), _column(sub, "output_tokens")
```

Update the `fill_costs` docstring's last sentence to: "Rows whose model has no price, or where either token count is unknown, are left NaN."

- [ ] **Step 4: Run the pricing, data and battery tests**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data tests/evaluation -q`
Expected: all pass. (`impute_missing_usage` works on rows with `0` tokens, which this change does not touch.)

- [ ] **Step 5: Commit**

```bash
git add src/training/data/pricing.py tests/training/data/test_pricing.py
git commit -m "fix(data): unknown token counts leave cost NaN instead of pricing them at \$0" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 13: Close out (verification, bug entries, memory)

**Files:**
- Modify: each fixed bug's entry under `known_bugs/`, plus `known_bugs/README.md`
- Modify: `docs/bug_hunt_2026-09.md` (status column)
- Modify: the `known-bugs-hunt.md` memory note

- [ ] **Step 1: Full verification**

Run each and read the output (do not claim success from a subset):

```
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\lint-imports.exe
.venv\Scripts\python.exe known_bugs\_repros\scan_stale_imports.py
```
Expected: the suite passes except failures that already existed before this work (compare against a baseline run made on a clean checkout of the branch point before Task 1; record any pre-existing failures in the handoff rather than fixing them here); import-linter reports all contracts kept; the stale-import scan prints `0 unresolved first-party imports`.

- [ ] **Step 2: Re-run the reproductions**

Run each script and confirm the *fixed* behaviour (repro scripts live in `known_bugs/_repros/`):

| script | expected after the fixes |
|---|---|
| `repro_encode_reseeds_globals.py` | `draws after the first encode identical -> False` |
| `repro_build_store_resume.py` | `batches encoded in run 2: 5` |
| `repro_bank_l2_and_judge_retries.py` | section 1 puts the weight on the near neighbour (`[1. 0.]`-ish, not `[0. 1.]`); section 2 reports 1 attempt made |
| `repro_small_data_edges.py` | (a) `unanimous_half` no longer in the discriminative stratum; (b) `corrected_score [nan, nan, 1.0]` **and** `chance_corrected [False, False, True]` (the script prints both lists) |
| `repro_prompt_decomposition_training.py` | section 0 prints `OK` for both builders (this script installs a shim for sections 1-3; once Task 1 lands, delete the shim block and the alias lines so the script exercises the real path) |

- [ ] **Step 3 (mechanical): Mark the entries fixed**

Run this once, after filling `SHA` and `TEST` from `git log --oneline` and the tests you added (paths are relative to `known_bugs/`; use the one-line commit hash of the task's commit). Bugs skipped or deferred (Task 12 if D2 said no; the D1 OOD-taxonomy residual) are simply left out and stay open.

> **Updated 2026-09-27:** fixed bugs now move out of `known_bugs/` into `fixed_bugs/<Class>/<ID-slug>/{bug.md,fix.md}` (see `fixed_bugs/README.md`) instead of being marked fixed in place. After this script runs, move each marked entry there and add its row to the `fixed_bugs/README.md` index. The prompt-decomposition dataset builders (MOD-001, Task 1) and the taxonomy leak (MOD-001, Task 3) have already moved, so they are no longer in `FIXES` below.

```python
import re, pathlib
ROOT = pathlib.Path("known_bugs")
FIXES = {  # entry path -> (commit sha, regression test id)
    "_module_level/training.nirt.pairwise/MOD-001-auxiliary-battles-include-cold-start-models.md":
        ("SHA", "tests/training/nirt/test_pairwise.py::test_build_pairwise_arrays_drops_battles_with_cold_start_models"),
    "TextEncoder/TEXTENCODER-001-encode-reseeds-global-rngs.md":
        ("SHA", "tests/router/embeddings/test_encoder_rng.py"),
    "_module_level/router.embeddings/MOD-001-build-store-resume-ignores-vector-fingerprint.md":
        ("SHA", "tests/router/embeddings/test_build_store_resume.py"),
    "QueryBank/QUERYBANK-001-similarity-weighting-inverted-on-flat-l2.md":
        ("SHA", "tests/training/retrieval/test_retrieval.py::test_similarity_weighting_favours_the_nearer_neighbour_on_an_l2_bank"),
    "JudgeClient/JUDGECLIENT-001-max-retries-zero-makes-no-attempt.md":
        ("SHA", "tests/evaluation/judge/test_judge_client.py::test_zero_max_retries_still_makes_one_attempt"),
    "_module_level/training.data.chance_correction/MOD-001-n-choices-below-two-gives-nan-but-flagged-corrected.md":
        ("SHA", "tests/training/data/test_chance_correction.py::test_fewer_than_two_choices_is_left_uncorrected_not_nan"),
    "_module_level/evaluation.data.stratify/MOD-001-unanimous-graded-rows-counted-discriminative.md":
        ("SHA", "tests/evaluation/data/test_stratify.py::test_discriminative_excludes_unanimous_graded_rows"),
    "_module_level/training.prompt_decomposition.experiment/MOD-001-run-all-records-no-failures-when-nothing-succeeds.md":
        ("SHA", "tests/prompt_decomposition/test_experiment.py"),
    "_module_level/training.prompt_decomposition.data.sources/MOD-001-missing-synthetic-file-raises-instead-of-warning.md":
        ("SHA", "tests/prompt_decomposition/test_data.py::test_load_synthetic_tasks_warns_and_skips_a_missing_file"),
    "_module_level/evaluation.prompt_decomposition.length_audit/MOD-001-target-ceiling-is-a-reliability-not-a-correlation-ceiling.md":
        ("SHA", "tests/prompt_decomposition/test_length.py"),
    # Task 12, only if D2 was accepted:
    # "_module_level/training.data.pricing/MOD-001-missing-tokens-priced-at-zero.md":
    #     ("SHA", "tests/training/data/test_pricing.py::test_fill_costs_leaves_unknown_token_counts_nan_not_zero"),
}
readme = (ROOT / "README.md").read_text(encoding="utf-8")
lines = readme.split("\n")
for rel, (sha, test) in FIXES.items():
    p = ROOT / rel
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"^- \*\*Status:\*\*.*$", f"- **Status:** fixed ({sha}; regression test `{test}`)", s, count=1, flags=re.M)
    p.write_text(s, encoding="utf-8")
    for i, line in enumerate(lines):
        if rel in line and line.startswith("|"):
            cells = line.split(" | ")
            cells[3] = "fixed"          # | [ID](path) | Class | Sev | Status | One line |
            lines[i] = " | ".join(cells)
(ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
```

Then read `git diff known_bugs/README.md` and check that only the `Status` cells of those rows changed (the row's leading `| [ID](path)` cell is split on `" | "`, so cell index 3 is `Status`).

- [ ] **Step 4: Update the report and memory**

In `docs/bug_hunt_2026-09.md` change each fixed row's Status cell to `fixed (<sha>)`. Update the memory note `known-bugs-hunt.md` (`C:\Users\dogpl\.claude\projects\c--Users-dogpl-VisualCode-Projects-agentic-llm-router\memory\`): replace "none of the 14 are fixed yet" with the fixed list, the still-open items (D1 OOD taxonomy residual, D3 rebuild, Task 12 if deferred) and the commit range.

- [ ] **Step 5: Commit**

```bash
git add known_bugs docs/bug_hunt_2026-09.md
git commit -m "docs(known_bugs): mark the 2026-09-24/25 hunt's fixed bugs" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage** (report table -> task): 1 -> T1; 2 -> T3 (in flight; verification only); 3 -> T2; 4 -> already fixed elsewhere (listed, not planned); 5 -> T4; 6 -> T5; 7 -> T11; 8 -> T12 (gated); 9 -> T6; 10 -> T7; 11 -> T8; 12 -> T9; 13 and 14 -> T10. Every open bug has a task; the residuals that are deliberately *not* fixed (D1 OOD-family exclusion for a separate taxonomy, D3 rebuild and retrain) are named as decisions, not left implicit.

**Placeholder scan:** the only `SHA` tokens are in Task 13's mechanical script and are filled from `git log` at close-out, which cannot be known now. No "TBD" / "handle edge cases" steps; every code step shows the code.

**Type / name consistency:** `preserve_rng_state` (T4 helper, tests, encoder import), `correlation_ceiling` (T11 helper, tests, docstrings), `load_handlabelled(path=None)` (T1 tests and implementation agree), `_StubPairwiseData(battles, q_store, m_store, cold=())` and `cold_start_model_ids()` (T2). The judge message stays "after {attempts} retries" so the existing `match="after 5 retries"` test holds.

**Contract checks made while writing (each changed the plan):** the judge retry test pins 5 attempts (so clamp, not `+1`); `test_cost_for_known_tokens` pins the $0 behaviour (so Task 12 is a decision); the existing resume test writes a progress file without `vector_fingerprint` (so the fallback path); the existing chance-correction warning test matches the substring "no n_choices" (so the new warning text keeps it); `_nirt_query_ids`' existing test passes a frame with no `split` column (so `split=None` must not touch it).

**Known limits:** the `docs/` edits in Task 11 assume the two arena models are roughly parallel measures (stated in the new text); tasks 1 and 10 rely on the tests running with CWD at the repo root (the suite's autouse fixture provides it, a bare `python -c` outside the repo root will not).

## Validation record (2026-09-25)

The plan was executed once in a throwaway copy of `src/` (with `configs/`, `data/`, `docs/`, `scripts/` linked in), so every claim below was observed, not assumed. Nothing in the real tree was modified.

- **Tests fail first, as stated.** Every Step-1 test was extracted from this file (not retyped) and run against the unpatched source: 17 failed for the tasks that remain in this plan (the first draft's two Task 3 tests also failed, and were later replaced by the in-flight fix), and `test_encoder_rng.py` errored at import (the helper does not exist yet). The only 3 that passed are the deliberate no-regression pins (a build with changed vectors restarts; old progress files resume or restart correctly). Task 12's edited and added tests: 2 failed unpatched, 12 passed patched.
- **Patched code passes.** Every extracted test and every existing test in the touched files passed; the full suite against the patched copy gave 950 passed, 2 skipped, 2 failed. The 2 failures (`test_shipped_experiment_configs_all_load_and_are_uniquely_named`, `test_shipped_config_name_matches_its_filename`) come from the scratch copy lacking an `experiments/` link and pass in the real tree.
- **Baseline on the real tree:** 927 passed, 2 failed, and those 2 were the taxonomy tests written by the in-flight change while its source edit had not landed yet (now passing).
- **Layering:** `lint-imports` on the patched copy: 4 contracts kept, 0 broken.
- **Real data:** the patched builders produced `real_only` splits of 1,631 / 306 / 502 with all 400 frozen-eval rows in `test`, and `task` splits of 1,430 / 149 / 201.
- **Anchors:** every "old text" this plan quotes for a source edit matched the current source verbatim (the patch script asserts each one).
- **Task 3:** the in-flight change was verified by running its tests only; its optional Step 3 (`exclude_query_ids`) was validated on top of the in-flight `clustering.py` (fails unpatched with `TypeError`, passes patched). Note that the first-draft version of Task 3 in this record was superseded and never applies.
- **Not executed:** Task 13's mechanical bug-marking script, the doc edits in Task 11 Step 5, and Task 3's optional script flag (Step 4). One gotcha for anyone repeating this in a copy: `pyproject.toml` sets pytest `pythonpath = ["tests", "src"]`, which overrides `PYTHONPATH`; use `-o pythonpath=<dir>` to point a run at a different `src`.
