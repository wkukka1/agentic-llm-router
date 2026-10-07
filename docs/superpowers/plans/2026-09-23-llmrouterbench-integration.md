# LLMRouterBench Integration (pool-expansion E4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LLMRouterBench performance-cost benchmark (arXiv 2601.07206, 13 flagship
models x 10 datasets, real $/token cost) loadable into this project's canonical
`(query_id, model_id, score, metric_type, source, ...)` response schema, as a new, fully
isolated data pipeline — this is the `E4` phase named (but not designed) in
`docs/pool_expansion.md`.

**Architecture:** Follow the `configs/irt_router.yaml` precedent exactly: a new loader
(`load_llmrouterbench`) in `training/data/loaders.py`, a new isolated config
(`configs/llmrouterbench.yaml`, own `paths`/`source_order`, sharing nothing with
`phase0.yaml` or `irt_router.yaml`), and two new controlled-vocabulary entries in
`training/data/schemas.py`. The existing generic scripts
(`scripts/data/build_response_matrix.py`, `build_splits.py`) need no changes — they already
drive off `cfg.get("source_order")`. LLMRouterBench's 13 models share **zero** overlap with
the existing RouterBench/IRT-Router pools, so this is new coverage, not a join.

**Tech Stack:** pandas, the existing `training.data.*` package, `huggingface_hub` (already a
dependency) for the one-time results download.

**Spec:** No separate spec doc — this plan's Architecture section and the investigation
below (file:line citations into both this repo and the vendored
`evaluations/LLMRouterBench/` framework) is the spec of record, produced directly with the
person requesting this feature in the same conversation this plan was written in.

## Global Constraints

- Only the **performance-cost** setting (13 flagship models, real cost) is integrated. The
  performance-only setting (20 lightweight models, no cost) is explicitly out of scope —
  it has no cost signal for this project's `lam`-weighted routing and would double the
  model-registry/config surface for no benefit. Revisit only if asked.
- New source lives in its **own** `configs/llmrouterbench.yaml`, never added to
  `configs/phase0.yaml`'s `sources:`/`multiple_choice:` blocks — this is the same isolation
  decision `configs/irt_router.yaml`'s header documents, and here it also structurally
  avoids a real collision (see Task 3, Review Focus item 1).
- No fabricated values: a record with `score: null` (the reference framework's own
  `BaselineDataLoader` coerces this to `0.0` — see `baselines/data_loader.py:265` in the
  vendored repo) must be **dropped**, not coerced to a wrong-answer score. This project's own
  schema docstring (`training/data/schemas.py:10-13`) already states this rule; the loader
  must follow it even though the reference implementation it's mirroring does not.
- `cost`/`input_tokens`/`output_tokens` pass through verbatim (already real $ and real token
  counts from the benchmark's own collection run) — no price-table lookup, matching how
  `load_irt_router` treats IRT-Router's own `cost` column.

## Review Focus

- **`mmlupro` chance-correction must not inherit RouterBench's `mmlu*` → 4 prefix mapping.**
  `evaluation/MMLUPro/mmlupro.py`'s `ANSWER_PATTERN` allows `[A-J]` (10 choices); RouterBench's
  `configs/phase0.yaml` maps any `mmlu*`-prefixed task to 4. Giving LLMRouterBench its own
  config file avoids the collision structurally (Task 2), but Task 3 pins it with a test that
  would fail loudly if that isolation were ever accidentally removed.
- **Duplicate result files for the same `(dataset, split, model)`.** The reference framework
  re-runs collection and leaves multiple timestamped JSON files per combination
  (`baselines/data_loader.py:204-214`, its own dedup-to-latest logic). Loading every file
  found on disk without deduping would silently double- or triple-count observations for
  whichever combination got re-run, inflating that slice's weight in training.
- **A `score: null` record must be dropped with a visible warning, not defaulted to `0.0`.**
  See Global Constraints above — this is the one place this loader must deliberately NOT
  mirror its reference implementation.
- **An empty/missing `results_dir` must raise, not silently return an empty frame.** Unlike
  `load_lm_harness` (an optional, often-absent supplementary source that warns-and-returns-
  empty), LLMRouterBench is a primary source configured explicitly by the person running it —
  like `load_irt_router`, a missing directory means a real setup mistake (forgot to run the
  download script) and should fail loudly with the fix in the message.
- **A `model_name` outside the configured 13-model pool must be dropped, not merged in.** The
  reference framework's own config comments call out at least one non-model reference row
  (`# reference_models: ["openrouter"]`, `baseline_config_performance_cost.yaml:35-37`)
  that must never enter aggregate model comparisons; an un-filtered loader would treat it as
  a real, 14th routing candidate.

---

## Prerequisite (manual, outside this plan's tasks)

The vendored framework at `evaluations/LLMRouterBench/` is code only — the actual
per-instance result data is not in this repo and is not fetchable by a simple scripted URL
the way `configs/irt_router.yaml`'s Git-LFS media fetch is. Task 5 below writes a download
script against the Hugging Face mirror (`huggingface.co/datasets/NPULH/LLMRouterBench`), but
**running it** (network access, ~multi-GB download) is a manual step for the person executing
this plan, not something the TDD tasks below can verify. Tasks 1-4 and 6 do not depend on it
— they're fully testable now against synthetic fixtures shaped exactly like the real files
(confirmed against `evaluations/LLMRouterBench/baselines/data_loader.py`'s own parsing code,
which is the ground truth for the on-disk JSON shape). Task 7 is the real run, and is gated
on this prerequisite.

---

### Task 1: Schema additions

**Files:**
- Modify: `src/training/data/schemas.py:34-47` (`Source`), `:50-71` (`MetricType`), `:74-85`
  (`METRIC_RANGES`)
- Test: `tests/training/data/test_schemas.py` (new file — no dedicated schema test file
  exists today; small enough to own its file rather than crowd into `test_data.py`)

**Interfaces:**
- Produces: `schemas.Source.LLMROUTERBENCH = "llmrouterbench"` (added to `Source.ALL`, NOT to
  `Source.PAIRWISE` — this source is absolute, not pairwise); `schemas.MetricType.LLM_JUDGE_SCORE
  = "llm_judge_score"` (added to `MetricType.ALL` and `MetricType.CORRECTNESS`, NOT to
  `MetricType.PAIRWISE` — HLE/SimpleQA/ArenaHard's LLM-judge score is an absolute per-`(query,
  model)` correctness judgement, not a pairwise comparison between two models' answers, so it
  belongs in the same bucket as `ACCURACY`/`PASS_AT_1`, unlike `JUDGE_PREFERENCE` which the
  Arena/GPT-4-Judge pairwise sources already use for a different thing).

- [ ] **Step 1: Write the failing test**

```python
"""Controlled-vocabulary contracts in training.data.schemas."""

from __future__ import annotations

from training.data import schemas


def test_llmrouterbench_source_is_registered_and_absolute():
    assert schemas.Source.LLMROUTERBENCH in schemas.Source.ALL
    assert schemas.Source.LLMROUTERBENCH not in schemas.Source.PAIRWISE


def test_llm_judge_score_metric_is_registered_as_correctness_not_pairwise():
    assert schemas.MetricType.LLM_JUDGE_SCORE in schemas.MetricType.ALL
    assert schemas.MetricType.LLM_JUDGE_SCORE in schemas.MetricType.CORRECTNESS
    assert schemas.MetricType.LLM_JUDGE_SCORE not in schemas.MetricType.PAIRWISE
    assert schemas.METRIC_RANGES[schemas.MetricType.LLM_JUDGE_SCORE] == (0.0, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data/test_schemas.py -v`
Expected: FAIL — `AttributeError: type object 'Source' has no attribute 'LLMROUTERBENCH'`

- [ ] **Step 3: Add the constants**

In `src/training/data/schemas.py`, change:

```python
class Source:
    ROUTERBENCH = "routerbench"
    ARENA = "chatbot_arena"
    GPT4_JUDGE = "gpt4_judge"
    LM_HARNESS = "lm_eval_harness"
    IRT_ROUTER = "irt_router"        # the IRT-Router paper's own 20-LLM x 12-dataset suite
    ANCHOR_JUDGE = "anchor_judge"    # manufactured anchor-topology judge on EXISTING
                                     # routerbench: query_ids (docs/anchor_judge.md) --
                                     # query_id is carried through verbatim from the
                                     # correctness data, never re-derived under this source.
    LLMROUTERBENCH = "llmrouterbench"  # arXiv 2601.07206 -- performance-cost setting only
                                       # (13 flagship models, real $/token); see docs/llmrouterbench.md
    ALL = {ROUTERBENCH, ARENA, GPT4_JUDGE, LM_HARNESS, IRT_ROUTER, ANCHOR_JUDGE, LLMROUTERBENCH}

    # Sources whose observations are pairwise preferences, not absolute scores.
    PAIRWISE = {ARENA, GPT4_JUDGE, ANCHOR_JUDGE}
```

and:

```python
class MetricType:
    """Original metric identity. Kept distinguishable end-to-end."""

    ACCURACY = "accuracy"            # 0/1 or graded correctness (RouterBench `performance`)
    EXACT_MATCH = "exact_match"
    F1 = "f1"
    PASS_AT_1 = "pass@1"
    MC_ACCURACY = "mc_accuracy"      # multiple-choice accuracy (chance-correctable)
    ARENA_PREFERENCE = "arena_preference"  # human pairwise outcome in {1.0, 0.5, 0.0}
    JUDGE_PREFERENCE = "judge_preference"  # LLM-judge pairwise outcome in {1.0, 0.5, 0.0}
    LIKELIHOOD_ACC = "acc"           # lm-harness `acc`
    LIKELIHOOD_ACC_NORM = "acc_norm"  # lm-harness `acc_norm`
    LLM_JUDGE_SCORE = "llm_judge_score"  # absolute per-(query,model) LLM-as-judge grade
                                          # (LLMRouterBench HLE/SimpleQA/ArenaHard) -- NOT
                                          # pairwise, unlike JUDGE_PREFERENCE above
    ALL = {
        ACCURACY, EXACT_MATCH, F1, PASS_AT_1, MC_ACCURACY,
        ARENA_PREFERENCE, JUDGE_PREFERENCE, LIKELIHOOD_ACC, LIKELIHOOD_ACC_NORM,
        LLM_JUDGE_SCORE,
    }

    # Pairwise-preference metrics: kept OUT of the absolute response matrix.
    PAIRWISE = {ARENA_PREFERENCE, JUDGE_PREFERENCE}
    # Absolute correctness metrics: the primary AMIRT response signal.
    CORRECTNESS = {ACCURACY, MC_ACCURACY, EXACT_MATCH, F1, PASS_AT_1,
                   LIKELIHOOD_ACC, LIKELIHOOD_ACC_NORM, LLM_JUDGE_SCORE}
```

and add one line to `METRIC_RANGES`:

```python
METRIC_RANGES: dict[str, tuple[float, float]] = {
    MetricType.ACCURACY: (0.0, 1.0),
    MetricType.EXACT_MATCH: (0.0, 1.0),
    MetricType.F1: (0.0, 1.0),
    MetricType.PASS_AT_1: (0.0, 1.0),
    MetricType.MC_ACCURACY: (0.0, 1.0),
    MetricType.ARENA_PREFERENCE: (0.0, 1.0),
    MetricType.JUDGE_PREFERENCE: (0.0, 1.0),
    MetricType.LIKELIHOOD_ACC: (0.0, 1.0),
    MetricType.LIKELIHOOD_ACC_NORM: (0.0, 1.0),
    MetricType.LLM_JUDGE_SCORE: (0.0, 1.0),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data/test_schemas.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/training/data/schemas.py tests/training/data/test_schemas.py
git commit -m "data: register LLMRouterBench source + LLM-judge-score metric type"
```

---

### Task 2: Isolated config `configs/llmrouterbench.yaml`

**Files:**
- Create: `configs/llmrouterbench.yaml`
- Test: `tests/training/data/test_llmrouterbench.py` (created in Task 3 — this task's config
  is exercised there, since a YAML file has nothing to unit-test on its own beyond "does it
  parse", which Task 3's fixture config already proves by construction)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `configs/llmrouterbench.yaml` with `source_order: [llmrouterbench]`,
  `sources.llmrouterbench.{local_dir, pool}`, `multiple_choice.by_task.{gpqa, mmlupro}`,
  `chance_correction`, `split` — the exact key names Task 3's `load_llmrouterbench` and the
  existing `_mc_choices` helper read.

- [ ] **Step 1: Write the config file**

```yaml
# LLMRouterBench (arXiv 2601.07206, Findings of ACL 2026) -- performance-cost setting only.
#
# A fully ISOLATED parallel pipeline, same pattern as configs/irt_router.yaml: 13 flagship
# proprietary models (Claude-4, Gemini-2.5, GPT-5, DeepSeek-V3/R1, Qwen3-235B, GLM-4.6,
# Kimi-K2, Intern-S1) over the 10-dataset performance-cost split (real $/token cost). The
# performance-only setting (20 lightweight open-weight models, no cost) is NOT integrated --
# see docs/pool_expansion.md's E4 entry for why. Nothing here shares state with
# configs/phase0.yaml or configs/irt_router.yaml: separate processed dir, taxonomy, splits,
# indexes. Zero model overlap with either of those pools.
#
# Data provenance: github.com/ynulihao/LLMRouterBench (framework, vendored read-only at
# evaluations/LLMRouterBench/), results tarball from
# huggingface.co/datasets/NPULH/LLMRouterBench, fetched by
# scripts/data/download_llmrouterbench.py into
# evaluations/LLMRouterBench/results/bench/ -- the framework's own expected location (see
# evaluations/LLMRouterBench/results/download.md).
#
# Pipeline:
#   scripts/data/download_llmrouterbench.py       --config configs/llmrouterbench.yaml
#   scripts/data/build_response_matrix.py          --config configs/llmrouterbench.yaml
#   scripts/data/build_splits.py                   --config configs/llmrouterbench.yaml
#   (embeddings / NIRT training -- see docs/llmrouterbench.md)

seed: 42

source_order: [llmrouterbench]

paths:
  raw: evaluations/LLMRouterBench/results
  processed: data/processed/llmrouterbench
  taxonomy: data/taxonomy/llmrouterbench
  splits: data/splits/llmrouterbench
  indexes: indexes/llmrouterbench

sources:
  llmrouterbench:
    # The framework's own on-disk convention: results/bench/<dataset>/<split>/<model>/<ts>.json
    local_dir: evaluations/LLMRouterBench/results/bench
    # The 13-model performance-cost pool (baseline_config_performance_cost.yaml
    # `filters.models`). A result file for any other `model_name` is dropped at load time --
    # in particular this excludes the framework's own "openrouter" reference-model rows,
    # which its config explicitly marks as excluded from aggregation.
    pool:
      - claude-sonnet-4
      - deepseek-v3-0324
      - deepseek-v3.1-terminus
      - deepseek-r1-0528
      - gemini-2.5-flash
      - gemini-2.5-pro
      - gpt-5-chat
      - gpt-5
      - qwen3-235b-a22b-2507
      - qwen3-235b-a22b-thinking-2507
      - glm-4.6
      - kimi-k2-0905
      - intern-s1

# Deliberately its OWN block, not an addition to phase0.yaml's -- gpqa/mmlupro must not
# inherit RouterBench's `mmlu*` -> 4 prefix rule (MMLU-Pro is 10-choice, not 4).
multiple_choice:
  by_task:
    gpqa: 4        # evaluation/GPQA/gpqa.py: ANSWER_PATTERN allows [A-D]
    mmlupro: 10    # evaluation/MMLUPro/mmlupro.py: ANSWER_PATTERN allows [A-J]
  by_prefix: {}

chance_correction:
  method: normalized
  clip: true
  warn_on_missing_choices: true

split:
  presplit: true
  train_fraction: 0.7
  validation_fraction: 0.3
```

- [ ] **Step 2: Commit**

```bash
git add configs/llmrouterbench.yaml
git commit -m "data: add isolated LLMRouterBench pipeline config"
```

---

### Task 3: `load_llmrouterbench` loader

**Files:**
- Modify: `src/training/data/loaders.py` (add a new section after `load_anchor_judge`, before
  the `LOADERS` dict at `:662`)
- Test: `tests/training/data/test_llmrouterbench.py` (new file, mirrors
  `tests/training/data/test_irt_router.py`'s fixture style exactly)

**Interfaces:**
- Consumes: `schemas.Source.LLMROUTERBENCH`, `schemas.MetricType.{ACCURACY, MC_ACCURACY,
  PASS_AT_1, LLM_JUDGE_SCORE}` (Task 1); `configs/llmrouterbench.yaml`'s
  `sources.llmrouterbench.{local_dir, pool}` keys (Task 2); the existing `_mc_choices(eval_name,
  cfg)`, `canonical_model_id(native)`, `make_query_id(source, dataset, text)`,
  `render_prompt(raw)`, `section(cfg, name)` helpers already imported/defined in
  `loaders.py`.
- Produces: `load_llmrouterbench(cfg: Config) -> pd.DataFrame` (canonical response frame),
  registered in the module-level `LOADERS` dict under `schemas.Source.LLMROUTERBENCH` so
  `load_all` / `scripts/data/build_response_matrix.py` pick it up automatically.

- [ ] **Step 1: Write the failing tests**

```python
"""LLMRouterBench loader (arXiv 2601.07206, performance-cost setting).

Fixture JSON shape confirmed against the vendored framework's own reference
parser: evaluations/LLMRouterBench/baselines/data_loader.py:236-269
(`BaselineDataLoader.load_records_iter`) and the file-path convention
documented in evaluations/LLMRouterBench/README.md's "Result File Structure".
"""

from __future__ import annotations

import json

import pytest

from router.config import Config
from training.data import schemas
from training.data.loaders import load_llmrouterbench
from training.data.normalize import make_query_id


def _write_result(root, dataset, split, model, records, *, ts="20260101_000000"):
    out = root / dataset / split / model
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset,
        "split": split,
        "model_name": model,
        "records": records,
    }
    (out / f"{ts}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def bench_config(tmp_path):
    root = tmp_path / "bench"
    # gpqa: MC (4-choice); aime: free-form; one out-of-pool model; one missing score.
    _write_result(root, "gpqa", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q gpqa 1", "prompt": "Answer the following...Q gpqa 1",
         "prediction": "A", "ground_truth": "A", "score": 1.0,
         "prompt_tokens": 100, "completion_tokens": 20, "cost": 0.01},
        {"index": 1, "origin_query": "Q gpqa 2", "prompt": "Answer the following...Q gpqa 2",
         "prediction": "B", "ground_truth": "A", "score": 0.0,
         "prompt_tokens": 90, "completion_tokens": 15, "cost": 0.009},
    ])
    _write_result(root, "gpqa", "test", "claude-sonnet-4", [
        {"index": 0, "origin_query": "Q gpqa 1", "prompt": "Answer the following...Q gpqa 1",
         "prediction": "A", "ground_truth": "A", "score": 1.0,
         "prompt_tokens": 100, "completion_tokens": 25, "cost": 0.02},
    ])
    _write_result(root, "aime", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q aime 1", "prompt": "Q aime 1",
         "prediction": "42", "ground_truth": "42", "score": 1.0,
         "prompt_tokens": 50, "completion_tokens": 500, "cost": 0.05},
        # a failed / unscored generation -- must be dropped, never treated as 0.0
        {"index": 1, "origin_query": "Q aime 2", "prompt": "Q aime 2",
         "prediction": "", "ground_truth": "7", "score": None,
         "prompt_tokens": 50, "completion_tokens": 0, "cost": 0.001},
    ])
    # out-of-pool model (like the framework's own excluded "openrouter" reference rows)
    _write_result(root, "aime", "test", "not-in-pool-model", [
        {"index": 0, "origin_query": "Q aime 1", "prompt": "Q aime 1",
         "prediction": "42", "ground_truth": "42", "score": 1.0,
         "prompt_tokens": 50, "completion_tokens": 500, "cost": 0.05},
    ])
    # duplicate re-run of gpt-5/gpqa under a LATER timestamp -- only this one must survive
    _write_result(root, "gpqa", "test", "gpt-5", [
        {"index": 0, "origin_query": "Q gpqa 1", "prompt": "Answer the following...Q gpqa 1",
         "prediction": "A", "ground_truth": "A", "score": 1.0,
         "prompt_tokens": 100, "completion_tokens": 20, "cost": 0.011},
    ], ts="20260201_000000")

    return Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"llmrouterbench": {
            "local_dir": str(root),
            "pool": ["gpt-5", "claude-sonnet-4"],
        }},
        "multiple_choice": {"by_task": {"gpqa": 4, "mmlupro": 10}, "by_prefix": {}},
        "chance_correction": {"method": "normalized", "clip": True},
    }, root=tmp_path)


def test_loader_emits_canonical_schema(bench_config):
    df = load_llmrouterbench(bench_config)
    for col in schemas.RESPONSE_COLUMNS:
        assert col in df.columns
    assert set(df["source"].unique()) == {schemas.Source.LLMROUTERBENCH}


def test_out_of_pool_model_is_dropped(bench_config):
    df = load_llmrouterbench(bench_config)
    assert set(df["model_id"].unique()) == {"gpt-5", "claude-sonnet-4"}


def test_gpqa_is_mc_with_chance_correctable_choice_count(bench_config):
    df = load_llmrouterbench(bench_config)
    gpqa = df[df["dataset"] == "gpqa"]
    assert (gpqa["metric_type"] == schemas.MetricType.MC_ACCURACY).all()
    assert (gpqa["n_choices"] == 4).all()
    assert (gpqa["is_multiple_choice"]).all()


def test_mmlupro_would_not_inherit_the_routerbench_mmlu_prefix_rule(bench_config):
    """Review Focus item 1: this only proves the config's own by_task entry wins;
    the real protection is that llmrouterbench.yaml never shares a Config load
    with phase0.yaml's `mmlu*` -> 4 by_prefix rule at all."""
    from training.data.loaders import _mc_choices
    assert _mc_choices("mmlupro", bench_config) == 10


def test_aime_is_free_form_accuracy(bench_config):
    df = load_llmrouterbench(bench_config)
    aime = df[df["dataset"] == "aime"]
    assert (aime["metric_type"] == schemas.MetricType.ACCURACY).all()
    assert aime["n_choices"].isna().all()


def test_missing_score_is_dropped_not_defaulted_to_zero(bench_config):
    df = load_llmrouterbench(bench_config)
    aime_gpt5 = df[(df["dataset"] == "aime") & (df["model_id"] == "gpt-5")]
    assert len(aime_gpt5) == 1  # the score=None record never made it in
    assert aime_gpt5["score"].iloc[0] == 1.0


def test_duplicate_result_files_keep_only_the_latest_by_timestamp(bench_config):
    df = load_llmrouterbench(bench_config)
    gpt5_gpqa1 = df[(df["dataset"] == "gpqa") & (df["model_id"] == "gpt-5")
                     & (df["query"].str.contains("Q gpqa 1"))]
    assert len(gpt5_gpqa1) == 1
    assert float(gpt5_gpqa1["cost"].iloc[0]) == 0.011  # the later (20260201) file's value


def test_cost_and_tokens_pass_through_verbatim(bench_config):
    df = load_llmrouterbench(bench_config)
    row = df[(df["dataset"] == "aime") & (df["model_id"] == "gpt-5")].iloc[0]
    assert float(row["cost"]) == 0.05
    assert int(row["input_tokens"]) == 50
    assert int(row["output_tokens"]) == 500


def test_query_id_is_content_addressed_on_origin_query(bench_config):
    qid = make_query_id(schemas.Source.LLMROUTERBENCH, "aime", "Q aime 1")
    df = load_llmrouterbench(bench_config)
    assert qid in set(df["query_id"])


def test_missing_results_dir_raises_with_the_fix_in_the_message(tmp_path):
    cfg = Config({
        "seed": 42,
        "paths": {"processed": str(tmp_path / "proc"), "splits": str(tmp_path / "splits")},
        "sources": {"llmrouterbench": {"local_dir": str(tmp_path / "does-not-exist")}},
        "multiple_choice": {"by_task": {}, "by_prefix": {}},
    }, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="download_llmrouterbench"):
        load_llmrouterbench(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data/test_llmrouterbench.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_llmrouterbench'`

- [ ] **Step 3: Write the loader**

In `src/training/data/loaders.py`, insert this new section between `load_anchor_judge`
(ending at line 656) and the `# combined` block (starting at line 659):

```python
# --------------------------------------------------------------------------- #
# LLMRouterBench (arXiv 2601.07206, Findings of ACL 2026)                     #
# --------------------------------------------------------------------------- #
# dataset_name -> canonical metric_type, per evaluations/LLMRouterBench/README.md's
# "Datasets" table (Performance-Cost Setting) and each dataset's own evaluator under
# evaluations/LLMRouterBench/evaluation/.
_LLMROUTERBENCH_METRIC_TYPES: dict[str, str] = {
    "aime": schemas.MetricType.ACCURACY,
    "livemathbench": schemas.MetricType.ACCURACY,
    "gpqa": schemas.MetricType.MC_ACCURACY,
    "hle": schemas.MetricType.LLM_JUDGE_SCORE,
    "livecodebench": schemas.MetricType.PASS_AT_1,
    "mmlupro": schemas.MetricType.MC_ACCURACY,
    "swe-bench": schemas.MetricType.PASS_AT_1,
    "simpleqa": schemas.MetricType.LLM_JUDGE_SCORE,
    "tau2": schemas.MetricType.ACCURACY,
    "arenahard": schemas.MetricType.LLM_JUDGE_SCORE,
}


def _llmrouterbench_latest_files(results_dir: Path) -> list[Path]:
    """One file per ``(dataset_name, split, model_name)``, keeping the file with the
    lexicographically-latest ``<timestamp>.json`` stem when a combination was re-run --
    mirrors the reference framework's own dedup (``baselines/data_loader.py``'s
    ``_find_result_files`` / ``_get_latest_file_by_timestamp``)."""
    groups: dict[tuple[str, str, str], list[Path]] = {}
    for path in sorted(results_dir.rglob("*.json")):
        try:
            header = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"LLMRouterBench: skipping unreadable {path}: {e}")
            continue
        key = (header.get("dataset_name"), header.get("split"), header.get("model_name"))
        if None in key:
            warnings.warn(
                f"LLMRouterBench: {path} missing dataset_name/split/model_name; skipped"
            )
            continue
        groups.setdefault(key, []).append(path)
    return [max(paths, key=lambda p: p.stem) for paths in groups.values()]


def load_llmrouterbench(cfg: Config) -> pd.DataFrame:
    """LLMRouterBench's performance-cost setting: 13 flagship models over 10 datasets,
    read from the framework's own ``results/bench/<dataset>/<split>/<model>/<ts>.json``
    layout (see ``evaluations/LLMRouterBench/README.md``, "Result File Structure").

    Each file is ``{dataset_name, split, model_name, records: [...]}``; each record
    carries ``index, origin_query, prompt, prediction, ground_truth, score,
    prompt_tokens, completion_tokens, cost``. ``score: null`` records (a failed
    generation) are dropped, never coerced to ``0.0`` -- see
    ``training.data.schemas``'s "missing values are never fabricated" rule. Multiple
    result files for the same ``(dataset, split, model)`` are deduplicated to the
    latest by filename timestamp.
    """
    src = section(cfg, "sources").get("llmrouterbench", {})
    results_dir = cfg.resolve(
        src.get("local_dir", "evaluations/LLMRouterBench/results/bench")
    )
    pool = {str(m) for m in (src.get("pool") or [])}
    if not results_dir.exists():
        raise FileNotFoundError(
            f"LLMRouterBench results dir not found: {results_dir}\n"
            f"Run: python scripts/data/download_llmrouterbench.py --config <this config>"
        )

    dropped_missing_score = 0
    records: list[dict] = []
    for path in _llmrouterbench_latest_files(results_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(payload["dataset_name"])
        split = payload.get("split")
        model_native = str(payload["model_name"])
        if pool and model_native not in pool:
            continue
        model_id = canonical_model_id(model_native)
        metric_type = _LLMROUTERBENCH_METRIC_TYPES.get(dataset, schemas.MetricType.ACCURACY)
        is_mc = metric_type == schemas.MetricType.MC_ACCURACY
        n_choices = _mc_choices(dataset, cfg) if is_mc else None

        for row in payload.get("records", []):
            if row.get("score") is None:
                dropped_missing_score += 1
                continue
            text = row.get("origin_query") or row.get("prompt") or ""
            prompt = render_prompt(text)
            qid = make_query_id(schemas.Source.LLMROUTERBENCH, dataset, prompt)
            records.append({
                "query_id": qid,
                "model_id": model_id,
                "query": prompt,
                "score": float(row["score"]),
                "metric_type": metric_type,
                "dataset": dataset,
                "split": split,
                "source": schemas.Source.LLMROUTERBENCH,
                "is_multiple_choice": is_mc,
                "n_choices": n_choices,
                "input_tokens": row.get("prompt_tokens"),
                "output_tokens": row.get("completion_tokens"),
                "latency": None,
                "cost": row.get("cost"),
                "metadata": {
                    "native_model_name": model_native,
                    "record_index": row.get("index"),
                    "result_file": path.name,
                },
            })

    if dropped_missing_score:
        warnings.warn(
            f"LLMRouterBench: dropped {dropped_missing_score} record(s) with score=null "
            f"(a failed generation the framework itself would otherwise silently score 0.0)"
        )
    return schemas.coerce_response_frame(pd.DataFrame.from_records(records))
```

Then register it in the `LOADERS` dict:

```python
LOADERS = {
    schemas.Source.ROUTERBENCH: load_routerbench,
    schemas.Source.ARENA: load_arena,
    schemas.Source.GPT4_JUDGE: load_gpt4_judge,
    schemas.Source.LM_HARNESS: load_lm_harness,
    schemas.Source.IRT_ROUTER: load_irt_router,
    schemas.Source.ANCHOR_JUDGE: load_anchor_judge,
    schemas.Source.LLMROUTERBENCH: load_llmrouterbench,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data/test_llmrouterbench.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Run the full data test suite to check for regressions**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data -q`
Expected: all pass, no change to any existing test's outcome

- [ ] **Step 6: Commit**

```bash
git add src/training/data/loaders.py tests/training/data/test_llmrouterbench.py
git commit -m "data: add load_llmrouterbench loader for the performance-cost setting"
```

---

### Task 4: Model registry entries

**Files:**
- Modify: `configs/model_registry.yaml` (append a new section)
- Test: `tests/training/data/test_model_registry_llmrouterbench.py` (new — no dedicated
  `model_registry` test file exists today, mirrors the tiny, focused style of
  `tests/training/data/test_benchmark_id.py`)

**Interfaces:**
- Consumes: nothing from earlier tasks (independent of the loader; this only affects what
  `canonical_model_id` returns for these 13 strings — the loader already works without it,
  since unregistered strings pass through slugified per `model_registry.py:88-99`).
- Produces: 13 new `canonical_id` entries so `is_canonical("gpt-5")` etc. is `True` and the
  `uncanonicalized_model` quality check (`training/data/quality.py:155`) doesn't flag them.

- [ ] **Step 1: Write the failing test**

```python
"""Every LLMRouterBench performance-cost model must be a registered canonical id --
otherwise training.data.quality's uncanonicalized_model check flags it on every run."""

from __future__ import annotations

from training.data.model_registry import canonical_model_id, is_canonical

_LLMROUTERBENCH_MODELS = [
    "claude-sonnet-4", "deepseek-v3-0324", "deepseek-v3.1-terminus", "deepseek-r1-0528",
    "gemini-2.5-flash", "gemini-2.5-pro", "gpt-5-chat", "gpt-5",
    "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507", "glm-4.6",
    "kimi-k2-0905", "intern-s1",
]


def test_every_llmrouterbench_model_is_a_registered_canonical_id():
    for native in _LLMROUTERBENCH_MODELS:
        cid = canonical_model_id(native)
        assert is_canonical(cid), f"{native!r} -> {cid!r} not registered as canonical"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data/test_model_registry_llmrouterbench.py -v`
Expected: FAIL — several assertions fail (none of the 13 are registered yet)

- [ ] **Step 3: Add the registry section**

Append to `configs/model_registry.yaml` (after the existing `# ---- pool-expansion E3 ----`
section, following the same one-entry-per-model shape used throughout the file):

```yaml
  # ---- pool-expansion E4: LLMRouterBench performance-cost flagship pool (13) ----
  # Native strings are exact provider-facing labels from
  # evaluations/LLMRouterBench/config/baseline_config_performance_cost.yaml `filters.models`.
  # Identity mappings (confidence: high) -- no cross-source merge decisions needed, since
  # this pool shares no checkpoint with RouterBench/Arena/IRT-Router.
  - canonical_id: claude-sonnet-4
    provider: anthropic
    family: claude-4
    version: "sonnet"
    aliases:
      - { native: "claude-sonnet-4", confidence: high }

  - canonical_id: deepseek-v3-0324
    provider: deepseek
    family: deepseek-v3
    version: "0324"
    aliases:
      - { native: "deepseek-v3-0324", confidence: high }

  - canonical_id: deepseek-v3.1-terminus
    provider: deepseek
    family: deepseek-v3.1
    version: "terminus"
    aliases:
      - { native: "deepseek-v3.1-terminus", confidence: high }

  - canonical_id: deepseek-r1-0528
    provider: deepseek
    family: deepseek-r1
    version: "0528"
    aliases:
      - { native: "deepseek-r1-0528", confidence: high }

  - canonical_id: gemini-2.5-flash
    provider: google
    family: gemini-2.5
    version: "flash"
    aliases:
      - { native: "gemini-2.5-flash", confidence: high }

  - canonical_id: gemini-2.5-pro
    provider: google
    family: gemini-2.5
    version: "pro"
    aliases:
      - { native: "gemini-2.5-pro", confidence: high }

  - canonical_id: gpt-5-chat
    provider: openai
    family: gpt-5
    version: "chat"
    aliases:
      - { native: "gpt-5-chat", confidence: high }

  - canonical_id: gpt-5
    provider: openai
    family: gpt-5
    version: "medium"
    notes: "LLMRouterBench's `GPT-5-medium` pool entry; native string is bare `gpt-5`."
    aliases:
      - { native: "gpt-5", confidence: high }

  - canonical_id: qwen3-235b-a22b-2507
    provider: alibaba
    family: qwen3-235b-a22b
    version: "2507"
    aliases:
      - { native: "qwen3-235b-a22b-2507", confidence: high }

  - canonical_id: qwen3-235b-a22b-thinking-2507
    provider: alibaba
    family: qwen3-235b-a22b
    version: "thinking-2507"
    aliases:
      - { native: "qwen3-235b-a22b-thinking-2507", confidence: high }

  - canonical_id: glm-4.6
    provider: zhipu
    family: glm-4
    version: "4.6"
    aliases:
      - { native: "glm-4.6", confidence: high }

  - canonical_id: kimi-k2-0905
    provider: moonshot
    family: kimi-k2
    version: "0905"
    aliases:
      - { native: "kimi-k2-0905", confidence: high }

  - canonical_id: intern-s1
    provider: shanghai-ai-lab
    family: intern-s1
    version: null
    aliases:
      - { native: "intern-s1", confidence: high }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data/test_model_registry_llmrouterbench.py -v`
Expected: PASS

- [ ] **Step 5: Run the full model-registry-adjacent test suite to check for regressions**

Run: `.venv\Scripts\python.exe -m pytest tests/training/data -q`
Expected: all pass — in particular any existing "N canonical models" count assertions (grep
for a hardcoded `69` or similar before running, per `docs/data_sources.md`'s "69 as of
2026-08-27" count, and update that count in the doc if such an assertion exists in code, not
just prose)

- [ ] **Step 6: Commit**

```bash
git add configs/model_registry.yaml tests/training/data/test_model_registry_llmrouterbench.py
git commit -m "data: register the 13 LLMRouterBench performance-cost models"
```

---

### Task 5: Download script

**Files:**
- Create: `scripts/data/download_llmrouterbench.py`

**Interfaces:**
- Consumes: `configs/llmrouterbench.yaml`'s `sources.llmrouterbench.local_dir` (Task 2);
  `training.cli.{base_parser, get_config, info}` (existing shared CLI helpers, same ones
  `download_irt_router.py` uses).
- Produces: populates `<local_dir>` (default `evaluations/LLMRouterBench/results/bench/`)
  from the Hugging Face dataset mirror.

Not TDD — matches the existing convention for `scripts/data/download_*.py`: no test file
exists for `download_irt_router.py` or `download_arena.py` either (pure network I/O against
an external host; nothing to usefully mock without testing `huggingface_hub` itself). Review
by reading, not by running, same as its siblings.

- [ ] **Step 1: Write the script**

```python
"""Obtain the LLMRouterBench performance-cost benchmark results
(arXiv 2601.07206, Findings of ACL 2026).

The paper's authors publish pre-collected result JSON files as a tarball on the Hugging
Face Hub (huggingface.co/datasets/NPULH/LLMRouterBench) -- see
evaluations/LLMRouterBench/README.md's "Data Download" section for the alternative Google
Drive / Baidu Netdisk mirrors if this one is unavailable. This script fetches and extracts
that tarball into ``sources.llmrouterbench.local_dir``.

    python scripts/data/download_llmrouterbench.py --config configs/llmrouterbench.yaml

This project only uses the performance-cost setting (13 flagship models); the tarball
contains both settings, so this only extracts the ``results/bench/`` subtree relevant here
-- run `du -sh` on the extracted directory to check size before committing disk space, the
full release is multi-GB.
"""

from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from training.cli import base_parser, get_config, info
from training.data.config_section import section  # same helper loaders.py uses

REPO_ID = "NPULH/LLMRouterBench"
ARCHIVE_FILENAME = "bench-release.tar.gz"


def main() -> int:
    parser = base_parser(__doc__)
    args = parser.parse_args()
    cfg = get_config(args)

    src = section(cfg, "sources").get("llmrouterbench", {})
    dest = cfg.resolve(src.get("local_dir", "evaluations/LLMRouterBench/results/bench"))

    info(f"downloading {ARCHIVE_FILENAME} from {REPO_ID} ...")
    archive_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=ARCHIVE_FILENAME)

    info(f"extracting into {dest.parent} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest.parent)

    if not dest.exists():
        info(f"ERROR: expected {dest} after extraction; check the archive's internal layout")
        return 1
    n_files = sum(1 for _ in dest.rglob("*.json"))
    info(f"done: {n_files} result files under {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Confirm `section` is actually importable from where this script assumes**

`loaders.py` imports `section` via `from router.config import Config, section` (see
`loaders.py:34`), not from a `training.data.config_section` module — that module doesn't
exist. Fix the import before committing:

```python
from router.config import section
```

(This step exists because the plan author has not run this script — it depends on
`huggingface_hub`'s actual API surface and the real tarball's internal directory layout,
neither of which is verifiable without network access in this session. Whoever runs Task 7
should expect to fix the extracted-path assumption (`tf.extractall(dest.parent)` assumes the
tarball's top-level entry is literally `bench/`, matching `evaluations/LLMRouterBench/
README.md`'s documented layout — confirm this against the real archive before trusting the
`n_files` count.)

- [ ] **Step 3: Commit**

```bash
git add scripts/data/download_llmrouterbench.py
git commit -m "data: add LLMRouterBench results download script"
```

---

### Task 6: Docs

**Files:**
- Create: `docs/llmrouterbench.md`
- Modify: `docs/pool_expansion.md:100` (the E4 table row)

**Interfaces:**
- Consumes: everything decided in Tasks 1-5 (this task only writes prose/tables describing
  them — no code).

- [ ] **Step 1: Write `docs/llmrouterbench.md`**

```markdown
# LLMRouterBench (pool-expansion E4)

Integrates the **performance-cost setting** of LLMRouterBench
([arXiv 2601.07206](https://arxiv.org/abs/2601.07206), Findings of ACL 2026): 13 flagship
proprietary models over 10 datasets, with real per-token cost. A fully isolated parallel
pipeline, the same pattern as [irt_router.md](irt_router.md) — nothing shares state with
`configs/phase0.yaml` or `configs/irt_router.yaml`. The performance-only setting (20
lightweight open-weight models, no cost signal) is **not** integrated — see
[pool_expansion.md](pool_expansion.md)'s E4 row for why.

## Data provenance

Framework vendored read-only at `evaluations/LLMRouterBench/` (`github.com/ynulihao/
LLMRouterBench`). Result data fetched separately by `scripts/data/download_llmrouterbench.py`
from `huggingface.co/datasets/NPULH/LLMRouterBench` into
`evaluations/LLMRouterBench/results/bench/` — the framework's own expected location.

Each result file is `results/bench/<dataset>/<split>/<model>/<timestamp>.json`:
`{dataset_name, split, model_name, records: [{index, origin_query, prompt, prediction,
ground_truth, score, prompt_tokens, completion_tokens, cost}]}`. A `score: null` record (a
failed generation) is dropped, never coerced to `0.0` the way the framework's own
`BaselineDataLoader` does — see `training/data/schemas.py`'s "missing values are never
fabricated" rule.

## Datasets (10) and metric types

| dataset | category | metric_type | multiple choice |
|---|---|---|---|
| aime | Math | `accuracy` | no |
| livemathbench | Math | `accuracy` | no |
| gpqa | Knowledge | `mc_accuracy` | yes, 4-choice (`ANSWER_PATTERN [A-D]`) |
| hle | Knowledge | `llm_judge_score` | no |
| livecodebench | Code | `pass@1` | no |
| mmlupro | Knowledge | `mc_accuracy` | yes, 10-choice (`ANSWER_PATTERN [A-J]`) |
| swe-bench | Code | `pass@1` | no |
| simpleqa | Knowledge | `llm_judge_score` | no |
| tau2 | Tool Use | `accuracy` | no |
| arenahard | Instruction Following | `llm_judge_score` | no |

`gpqa`/`mmlupro`'s choice counts live in `configs/llmrouterbench.yaml`'s own
`multiple_choice.by_task` block, not `phase0.yaml`'s — RouterBench's `mmlu*` → 4 `by_prefix`
rule would otherwise wrongly apply to `mmlupro` (10-choice, not 4).

## Model pool (13, zero overlap with RouterBench/IRT-Router)

claude-sonnet-4, deepseek-v3-0324, deepseek-v3.1-terminus, deepseek-r1-0528,
gemini-2.5-flash, gemini-2.5-pro, gpt-5-chat, gpt-5, qwen3-235b-a22b-2507,
qwen3-235b-a22b-thinking-2507, glm-4.6, kimi-k2-0905, intern-s1 — registered in
`configs/model_registry.yaml`'s "pool-expansion E4" section as identity aliases (no
cross-source merge decisions needed).

## Pipeline

```
python scripts/data/download_llmrouterbench.py   --config configs/llmrouterbench.yaml
python scripts/data/build_response_matrix.py      --config configs/llmrouterbench.yaml
python scripts/data/build_splits.py               --config configs/llmrouterbench.yaml
```

## Results

**Not yet run in this environment** — Task 5/7 of the E4 implementation plan
(`docs/superpowers/plans/2026-09-23-llmrouterbench-integration.md`) require downloading the
real results tarball, which needs network access this planning session didn't have. Once run,
this section should report: row/query/model counts (mirroring the tables in
[pool_expansion.md](pool_expansion.md)'s E0-E2 sections), the data-quality report from
`build_response_matrix.py`'s `check_responses` output, and — after a retrain — the same
oracle-ceiling / ZOIB-NLL metrics those sections report, to show what this data source moved.
```

- [ ] **Step 2: Update the E4 row in `docs/pool_expansion.md`**

Change line 100 from:

```
| **E4** | an external benchmark source (RouteLLM / MixInstruct / the `evaluations/LLMRouterBench` submodule) |
```

to:

```
| **E4** | LLMRouterBench performance-cost setting (13 flagship models, 10 datasets, real cost) -- see [llmrouterbench.md](llmrouterbench.md) |
```

- [ ] **Step 3: Commit**

```bash
git add docs/llmrouterbench.md docs/pool_expansion.md
git commit -m "docs: write up the LLMRouterBench (E4) integration"
```

---

### Task 7: Real run (manual, gated on the Prerequisite)

Not part of the automated task sequence — this is what the person executing this plan does
once `scripts/data/download_llmrouterbench.py` has actually been run successfully against
live network access:

1. `python scripts/data/build_response_matrix.py --config configs/llmrouterbench.yaml` —
   read its printed `check_responses` report; if it has errors (not just warnings), fix the
   loader before proceeding, don't silently write bad output.
2. `python scripts/data/build_splits.py --config configs/llmrouterbench.yaml`
3. Fill in `docs/llmrouterbench.md`'s "Results" section with the real row/query/model counts
   from step 1's output and `data/processed/llmrouterbench/{responses,queries,models}.parquet`.
4. Only then decide whether to build embeddings / train a NIRT model against this source —
   that's a separate follow-up, out of scope for this plan (mirrors how E0-E2 each landed the
   data change first and reported retrain numbers in a later, separate pass).

---

## Self-Review

**1. Spec coverage.** Loader (Task 3), config (Task 2), schema vocabulary (Task 1), model
registry (Task 4), download path (Task 5), docs (Task 6) — every piece named in the
Architecture section has an owning task. The "only performance-cost, not performance-only"
scope decision from Global Constraints is enforced by Task 2's config only listing the 13-model
pool and Task 4 only registering those 13 — there's no task that could accidentally pull in
the 20-model setting.

**2. Placeholder scan.** No TBD/"add appropriate"/"similar to Task N" — Task 5's Step 2 is an
explicit, named, unverified assumption (flagged as such, with the exact fix) rather than a
vague placeholder; Task 6's "Results" section is real, honest content stating current status,
not a stand-in for a deliverable.

**3. Type consistency.** `load_llmrouterbench(cfg: Config) -> pd.DataFrame` used consistently
across Tasks 3, 5, 6. `schemas.MetricType.LLM_JUDGE_SCORE` / `schemas.Source.LLMROUTERBENCH`
spelled identically everywhere they're referenced (Tasks 1, 3, 6).

**4. Review Focus.** All five items have an owning test: mmlupro prefix collision
(`test_mmlupro_would_not_inherit_the_routerbench_mmlu_prefix_rule`), duplicate result files
(`test_duplicate_result_files_keep_only_the_latest_by_timestamp`), null score
(`test_missing_score_is_dropped_not_defaulted_to_zero`), missing results dir
(`test_missing_results_dir_raises_with_the_fix_in_the_message`), out-of-pool model
(`test_out_of_pool_model_is_dropped`) — all in Task 3.
