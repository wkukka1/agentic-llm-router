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
`{dataset_name, split, model_name, demo, records: [{index, origin_query, prompt, prediction,
ground_truth, score, prompt_tokens, completion_tokens, cost}]}`. A `score: null` record (a
failed generation) is dropped, never coerced to `0.0` the way the framework's own
`BaselineDataLoader` does — see `training/data/schemas.py`'s "missing values are never
fabricated" rule.

The loader also drops, each with a warning naming what was dropped: files with `demo: true`
(smoke-test runs, which can otherwise carry a later timestamp than a real one and evict it
from the dedup-to-latest step); a `model_name` outside `sources.llmrouterbench.pool`; a
`split` outside `sources.llmrouterbench.splits`; and a `dataset_name` outside the ten in the
table below (the raw results tree includes datasets from the *performance-only* setting or
excluded ones like `arc-agi` that must never enter this pool). If every file present gets
filtered out, or none exist at all, `load_llmrouterbench` raises with an actionable message
rather than failing downstream with a generic "missing columns" error.

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

The last step uses content-hash group splitting (`split.presplit: false`), not IRT-Router's
pre-partitioned mode — LLMRouterBench's own `split` field is a run identifier
(`test`/`hybrid`/`v1`/`test_3000`/`verified`), not a train/test partition, so there is no
`metadata['origin']` for `make_presplit` to key off.

## Results

First run: October 2026, on the 2026 release tarball (1.28 GB, 6.5 GB extracted).

**Data:**

* 12,336 queries × 13 models;
* 159,688 NIRT observations;
* splits 8,602 / 1,896 / 1,838 train / validation / test (by query; content-hash group split);
* 43 taxonomy clusters, 33.9% noise.

**What running it changed:**

* `download_llmrouterbench.py` accepts the tarball's new top-level
  `bench-release/` directory.
* The loader sets 10 negative `completion_tokens` / `cost` values to unknown
  (`None`) and keeps their scores.
* The loader skips files outside the pool, splits and datasets **by path,
  before reading them**. Most of the release is outside this setting, and
  one excluded 14 MB `arc-agi` file takes ~15 minutes to `json.loads`.
  The skip is reported in a warning that names what was skipped.
* The config gained `embedding` / `profiles` / `clustering` / `taxonomy` /
  `retrieval` blocks, so `training.phase0 --sources llmrouterbench
  --embeddings --taxonomy --query-bank` runs end to end.

**Routing, test split, cheapest-within-margin rule:**

| Policy | Quality | Cost / 1k |
|---|---|---|
| always gpt-5 (strongest) | 0.596 | $28.30 |
| kNN k=25, margin 0.01 | 0.615 | $24.40 |
| kNN k=25, margin 0.05 | 0.596 | $17.25 |
| kNN k=25, margin 0.08 | 0.580 | $12.06 |
| NIRT 2-d projected, margin 0.01 | 0.610 | $29.06 |
| NIRT 2-d projected, margin 0.05 | 0.575 | $11.37 |
| oracle (cheapest correct) | 0.817 | $3.96 |

* NIRT on its own scores AUC 0.78, ECE 0.044, and 0.105 lower BCE than the
  per-model mean.
* kNN matches or beats it at every margin, which is why the gateway uses kNN
  (docs/gateway.md).
