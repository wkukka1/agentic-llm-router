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
