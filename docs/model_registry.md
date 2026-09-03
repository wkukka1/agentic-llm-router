# Canonical model registry

File: [`configs/model_registry.yaml`](../configs/model_registry.yaml)
Loader: [`src/router/data/model_registry.py`](../src/router/data/model_registry.py)

## Why

Three sources spell models differently and at different granularities. The AMIRT
model must not think `gpt-4-0314`, `gpt-4-0613`, `gpt-4-1106-preview` are one
test-taker — nor should it split a single checkpoint that two sources both used.

## Merge policy

Each alias carries a `confidence`:

| confidence | behavior |
|------------|----------|
| `high`     | merged to `canonical_id` at load time |
| `medium` / `low` | recorded (`alias_confidence()`), **not** merged; the native string becomes its own `canonical_id` |

`high` is reserved for (a) exact provider-label identity and (b) unambiguous
provider-prefix rewrites (`meta/llama-2-70b-chat` → `llama-2-70b-chat`), plus a
small number of cross-source merges where only one checkpoint ever existed.

## Cross-source decisions (needing periodic human review)

| models | decision | reason |
|--------|----------|--------|
| RB `mistralai/mixtral-8x7b-chat`, Arena/Judge `mixtral-8x7b-instruct-v0.1` | **merged** → `mixtral-8x7b-instruct` | only one Mixtral-8x7B-Instruct checkpoint (v0.1) ever existed |
| RB `claude-v2` vs Arena `claude-2.0`, `claude-2.1` | **kept separate** | RouterBench label is ambiguous between 2.0 and 2.1 |
| RB `claude-v1` vs Arena `claude-1` | kept separate | unconfirmed they are the same snapshot |
| RB `claude-instant-v1` vs Arena `claude-instant-1` | kept separate | same |
| RB `code-llama-34b-instruct` vs Arena `codellama-34b-instruct` | kept separate (probably mergeable) | not yet reviewed |
| RB `mistral-7b-instruct` vs Arena `mistral-7b-instruct-v0.1` / `-v0.2` | kept separate | RouterBench doesn't say which checkpoint |
| RB `wizardlm-13b-v1.2` vs Arena `wizardlm-13b` | kept separate | Arena version unconfirmed |
| `gpt-4-1106-preview` in RouterBench and in GPT-4-Judge | **merged** (same id) | identical version string |

Every model observed in the data has a `canonical_id` (69 as of 2026-08-27); the
quality check `uncanonicalized_model` fails loudly if a new source introduces an
unmapped string.

## API

```python
from router.data.model_registry import (
    canonical_model_id, model_info, is_canonical, alias_confidence, all_canonical_ids
)
canonical_model_id("mistralai/mixtral-8x7b-chat")   # -> "mixtral-8x7b-instruct"
alias_confidence("claude-2.1")                      # -> "high"  (identity alias)
model_info("gpt-4-1106-preview")                    # ModelInfo(provider="openai", family="gpt-4", version="1106-preview", ...)
```
