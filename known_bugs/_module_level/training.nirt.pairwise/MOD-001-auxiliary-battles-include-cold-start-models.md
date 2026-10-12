# MOD-001: the auxiliary pairwise loss trains on battles involving cold-start (held-out) models

- **Status:** reproduced (`known_bugs/_repros/repro_pairwise_cold_leak.py`)
- **Severity:** med (default-off feature: `train.preference.enabled`)
- **Where:** `build_pairwise_arrays` ([pairwise.py:64](../../../src/training/nirt/pairwise.py#L64)) calling `TrainingData.pairwise(..., models="all")` ([facade.py:194-233](../../../src/training/data/facade.py#L194-L233)); consumed by `fit` ([train.py:441-453](../../../src/training/nirt/train.py#L441-L453))
- **Found:** 2026-09-24

## What

The split module promises that cold-start models are "held out entirely. Their
observations appear in NONE of train/val/test" ([splits.py:12-15](../../../src/training/data/splits.py#L12-L15)),
and `TrainingData.correctness()` defaults to `models="warm"` to honour that.
`build_pairwise_arrays` instead passes `models="all"`, so a battle between a
warm model and a cold-start model is included as long as its *query* is in the
train split. (Query splits are computed from the warm models' rows only, so a
warm-vs-cold battle's query can land in train.)

The pairwise term then back-propagates through `model.model_parameters(e_b)`
for the cold model's own profile embedding, i.e. it directly supervises the
projection heads on the very models the cold-start evaluation later treats as
unseen.

## Evidence

Cold-start set: `claude-2.0`, `gpt_4o_mini`, `mistral_7b_instruct_v02`,
`mixtral_8x7b_instruct`, `vicuna-13b`.

```
pairwise(split='train', models='warm'): 131726 battles, 0 involve a cold-start model
pairwise(split='train', models='all'):  136343 battles, 4617 involve a cold-start model
  cold models present: ['claude-2.0', 'vicuna-13b']
```

4,617 battles (3.4%) involve `claude-2.0` / `vicuna-13b`.

## Why it may have been intended

`pairwise.py`'s docstring says `projected` mode is needed "to score Arena
models absent from the correctness training set". Those are *non-pool* models,
which is a legitimate reason to use `models="all"`. But `"all"` also admits the
declared cold-start set, and nothing removes it.

## Impact

Any cold-start number measured on a run with `train.preference.enabled: true`
(zero-shot profile projection vs `warm_mean`, LOMO priors) is optimistic.
Correctness-only runs are unaffected.

## Suggested fix

Drop battles where either participant is in `data.cold_start_model_ids()`
(add a `models="non_cold"` scope to `TrainingData`, or filter in
`build_pairwise_arrays`), and count them in `n_dropped`.
