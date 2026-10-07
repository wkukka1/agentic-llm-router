# MLPROUTERTRAINER-005: `train()` saves then immediately reloads its own artifact

- **Status:** confirmed (traced; code_review finding TT-04)
- **Severity:** low (inefficiency; correctness depends on save/load being a perfect round trip)
- **Class:** `MLPRouterTrainer` ([training/trainers/mlp_router_trainer.py:58-67](../../../src/training/trainers/mlp_router_trainer.py#L58-L67))
- **Found:** 2026-09-22

## What

```python
store = LocalArtifactStore(self.runs_dir)
store.save(RouterModelArtifact(...))
return store.load(config.run_name)
```

`train()` builds a `RouterModelArtifact` in memory, writes it to disk, then
immediately reads it back to produce the return value, instead of returning
the in-memory artifact it just built. This is a full serialize/deserialize
round trip (`torch.save`/`torch.load` plus JSON) on every training call, and
makes `train()`'s return value implicitly dependent on `load()` being a
lossless inverse of `save()` — any store bug that drops or defaults a field
on load (see the `LocalArtifactStore` bugs) now also corrupts `train()`'s
direct return value, not just a later reload.

## Failure scenario

Not wrong today by itself, but it compounds with any `LocalArtifactStore`
load-side bug (e.g. [LOCALARTIFACTSTORE-004](../../../fixed_bugs/ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-004-load-reads-model-pt-twice/bug.md))
and adds avoidable disk I/O and torch de/serialization cost to every
training run.

## Suggested fix

Return the artifact `train()` already built in memory; let callers that
actually need the on-disk round trip call `store.load()` themselves.
