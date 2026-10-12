# MLPROUTERTRAINER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`MLPArtifactPayload` now carries `pathway`/`query_pathway`/`query_features` (defaulted for backward compatibility with artifacts saved before this field existed), and `MLPRouterTrainer.train()` records the resolved values from `kw` (the same dict already passed to `fit_mlp_router`) onto the payload it saves. Covered by `tests/router/models/test_artifacts.py::test_mlp_artifact_payload_round_trips_pathway_fields` / `test_mlp_artifact_payload_from_state_defaults_pathway_for_old_artifacts` and `tests/training/trainers/test_mlp_router_trainer.py::test_train_persists_the_pathway_hyperparameters`.

## Fix suggested when the bug was filed

Pass `hyperparameters=dict(config.hyperparameters)` (plus the resolved seed/epochs) to the
`RouterModelArtifact`, and extend `MLPArtifactPayload` (or its config) with the data
pathways so `from_artifact` can restore them.

## Regression tests

`tests/router/models/test_artifacts.py:60` `test_mlp_artifact_payload_round_trips_pathway_fields`

```python
def test_mlp_artifact_payload_round_trips_pathway_fields():
    """MLPROUTERTRAINER-001: pathway/query_pathway/query_features must survive
    a to_state/from_state round trip so the serving side can restore them."""
    payload = MLPArtifactPayload(
        state_dict={}, model_ids=["a", "b"], in_dim=8, hidden=16, dropout=0.1,
        pathway="bert", query_pathway="bert-query", query_features="len",
    )
    restored = MLPArtifactPayload.from_state(payload.to_state())
    assert restored.pathway == "bert"
    assert restored.query_pathway == "bert-query"
    assert restored.query_features == "len"
```

`tests/router/models/test_artifacts.py:73` `test_mlp_artifact_payload_from_state_defaults_pathway_for_old_artifacts`

```python
def test_mlp_artifact_payload_from_state_defaults_pathway_for_old_artifacts():
    """Backward compatibility: an artifact saved before this field existed
    has no pathway/query_pathway/query_features keys at all."""
    old_state = {
        "format": "mlp", "state_dict": {}, "model_ids": ["a"],
        "in_dim": 4, "hidden": 8, "dropout": 0.1,
    }
    restored = MLPArtifactPayload.from_state(old_state)
    assert restored.pathway == "irt"
    assert restored.query_pathway is None
    assert restored.query_features is None
```

`tests/training/trainers/test_mlp_router_trainer.py:56` `test_train_persists_the_pathway_hyperparameters`

```python
def test_train_persists_the_pathway_hyperparameters(tmp_path):
    """MLPROUTERTRAINER-001: pathway/query_pathway/query_features must survive
    into the saved artifact, not just fit_mlp_router's in-memory call."""
    d = FakeTrainingData(make_split_obs(seed=15), query_dim=16)
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run-pathway",
        hyperparameters={"hidden": 16, "epochs": 2, "pathway": "irt", "query_features": None},
    )
    artifact = trainer.train(d, config)
    assert artifact.payload.pathway == "irt"
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/models/artifacts.py`

```diff
--- a/src/router/models/artifacts.py
+++ b/src/router/models/artifacts.py
@@ -127,13 +127,24 @@ class MLPArtifactPayload(ArtifactPayload):
     persisted nothing before :class:`~training.trainers.mlp_router.MLPRouterTrainer`
     -- there is no prior on-disk convention this wraps, this *is* the
     convention. Rebuild with ``build_mlp_router(in_dim, len(model_ids), hidden,
-    dropout)`` + ``load_state_dict``."""
+    dropout)`` + ``load_state_dict``.
+
+    ``pathway``/``query_pathway``/``query_features`` mirror the NIRT
+    artifact's ``config["data"]`` keys (MLPROUTERTRAINER-001) -- without them
+    ``MLPRouter.from_artifact`` can't know which embedding space the model
+    was trained on and silently serves the wrong one (MLPROUTER-001).
+    Defaulted in :meth:`from_state` so an artifact saved before this field
+    existed still loads, at the same ``pathway="irt"`` default the router
+    constructor used before either bug was fixed."""
 
     state_dict: dict[str, Any]
     model_ids: list[str]
     in_dim: int
     hidden: int
     dropout: float
+    pathway: str = "irt"
+    query_pathway: Any = None
+    query_features: Any = None
 
     def to_state(self) -> dict[str, Any]:
         return {
@@ -143,6 +154,9 @@ class MLPArtifactPayload(ArtifactPayload):
             "in_dim": self.in_dim,
             "hidden": self.hidden,
             "dropout": self.dropout,
+            "pathway": self.pathway,
+            "query_pathway": self.query_pathway,
+            "query_features": self.query_features,
         }
 
     @classmethod
@@ -153,6 +167,9 @@ class MLPArtifactPayload(ArtifactPayload):
             in_dim=state["in_dim"],
             hidden=state["hidden"],
             dropout=state["dropout"],
+            pathway=state.get("pathway", "irt"),
+            query_pathway=state.get("query_pathway"),
+            query_features=state.get("query_features"),
         )
 
 
```

`src/training/trainers/mlp_router_trainer.py`

```diff
--- a/src/training/trainers/mlp_router_trainer.py
+++ b/src/training/trainers/mlp_router_trainer.py
@@ -54,6 +54,9 @@ class MLPRouterTrainer(RouterModelTrainer):
             in_dim=int(model[0].in_features),
             hidden=int(model[0].out_features),
             dropout=float(model[2].p),
+            pathway=kw.get("pathway", "irt"),
+            query_pathway=kw.get("query_pathway"),
+            query_features=kw.get("query_features"),
         )
         store = LocalArtifactStore(self.runs_dir)
         store.save(RouterModelArtifact(
```
