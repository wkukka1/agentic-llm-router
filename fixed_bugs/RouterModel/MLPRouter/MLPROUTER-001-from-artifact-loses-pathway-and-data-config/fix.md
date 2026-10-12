# MLPROUTER-001 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

`MLPRouter.__init__` now accepts `query_pathway=`/`query_features=` (mirroring `NIRTRouter`'s constructor shape), `predict_scores` forwards both to `mlp_router_matrix` (which already supported them), and `from_artifact` reads `payload.pathway`/`.query_pathway`/ `.query_features` (persisted by the [MLPROUTERTRAINER-001](../../../RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-001-hyperparameters-not-persisted/bug.md) fix, landed first in the same pass) instead of leaving the router on the constructor's `pathway="irt"` default. Covered by `tests/training/trainers/test_mlp_router_trainer.py::test_from_artifact_restores_the_trained_pathway`.

## Fix suggested when the bug was filed

Persist `pathway`/`query_pathway`/`query_features`/phase0 config in the artifact (see the
trainer bug), read them in `from_artifact`, and give `MLPRouter` matching parameters.

## Regression tests

`tests/training/trainers/test_mlp_router_trainer.py:69` `test_from_artifact_restores_the_trained_pathway`

```python
def test_from_artifact_restores_the_trained_pathway(tmp_path):
    """MLPROUTER-001: from_artifact must not silently fall back to the
    constructor's pathway="irt" default when the artifact recorded a
    different one."""
    d = FakeTrainingData(make_split_obs(seed=16), query_dim=16)
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="mlp-run-pathway-2",
        hyperparameters={"hidden": 16, "epochs": 2, "pathway": "bert",
                         "query_pathway": "bert-query"},
    )
    artifact = trainer.train(d, config)
    assert artifact.payload.pathway == "bert"

    router = MLPRouter.from_artifact(artifact, data=d)
    assert router._pathway == "bert"
    assert router._query_pathway == "bert-query"
    assert router._query_features is None
```

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/routing/routers.py`

```diff
--- a/src/router/routing/routers.py
+++ b/src/router/routing/routers.py
@@ -457,11 +477,16 @@ class MLPRouter(RouterModel):
     kind = "mlp"
 
     def __init__(self, model, model_ids: Sequence[str], *, data=None,
-                 pathway: str = "irt", name: Optional[str] = None):
-        super().__init__(model_ids, name=name or "mlp")
+                 pathway: str = "irt", query_pathway: Optional[str] = None,
+                 query_features: Optional[str] = None, name: Optional[str] = None,
+                 cost_model: Optional["CostModel"] = None,
+                 artifact: Optional["RouterModelArtifact"] = None):
+        super().__init__(model_ids, name=name or "mlp", cost_model=cost_model, artifact=artifact)
         self._model = model
         self._data = load_training_data_for_router(data)
         self._pathway = pathway
+        self._query_pathway = query_pathway or pathway
+        self._query_features = query_features
         self._cost_cache = _UNSET
 
     @classmethod
@@ -484,12 +509,12 @@ class MLPRouter(RouterModel):
         model = build_mlp_router(payload.in_dim, len(payload.model_ids), payload.hidden, payload.dropout)
         model.load_state_dict(payload.state_dict)
         model.eval()
-        router = cls(
+        return cls(
             model, payload.model_ids, data=data, name=name or f"mlp:{artifact.artifact_id}",
+            pathway=payload.pathway, query_pathway=payload.query_pathway,
+            query_features=payload.query_features,
+            cost_model=cost_model, artifact=artifact,
         )
-        router.cost_model = cost_model
-        router.artifact = artifact
-        return router
 
     @property
     def default_model_costs(self) -> Optional[np.ndarray]:
@@ -501,6 +526,7 @@ class MLPRouter(RouterModel):
         return mlp_router_matrix(
             self._model, self._model_ids, self._data,
             [str(q) for q in query_ids], pathway=self._pathway,
+            query_pathway=self._query_pathway, query_features=self._query_features,
         )
 
 
```
