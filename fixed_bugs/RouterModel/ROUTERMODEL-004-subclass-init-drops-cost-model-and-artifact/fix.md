# ROUTERMODEL-004 fix

- **Bug:** [bug.md](bug.md)
- **Fixed:** 2026-09-23, uncommitted as of 2026-09-27
- **Diff base:** `263e04d` (last commit before the bug hunt) to the working tree on 2026-09-27

## What changed

Every concrete `RouterModel` subclass (`MatrixRouter`, `NIRTRouter`, `KNNRouter`, `MLPRouter`, `RandomRouter`) now takes keyword-only `cost_model=`/`artifact=` and forwards them to `super().__init__`. `NIRTRouter.from_artifact`/`MLPRouter.from_artifact` now pass both straight into the constructor instead of assigning them onto the instance after construction. Covered by `tests/router/routing/test_routing_interface.py::test_matrix_router_accepts_cost_model_and_artifact` / `test_random_router_accepts_cost_model_and_artifact` and `tests/router/routing/test_router_model_subclass_init.py`.

## Fix suggested when the bug was filed

Add `cost_model`/`artifact` keyword-only parameters to each subclass
`__init__` (or drop them to `**kwargs` forwarded to `super().__init__`), so
the base class's constructor surface is actually reachable. Fix alongside
ROUTERMODEL-002 — forwarding a parameter that's still never read doesn't fully
close the gap on its own.

## Regression tests

All of `tests/router/routing/test_router_model_subclass_init.py` (new module).

## Diff

Only the hunks this fix touched. A hunk can also carry a neighbouring fix in the same function; the bug IDs in the code comments tell them apart.

`src/router/routing/routers.py`

```diff
--- a/src/router/routing/routers.py
+++ b/src/router/routing/routers.py
@@ -149,8 +162,10 @@ class MatrixRouter(RouterModel):
     kind = "matrix"
 
     def __init__(self, scores: pd.DataFrame, *, name: Optional[str] = None,
-                 model_costs: Optional[Sequence[float]] = None):
-        super().__init__(list(scores.columns), name=name)
+                 model_costs: Optional[Sequence[float]] = None,
+                 cost_model: Optional["CostModel"] = None,
+                 artifact: Optional["RouterModelArtifact"] = None):
+        super().__init__(list(scores.columns), name=name, cost_model=cost_model, artifact=artifact)
         # the pool is stringified by RouterModel; the frame's labels must match or
         # every reindex misses (int-labelled frames would route on all-NaN)
         scores = scores.astype(np.float64)
@@ -191,8 +206,11 @@ class NIRTRouter(RouterModel):
         query_pathway: Optional[str] = None,
         query_features: Optional[str] = None,
         name: Optional[str] = None,
+        cost_model: Optional["CostModel"] = None,
+        artifact: Optional["RouterModelArtifact"] = None,
     ):
-        super().__init__(sorted(model_index, key=model_index.get), name=name)
+        super().__init__(sorted(model_index, key=model_index.get), name=name,
+                         cost_model=cost_model, artifact=artifact)
         self._model = model
         self._model_index = dict(model_index)
         self._data = load_training_data_for_router(data)
@@ -249,7 +267,7 @@ class NIRTRouter(RouterModel):
         model.load_state_dict(payload.state_dict)
         model.eval()
         dcfg = payload.config.get("data", {}) or {}
-        router = cls(
+        return cls(
             model,
             payload.model_index,
             data=load_training_data_for_router(data, dcfg.get("phase0_config")),
@@ -257,10 +275,9 @@ class NIRTRouter(RouterModel):
             query_pathway=dcfg.get("query_pathway") or None,
             query_features=dcfg.get("query_features") or None,
             name=name or f"nirt:{artifact.artifact_id}",
+            cost_model=cost_model,
+            artifact=artifact,
         )
-        router.cost_model = cost_model
-        router.artifact = artifact
-        return router
 
     @property
     def default_model_costs(self) -> Optional[np.ndarray]:
@@ -392,12 +409,15 @@ class KNNRouter(RouterModel):
         pathway: str = "retrieval",
         model_ids: Optional[Sequence[str]] = None,
         name: Optional[str] = None,
+        cost_model: Optional["CostModel"] = None,
+        artifact: Optional["RouterModelArtifact"] = None,
     ):
         d = load_training_data_for_router(data)
         if model_ids is None:
             obs = d.nirt_observations()
             model_ids = sorted(obs.loc[obs["split"] == "train", "model_id"].astype(str).unique())
-        super().__init__(model_ids, name=name or f"knn(k={k})")
+        super().__init__(model_ids, name=name or f"knn(k={k})",
+                         cost_model=cost_model, artifact=artifact)
         self._data = d
         self._k = int(k)
         self._pathway = pathway
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
@@ -517,8 +543,10 @@ class RandomRouter(RouterModel):
 
     kind = "random"
 
-    def __init__(self, model_ids: Sequence[str], *, seed: int = 0, name: Optional[str] = None):
-        super().__init__(model_ids, name=name or "random")
+    def __init__(self, model_ids: Sequence[str], *, seed: int = 0, name: Optional[str] = None,
+                 cost_model: Optional["CostModel"] = None,
+                 artifact: Optional["RouterModelArtifact"] = None):
+        super().__init__(model_ids, name=name or "random", cost_model=cost_model, artifact=artifact)
         self._seed = int(seed)
 
     def predict_scores(self, query_ids: Sequence[str]) -> pd.DataFrame:
```
