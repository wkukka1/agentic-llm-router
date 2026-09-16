# The `Router` interface (`router.routing`)

One contract for every routing strategy, so NIRT, k-NN, an MLP head, a bandit, or
an LLM-judge cascade are interchangeable at the call site and a new strategy is
just *"how good is model `m` for query `q`"*.

```
predict_scores(query_ids)  ->  [query_id × model_id]  E[Y | q, m]
        |
        +--- route(lam, model_costs, eligible)  ->  RoutingResult   (router, no labels)
        |
        +--- evaluate_router(true_df, cost_df)  ->  oracle metrics  (evaluation, needs labels)
```

`router.routing` is the **decision layer** — no ground truth involved. It reuses
the selection policy in `router.nirt.routing_decision` (`routing_decision`).
Oracle-relative scoring needs labels, so it isn't a method on `Router`: it's
`evaluation.routing.oracle.evaluate_router`, which delegates to
`routing_evaluation`. This is the boundary the package split enforces — a
serving router must be usable with no `evaluation` (or `training`) import at
all.

## The contract

```python
from router.routing import Router, register

@register
class MyRouter(Router):
    kind = "my_router"                      # distinct, stable; how the registry finds it

    def predict_scores(self, query_ids):
        # -> DataFrame indexed by query_id, columns ⊇ self.model_ids,
        #    values = predicted quality E[Y | q, m]
        ...
```

Everything else is provided by the base class:

| method | what it does |
|---|---|
| `route(query_ids, *, lam=0, model_costs=None, eligible=None)` | pick one model per query; returns `RoutingResult`. `lam>0` ⇒ cost-aware utility `pred − lam·C(m)`. `__call__` is an alias. |
| `model_ids` | the candidate pool, in `predict_scores` column order |
| `default_model_costs` | per-model cost used when `route` gets no explicit `model_costs` (data-backed routers fill this from the train-split mean cost) |

`RoutingResult` carries `.selected_model_ids`, `.selected_scores`, `.model_mix()`,
`.to_frame()` (one row per query), `.score_frame()` (the full `[Q×M]` matrix).

Oracle-relative scoring is a free function, not a method:

| function (`evaluation.routing.oracle`) | what it does |
|---|---|
| `evaluate_router(router, true_df, cost_df=None, *, lam=0, tolerance=0.01)` | oracle hit rate, regret quantiles, selected-vs-oracle quality/cost, cost-aware oracle block (delegates to `routing_evaluation`) |
| `compare_routers(routers, true_df, cost_df=None, *, lam=0)` | scores several routers side by side, adding the hard-oracle upper bound and a random floor |

`query_ids` are ids into the project's prebuilt query set / embedding stores — the
same currency the rest of the stack uses. Routing on raw text is a future
extension (an encoder in front of `predict_scores`); the contract is unchanged.

## Built-in strategies (`router.routing.routers`)

| `kind` | class | backed by |
|---|---|---|
| `matrix` | `MatrixRouter(scores_df)` | a precomputed `[query × model]` frame (ZOIB `E[Y]`, the Bernoulli baseline, a hand-made table) |
| `nirt` | `NIRTRouter.from_run("nirt-2d-projected")` | a trained NIRT / IRT-Router run (`router.nirt.checkpoint.load_run` + `router.nirt.predict.predict_dataset`) |
| `knn` | `KNNRouter(data, k=5)` | RouterBench-style nearest-neighbour (`router.nirt.baselines_infer.knn_router_matrix`) |
| `mlp` | `training.trainers.mlp_router.fit_mlp_router_as_router(data)` | the IRT-free `e_q → R^M` head (fits with `fit_mlp_router`, wraps the result as an `MLPRouter`) |
| `random` | `RandomRouter(model_ids)` | uniform pick — an evaluation floor |

Each is a thin adapter over machinery that already exists; the value is the shared
interface. `MLPRouter` has no `.train()` of its own — fitting is a `training`
concern, so it's a function in `training.trainers.mlp_router` that returns an
`MLPRouter`, not a classmethod on the serving class.

## Usage

```python
from router.config import load_config
from router.routing import NIRTRouter, KNNRouter, build_router
from training.data.facade import load_training_data
from evaluation.nirt.routing import eval_matrices
from evaluation.routing.oracle import compare_routers, evaluate_router

d = load_training_data(load_config())
true_df, cost_df = eval_matrices(d, split="test")
qids = list(true_df.index)

nirt = NIRTRouter.from_run("nirt-2d-projected", data=d)
knn  = build_router("knn", data=d, k=5)                 # or KNNRouter(d, k=5)

nirt.route(qids, lam=0.3).to_frame()                    # the decision, per query -- router only
evaluate_router(nirt, true_df, cost_df, lam=0.3)        # oracle-relative metrics -- evaluation only
summary, detail = compare_routers([nirt, knn], true_df, cost_df, lam=0.3)
```

`compare_routers` adds the hard-oracle upper bound and a random floor and returns
the same `(summary_df, detail)` shape as
`evaluation.routing.oracle.compare_routing_strategies`.
