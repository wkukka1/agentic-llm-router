---
name: aar-architecture
description: The agreed class structure for this repo, as a UML class diagram. Load before adding, renaming, moving or refactoring any class, module or package here, and before proposing a design - so the work lands on the existing types instead of inventing parallel ones. Also load when asked where something belongs, what a class is called, or whether a class already exists.
---

# The class structure is fixed. Work inside it.

`reference/uml.mmd` is the agreed design for this project: 69 classes, their
fields, their methods, and the arrows between them. It is the source of truth
for **what exists and what it is called**.

**Do not invent classes.** The failure this prevents is a plausible-looking
parallel hierarchy -- a `ModelSelector` beside `RouterModel`, a `Budget` beside
`BudgetLedger` -- which compiles, passes tests, and quietly forks the design.

Before writing a class, check the diagram:

```bash
grep -n "class <Name>" .claude/skills/aar-architecture/reference/uml.mmd
grep -rn "^class <Name>\b" src/          # is it built yet?
```

* **In the diagram and in `src/`** -> extend it where it lives. Do not re-declare it.
* **In the diagram, not in `src/`** -> build it at the path in the map below,
  with the fields and methods the diagram gives it. Nothing more.
* **Not in the diagram** -> stop. Say what is missing and what it would own, and
  get the diagram changed first. A class that is worth adding is worth adding
  to the design; one that is not worth adding to the design is not worth adding
  to the code.

## The two halves, and the rule between them

| | package | contains |
|---|---|---|
| **SERVING** (solid boxes) | `src/router/`, `src/decompose/` | everything that runs per request |
| **OFFLINE** (dashed boxes) | `src/training/`, `src/evaluation/` | data building, fitting, scoring |

**No serving class may point at an offline class.** `RouterModelArtifact` is the
only type both sides share: offline fits it, serving loads it. This is enforced
by `import-linter` contracts in `pyproject.toml` -- run `lint-imports` after
structural changes. `decompose` is stricter still: it imports neither.

Arrow meanings in the diagram: `-->` the source holds a field of that type;
`..>` a parameter, return value or id reference only. Copy that distinction into
the code -- a `..>` must not become a stored attribute.

## Where each class lives

Verified against the tree. Paths are under `src/`.

| UML class | file | status |
|---|---|---|
| `Router` (the pipeline) | `router/router.py` | built |
| `RoutingRequest`, `RoutingContext`, `Conversation`, `Message` | `router/context.py` | built |
| `RoutingConstraints`, `OptimizationObjective`, `RoutingMode` | `router/constraints.py` | built |
| `RoutingPolicy` | `router/policy.py` | built |
| `RoutingDecision`, `ModelScore`, `DecisionDestination` | `router/decision.py` | built |
| `PromptDecomposer`, `ClassificationInput` | `decompose/decomposer.py`, `decompose/classifiers/base.py` | built |
| `Classifier` + implementations | `decompose/classifiers/` | built |
| `Signal`, `PromptSignals` | `decompose/signals.py` | built |
| `Embedder`, `SentenceEmbedder`, `PromptEmbedder` | `decompose/embedders/` | built |
| `LLMProfile`, `LLMRegistry`, `LLMClient`, `ProviderAdapter` | `router/llm/` | built |
| `RouterModelArtifact`, `ArtifactFormat`, `ArtifactPayload` | `router/models/artifacts.py` | built |
| `Orchestrator` | `router/execution/orchestrator.py` | built |
| `BudgetLedger`, `ExecutionLimits` | `router/execution/budget.py` | built |
| `Plan`, `PlanStep`, `PlanStatus` | `router/execution/plan.py` | built |
| `PlanForecaster`, `CostForecast`, `PlanScore` | `router/execution/forecast.py` | built |
| `AgentTask`, `TaskStatus` | `router/execution/task.py` | built |
| `ExecutionResult` | `router/execution/results.py` | built (stub by design) |
| `ExecutionTrace` | `router/tracing/traces.py` | built |
| `Tool`, `MCPTool`, `RouterTool` | `router/tools/` | built |
| `RouterModel` (per-(query, model) scorer) | `router/routing/base.py` | built |
| `RouterModelFactory` | `router/models/registry.py` | **not built** |
| `ArtifactStore`, `LocalArtifactStore` | `router/models/artifacts.py` | **not built** |
| `CostModel` | `router/llm/cost.py` | **not built** |
| `Agent`, `AgentFactory`, `AgentConfig` | `router/execution/` | **not built** |
| `RouterModelTrainer` | `training/trainers/` | **not built** |
| `TrainingHarness`, `TrainingJob`, `TrainingRun` | `training/pipeline.py` | **not built** |
| `TrainingMetrics`, `ValidationMetrics` | `training/pipeline.py` | **not built** |
| `DatasetRef`, `DatasetSplit`, `TrainingConfig` | `training/datasets/` | **not built** |

"Not built" means the design has a place for it and nobody has written it yet.
Build it there, under that name.

## Deliberate deviations -- follow these, not the diagram

Two places where the repo is right and the diagram reads wrong. Do not
"fix" the code to match.

1. ~~**`Router` is two different things.**~~ **RESOLVED** by the
   `refactor-architecture` merge. The code now matches the diagram: the
   top-level pipeline is `router.router.Router`, and the per-`(query, model)`
   scorer it was colliding with has been renamed to
   `router.routing.base.RouterModel`. `RoutingPipeline` no longer exists; read
   `Router` as the diagram intends. The class map below has been updated.
2. **`decompose` is top-level, not `router/decompose/`.** It is a standalone
   package that imports nothing from `router`, `training` or `evaluation`
   (one exception: the embedders wrap `router.embeddings.encoder`, listed in
   the import-linter contract). Keep it dependency-free.

## Things already mapped onto these types

* The prompt-decomposition heads (domain, task, length, attributes, surface) are
  `Classifier` implementations in `decompose/classifiers/prompt_heads.py`, each
  emitting `Signal`s into `PromptSignals`. A new prompt signal is **a new
  `Classifier`**, not a new subsystem.
* NIRT is one `RouterModel` variant. The diagram says concrete rankers (KNN,
  IRT, NIRT, ML) are configuration, not design-level types -- they register with
  `RouterModelFactory` under a `routerModelName` and are not drawn.
* Anything that decides *whether* to escalate belongs in `RoutingPolicy`
  (`escalationThreshold`, `selectDestination`), not inside a `RouterModel`.
  `RouterModel` ranks; the policy decides.
* `OptimizationObjective.utility(...)` is the only definition of utility, and
  `lambda` is `costWeight`. Offline code imports it; it never reimplements the
  formula.
* Predicted cost lives in `CostModel`. Observed cost lives on `LLMResponse` and
  in the dataset. Never store observed cost on `CostModel`.

## Before you write

1. Found the class in `reference/uml.mmd`, or got the diagram amended.
2. Putting it at its mapped path, with the diagram's fields and methods.
3. Fields match `-->` arrows; `..>` stayed a parameter or return type.
4. No serving module imports `training` or `evaluation` -- `lint-imports` passes.
5. If the diagram itself needs to change, update `reference/uml.mmd` in the same
   change, and say so in the commit message.
