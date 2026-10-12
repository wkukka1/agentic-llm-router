# Known bugs

Top-down bug hunt: highest abstractions first, then their implementations. Started 2026-09-19.
This restarts the earlier review; findings from it are re-checked before they are filed here.

**This is the single source of truth for open bugs in `src/`.** Fixed bugs live in
[`fixed_bugs/`](../fixed_bugs/README.md). It absorbed
`code_review/`'s 52 delta findings on 2026-09-22: every finding was re-verified
against current source (not just trusted from commit messages), duplicates of
bugs already filed here were cross-linked instead of re-filed, and every
still-open finding with no existing entry was filed below. `code_review/`
itself is kept as the narrative record of that review (file:line evidence,
severity rationale) but is no longer where you check current bug status —
check here. See `code_review/README.md`'s banner for the full disposition of
each of the 52 findings. `investigations/` (e.g. `oracle_savings_gap/`) is a
separate thing: research into router/model behavior, not code defects — it
doesn't belong in this folder.

## Layout

```
known_bugs/
  <Class>/                     bugs in the class itself (one .md per bug)
    <Subclass>/                bugs specific to a subclass
  _module_level/<package>/     bugs in free functions or scripts (no class to file under)
```

Open bugs only. A fixed bug moves to the same path under `fixed_bugs/`, as a folder.

- File name: `<CLASSNAME>-NNN-short-slug.md`; the number is per class. Module-level bugs use `MOD-NNN`.
- A bug that shows up in several classes lives with its **root cause** and is cross-linked from the others.
- Only classes with something to report get a folder. "No bugs" is recorded in the coverage table below.

## Status vocabulary

| Status | Meaning |
|---|---|
| `reproduced` | Ran code and saw the failure. A repro script is in `known_bugs/_repros/` or is described in the file. |
| `confirmed` | Read the code and traced the failure through it. Not executed. |
| `unverified` | Reported by someone or something else; not yet checked. Kept in "Leads" below, not filed. |
| `fixed` | Fix landed. The bug moves to `fixed_bugs/<Class>/<ID-slug>/` as `bug.md` + `fix.md`. |

"Latent" means the bug is real but nothing in `src/` triggers it today.

## Bugs

| ID | Class | Sev | Status | One line |
|---|---|---|---|---|
| [OPTIMIZATIONOBJECTIVE-001](OptimizationObjective/OPTIMIZATIONOBJECTIVE-001-zero-weight-times-nonfinite-term.md) | OptimizationObjective | low | confirmed, latent | `0 * nan` poisons the utility |
| [MOD-001](_module_level/training.nirt.baseline/MOD-001-inspect-baseline-skips-relevance-dim-guard.md) | (script) | low | confirmed | `inspect_baseline.py` skips the relevance-dim guard |
| [KNNROUTER-001](RouterModel/KNNRouter/KNNROUTER-001-unknown-query-ids-crash.md) | KNNRouter | low-med | reproduced | Routing only unknown query ids crashes inside scikit-learn |
| [ROUTERMODELARTIFACT-001](RouterModelArtifact/ROUTERMODELARTIFACT-001-content-hash-never-computed.md) | RouterModelArtifact | low-med | confirmed | `content_hash` is required but never computed or verified |
| [TRAININGHARNESS-001](TrainingHarness/TRAININGHARNESS-001-created-at-not-persisted.md) | TrainingHarness | low | confirmed | `created_at` stamped in memory only |
| [TRAININGHARNESS-002](TrainingHarness/TRAININGHARNESS-002-metrics-from-mismatched-epochs.md) | TrainingHarness | low | confirmed | Train metrics from last epoch, validation from best epoch |
| [MLPROUTERTRAINER-002](RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-002-preprocessing-ignored.md) | MLPRouterTrainer | low | confirmed | `TrainingConfig.preprocessing` silently ignored |
| [AGENTICROUTER-003](AgenticRouter/AGENTICROUTER-003-max-calls-and-cost-exclude-helper-llm-calls.md) | AgenticRouter | med | fixed (partial) | `max_calls` and cost ignore triage / decompose / synthesize / orchestrator LLM calls |
| [NAIVEDECOMPOSER-002](NaiveDecomposer/NAIVEDECOMPOSER-002-then-comma-marker-never-matches.md) | NaiveDecomposer | low | reproduced | `\bthen,\b` never matches "then, " |
| [ROUTER-004](Router/ROUTER-004-build-context-candidates-overwritten.md) | Router | low | confirmed | `build_context`'s candidates are thrown away and rebuilt in no-registry mode |
| [RECURSIVEORCHESTRATOR-001](RecursiveOrchestrator/RECURSIVEORCHESTRATOR-001-children-solved-sequentially.md) | RecursiveOrchestrator | low | confirmed | Sub-tasks are solved sequentially, not concurrently |
| [PROMPTDECOMPOSER-001](PromptDecomposer/PROMPTDECOMPOSER-001-classifiers-no-shared-embeddings.md) | PromptDecomposer | low-med | confirmed | Each head classifier recomputes its own embedding pass |
| [MOD-001](_module_level/decompose/MOD-001-default-prompt-decomposer-not-wired.md) | (module) decompose | low | confirmed | `default_prompt_decomposer` is built but `Router` never calls it |
| [LENGTHCLASSIFIER-001](LengthClassifier/LENGTHCLASSIFIER-001-length-json-reparsed.md) | LengthClassifier | low | confirmed | `length.json` parsed twice (once by `LengthHead`, once by the classifier) |
| [TRAININGHARNESS-004](TrainingHarness/TRAININGHARNESS-004-job-run-id-ignored.md) | TrainingHarness | low-med | confirmed | `TrainingJob.run_id` ignored; harness uses `config.run_name` instead |
| [MLPROUTERTRAINER-004](RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-004-hardcoded-positional-indexing.md) | MLPRouterTrainer | low-med | confirmed | Architecture read back via hardcoded `nn.Sequential` positions |
| [MLPROUTERTRAINER-005](RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-005-double-round-trip-save-load.md) | MLPRouterTrainer | low | confirmed | `train()` saves then immediately reloads its own artifact |
| [ROUTERMODELTRAINER-001](RouterModelTrainer/ROUTERMODELTRAINER-001-divergent-constructor-signatures.md) | RouterModelTrainer | low | confirmed | Concrete trainers' constructors disagree on a `runs_dir` default |
| [NIRTTRAINER-001](RouterModelTrainer/NIRTTrainer/NIRTTRAINER-001-phase0-cfg-not-forwarded.md) | NIRTTrainer | low-med | confirmed | `phase0_cfg` unreachable through the wrapper; runs_dir logic duplicated by hand |
| [MOD-002](_module_level/evaluation.nirt.routing/MOD-002-cost-savings-vs-ref-divide-by-zero.md) | (module) evaluation.nirt.routing | low-med | confirmed | `cost_savings_vs_ref` divides by the reference cost with no zero guard |
| [LENGTHHEAD-001](LengthHead/LENGTHHEAD-001-negative-expected-tokens.md) | LengthHead (decompose) | low-med | reproduced | `expected_tokens = expm1(ridge)` goes negative for empty / one-word prompts |
| [FROZENENCODERMODEL-001](FrozenEncoderModel/FROZENENCODERMODEL-001-prefix-and-normalize-ignored.md) | `_FrozenEncoderModel` | low-med | reproduced | `prefix` / `normalize` params are accepted but never reach `EmbeddingEncoder` |
| [PROMPTDECOMPOSER-002](PromptDecomposer/PROMPTDECOMPOSER-002-duplicate-classifier-fails-on-request-path.md) | PromptDecomposer | low | reproduced | Duplicate classifier accepted by `add_classifier`, then every `extract_signals` raises |
| [MOD-001](_module_level/decompose.prompt_decomposition.signals/MOD-001-surface-regex-false-positives.md) | (module) decompose.prompt_decomposition.signals | low-med | reproduced | `has_code_shape` fires on "from"/"class"; `has_math` on `$5`, dates, ranges (retrain caveat) |
| [MOD-001](_module_level/decompose.prompt_decomposition.settings/MOD-001-dotenv-inline-comment-crashes-import.md) | (module) decompose.prompt_decomposition.settings | low | reproduced | `.env` inline comment makes `settings()` raise, which breaks importing the encoder |
| [MOD-002](_module_level/decompose/MOD-002-default-factory-cannot-set-defer-below.md) | (module) decompose | low | confirmed | `default_prompt_decomposer` cannot set `defer_below`, so `should_defer` is always False |
| [MOD-001](_module_level/training.nirt.pairwise/MOD-001-auxiliary-battles-include-cold-start-models.md) | (module) training.nirt.pairwise | med | reproduced | Auxiliary Arena loss trains on battles with cold-start models (4,617 battles; `models="all"`) |
| [MOD-001](_module_level/router.embeddings/MOD-001-build-store-resume-ignores-vector-fingerprint.md) | (module) router.embeddings | low-med | reproduced | `build_store` won't resume a partial build after a device / batch-size change, contrary to `vector_fingerprint`'s contract |
| [MOD-001](_module_level/training.data.pricing/MOD-001-missing-tokens-priced-at-zero.md) | (module) training.data.pricing | low-med | reproduced, latent | `fill_costs` prices rows with missing token counts at $0 instead of leaving NaN |
| [QUERYBANK-001](QueryBank/QUERYBANK-001-similarity-weighting-inverted-on-flat-l2.md) | QueryBank | low | reproduced, latent | `weighting="similarity"` on a `flat_l2` bank weights by distance (far neighbour wins) |
| [JUDGECLIENT-001](JudgeClient/JUDGECLIENT-001-max-retries-zero-makes-no-attempt.md) | JudgeClient | low | reproduced | `max_retries=0` makes zero attempts and raises |
| [MOD-001](_module_level/training.data.chance_correction/MOD-001-n-choices-below-two-gives-nan-but-flagged-corrected.md) | (module) training.data.chance_correction | low | reproduced, latent | `n_choices <= 1` gives NaN `corrected_score` yet `chance_corrected=True` |
| [MOD-001](_module_level/evaluation.data.stratify/MOD-001-unanimous-graded-rows-counted-discriminative.md) | (module) evaluation.data.stratify | low | reproduced | A unanimous graded row (all 0.5) lands in the "discriminative" stratum |
| [MOD-001](_module_level/evaluation.prompt_decomposition.length_audit/MOD-001-target-ceiling-is-a-reliability-not-a-correlation-ceiling.md) | (module) evaluation.prompt_decomposition.length_audit | low-med | confirmed (analytic) | `target_ceiling` (0.742) is a reliability, but is compared to Spearman rho; the correlation ceiling is sqrt (0.861), so the head is at ~67%, not ~77% |
| [MOD-001](_module_level/training.prompt_decomposition.experiment/MOD-001-run-all-records-no-failures-when-nothing-succeeds.md) | (module) training.prompt_decomposition.experiment | low | reproduced | `run_all` writes no `failures.json` / leaderboard when no experiment succeeds |
| [MOD-001](_module_level/training.prompt_decomposition.data.sources/MOD-001-missing-synthetic-file-raises-instead-of-warning.md) | (module) training.prompt_decomposition.data.sources | low | reproduced | `load_synthetic_tasks` raises `FileNotFoundError`; its "no file" warning branch is unreachable |

Fixed bugs are moved out of this folder into [`fixed_bugs/`](../fixed_bugs/README.md), one folder per bug with `bug.md` (the report) and `fix.md` (what changed, the regression tests and the diff).

Filed 2026-09-22, from the `code_review/` consolidation (all traced against current source, not just commit titles). Findings from that review confirmed as duplicates of bugs above rather than re-filed: RR-05 (→ [ROUTERMODEL-002](../fixed_bugs/RouterModel/ROUTERMODEL-002-cost-model-never-consumed/bug.md)), RM-01 (→ [LOCALARTIFACTSTORE-002](../fixed_bugs/ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-002-save-clobbers-fit-run-json/bug.md)/[-005](../fixed_bugs/ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-005-save-order-can-pair-new-weights-with-old-metadata/bug.md), remediated by the atomic-write fix — the narrower cross-file race LOCALARTIFACTSTORE-005 describes is still open), RM-02 (→ [ROUTERMODELFACTORY-001](../fixed_bugs/RouterModelFactory/ROUTERMODELFACTORY-001-cost-model-kwarg-overwritten/bug.md), fixed), RM-04 (→ [LOCALARTIFACTSTORE-004](../fixed_bugs/ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-004-load-reads-model-pt-twice/bug.md), still open), RM-07 (→ [ROUTERMODELARTIFACT-001](RouterModelArtifact/ROUTERMODELARTIFACT-001-content-hash-never-computed.md), still open), RM-03/TN-05 (→ [TRAININGHARNESS-001](TrainingHarness/TRAININGHARNESS-001-created-at-not-persisted.md), `created_at` now fixed at the harness level; `training_dataset_version`/`content_hash` remain open per ROUTERMODELARTIFACT-001), TP-01 (→ [TRAININGHARNESS-003](../fixed_bugs/TrainingHarness/TRAININGHARNESS-003-unpopulated-metrics-read-as-perfect/bug.md), train-side loss now populated, other train metrics still 0.0 by documented design), TT-01 (→ [MLPROUTER-001](../fixed_bugs/RouterModel/MLPRouter/MLPROUTER-001-from-artifact-loses-pathway-and-data-config/bug.md), `from_artifact` now exists so the *dead-end* TT-01 described is fixed, but the more specific MLPROUTER-001 defect in that same method is still open), and all four AGENTICROUTER-001..004 (already filed here; re-confirmed still open as of the 2026-09-22 consolidation, then fixed 2026-09-23 — see each file — the `router.agentic.md` review's "no previously-known bugs recurred" claim was about an older, unrelated bug set, not these).

## Coverage (top-down)

Layer 0 is the entry point. Each lower layer is what the one above holds.

| Layer | Class | Where | State |
|---|---|---|---|
| 0 | `Router` | `router/router.py` | done: 4 bugs |
| 1 | `RoutingRequest`, `RoutingContext`, `RoutingConstraints` | `router/context.py`, `constraints.py` | done: no bugs |
| 1 | `OptimizationObjective` | `router/constraints.py` | done: 1 latent |
| 1 | `RoutingDecision`, `ModelScore` | `router/decision.py` | done: no bugs (data classes) |
| 1 | `RoutingPolicy` / `DefaultRoutingPolicy` | `router/policy.py` | done: no bugs |
| 1 | `RouterModel` (base) | `router/routing/base.py` | done: 4 bugs |
| 1 | `LLMRegistry`, `LLMProfile` | `router/llm/` | done: no bugs |
| 1 | `PromptDecomposer`, `Classifier`, `PromptSignals` | `decompose/` | done: 4 bugs (2 `PromptDecomposer`, 2 `default_prompt_decomposer`); the five prompt-head classifiers: 2 bugs (`DomainClassifier`, `LengthClassifier`) |
| 2 | `MatrixRouter`, `RandomRouter` | `router/routing/routers.py` | done: no bugs |
| 2 | `NIRTRouter`, `KNNRouter`, `MLPRouter` | `router/routing/routers.py` | done: 2 bugs (`MLPRouter`, `KNNRouter`); `NIRTRouter` clean |
| 3 | `NIRTModel`, `IRTRouterModel`, `_MLPHead` | `router/nirt/model.py` | done: no bugs |
| 3 | `WarmupBlender`, `DiscriminationHead`, `DifficultyHead`, `InteractionLayer`, `LengthHead` | `router/nirt/components.py` | done: no bugs |
| 3 | scoring and decision helpers | `router/nirt/predict.py`, `routing_decision.py`, `baselines_infer.py` | done: 1 bug (kNN, filed under `KNNRouter`) |
| 3 | `shrinkage.py` (`novelty_weight`, `shrink_predictions`) | `router/nirt/shrinkage.py` | done: no bugs |
| 3 | `frames.py`, `data/nirt.py` (`NIRTDataset`) | `router/` | done 2026-09-24: no bugs |
| 3 | `TextEncoder`, `EncoderConfig`, `EmbeddingStore`, `build_store` | `router/embeddings/encoder.py` | done 2026-09-24: 2 bugs (`TextEncoder` reseed; `build_store` resume); `EmbeddingStore` clean apart from leads below |
| 2 | training data layer: `splits`, `response_matrix`, `chance_correction`, `quality`, `normalize`, `families`, `model_registry`, `pricing`, `query_features`, `benchmark_id`, `embeddings`, `nirt`, `facade` (`TrainingData`), `schemas`, `loaders` | `training/data/` | done 2026-09-24: 3 bugs (`chance_correction`, `pricing`, `loaders` IRT-Router zero/zero costs); `splits` / `facade` / `response_matrix` clean; the RouterBench / Arena / lm-harness / anchor-judge loaders read, clean. LLMRouterBench loader read but **not exercised** (no raw results on disk) |
| 2 | taxonomy (`ClusterConfig`, `cluster_embeddings`, `relevance`, `taxonomy`) | `training/taxonomy/` | done 2026-09-24: 1 bug (clusters fit on all splits) |
| 2 | retrieval (`QueryBank`, `knn_impute`, `warmup`) | `training/retrieval/` | done 2026-09-24: 1 latent bug (`QueryBank` L2 weighting); self-exclusion by content group traced and correct |
| 2 | auxiliary pairwise loss, `fit()`, `fit_probe`, `metrics` | `training/nirt/pairwise.py`, `train.py`, `coldstart.py`, `metrics.py` | done 2026-09-24: 1 bug (pairwise cold-start leak); `fit()` early stopping / schedulers / grouped losses / eval-mode toggling traced, clean |
| 2 | `training/models/profiles.py` (profile text, `empirical_stats`) | `training/models/` | done 2026-09-24: no bugs (`include_empirical` defaults off, as documented) |
| 2 | `JudgeClient`, judge prompts | `evaluation/judge/` | done 2026-09-24: 1 bug (`max_retries=0`) |
| 2 | classical IRT baseline, `stratify`, `judge_responses`, `ood`, `coldstart_eval` | `evaluation/baselines`, `evaluation/data`, `evaluation/nirt/{ood,coldstart_eval}.py` | done 2026-09-24: 1 bug (`stratify`); `classical_irt`, `judge_responses`, `ood`, LOMO clean (one latent lead below) |
| 2 | pool-expansion battery | `evaluation/pool_expansion/battery.py` | done 2026-09-24: no new bugs (accuracy definitions consistent with `pareto`; the zero-reference-cost divide is already MOD-002 of `evaluation.nirt.routing`) |
| 3 | `Embedder`, `BertEmbedder`, `SentenceEmbedder`, `Classifier` / `CompositeClassifier`, `Conversation`, `Signal` / `PromptSignals` | `decompose/` | done 2026-09-24: no bugs (`SentenceEmbedder.embed_batch` has no `Embedder` / `BertEmbedder` counterpart, a wart, not a bug: no caller) |
| 3 | `Config`, `seed_everything`, `git_sha` / `file_digest`, `ExecutionTrace` | `router/config.py`, `determinism.py`, `provenance.py`, `tracing/` | done: no bugs |
| 1 | `RouterModelFactory` | `router/models/registry.py` | done: 1 bug |
| 1 | `RouterModelArtifact`, `ArtifactPayload` (NIRT / Baseline / MLP) | `router/models/artifacts.py` | done: 1 bug |
| 2 | `ArtifactStore` / `LocalArtifactStore` | `router/models/store.py` | done: 6 bugs |
| 1 | `RouterModelTrainer`, `TrainingConfig` | `training/trainers/base.py` | done: 1 bug (divergent constructor signatures) |
| 2 | `NIRTTrainer` | `training/nirt/trainer.py` | done: 1 bug |
| 2 | `BaselineNIRTTrainer` | `training/nirt/baseline/trainer.py` | done: no bugs (earlier lead refuted) |
| 2 | `MLPRouterTrainer` | `training/trainers/mlp_router_trainer.py` | done: 5 bugs |
| 1 | `TrainingHarness`, `TrainingMetrics`, `ValidationMetrics` | `training/pipeline.py` | done: 4 bugs |
| 3 | `BaselineNIRT` | `training/nirt/baseline/model.py` | done: no bugs |
| 3 | `ResponseHead`, `BernoulliResponseHead`, `NormalResponseHead`, `BetaResponseHead`, `ZOIBResponseHead` | `training/nirt/baseline/` | done: no bugs |
| 3 | `fit()` (NIRT, baseline), `fit_mlp_router` | `training/nirt/train.py`, `baseline/train.py`, `trainers/mlp_router.py` | 1 module-level bug (`training.nirt` runs_dir resolution); otherwise read for the trainer/harness questions only, not fully audited |
| 1 | `LLMClient`, `LLMClientFactory`, `CostModel`, `LLMResponse` | `router/llm/client.py`, `cost.py` | done: 3 bugs, all fixed 2026-09-23; `CostModel` clean |
| 2 | `ProviderAdapter`: `LangChainAdapter`, `EchoAdapter`, `CallableAdapter`, OpenAI / Anthropic / Google stubs | `router/llm/adapters/` | done: 3 bugs (2 `LangChainAdapter`, 1 module-level `guess_provider`); the three SDK adapters are stubs that raise `NotImplementedError` by design |
| 1 | `AgenticRouter`, `AgenticResult`, `SubCall` | `router/agentic/router.py`, `result.py` | done: 4 bugs |
| 2 | `ClientRegistry`, `HeuristicTriage`, `LLMTriage`, `NaiveDecomposer`, `LLMDecomposer`, `LLMSynthesizer`, `RecursiveOrchestrator`, `LangChainToolOrchestrator` | `router/agentic/` | done: 6 bugs; triage and synthesizers clean, `RecursiveOrchestrator` has 1 (sequential, not a correctness bug) |
| 1 | `Orchestrator`, `OrchestratorFactory`, `BudgetLedger`, `PlanForecaster`, `Plan`, `AgentTask`, `ExecutionResult` | `router/execution/` | done: 3 bugs, all latent (nothing constructs these yet) |
| 1 | `Tool`, `RouterTool`, `MCPTool` | `router/tools/` | done: no bugs (scaffolding that raises `NotImplementedError` as documented) |
| 2+ | the five prompt-head classifiers, `DomainClassifier` models | `decompose/` | `prompt_heads.py` classifiers done: 2 bugs (`DomainClassifier`, `LengthClassifier`). `prompt_decomposition/` read end to end 2026-09-24: 6 bugs (`LengthHead`, `_FrozenEncoderModel`, `PromptDecomposer`, surface regexes, `settings`, factory `defer_below`); serving heads, calibration, composite vector, sklearn/ensemble models, encoder and `cascade/signals.py` otherwise clean. Training and evaluation code under `prompt_decomposition/` only read for train/serve consistency |
| 2+ | evaluation (`oracle`, `pool_expansion`, `judge`) | `evaluation/` | `evaluation.nirt.routing` and `evaluation.routing.oracle`: 3 module-level bugs (from the code_review consolidation); `pool_expansion` and `judge` now done, see above. **Not started:** `evaluation/nirt/evaluate.py`, `training/nirt/baseline/{data,eval,continuous_eval,diagnostics,calibration}.py`, `training/cli.py`, and everything under `scripts/` (bar `prompt_decomposition_cli.py`'s imports) |
| 2 | prompt-decomposition training: dataset builders / splits, `sources`, arena + cascade corpora, `arena_labels`, `ExperimentConfig`, `run_experiment` / `run_all`, `metrics`, the length / attribute / forest heads | `training/prompt_decomposition/` | done 2026-09-25: 3 bugs (dataset builders **high**, `run_all`, `load_synthetic_tasks`); `assert_no_leakage`, stratified `split_frame`, temperature-on-val calibration, per-source scoring, `metrics.evaluate` traced clean. The arena / cascade corpus builders **could not be run** (only `arena-human-preference-55k` is cached, they read the 140k dump) |
| 2 | prompt-decomposition evaluation: `analysis`, `external_eval`, `forest_report`, `length_audit`, `overfit`, `routing_value` | `evaluation/prompt_decomposition/` | done 2026-09-25: 1 analytic bug (`target_ceiling` scale); permutation / learning-curve / near-duplicate checks, forest ablations, `top_up_metrics` (no test-fitted temperature written back) clean |

## Leads: reported earlier, not yet verified

Filed under the class where they belong once that class is reached in the top-down pass.

- `router/nirt/routing_decision.py:regret`: NaN in `true` (a selected model with no observation) propagates to a NaN regret rather than being excluded. Looks like an evaluation-layer question; check when `oracle.py` is reached.

Found 2026-09-24 while going through the "not started" layers; each is real in the code but was not reproduced, is ambiguous by design, or has no consumer today:

- `training/retrieval/warmup.py:build_warmup_representations` discards `neighbor_weighted_mean`'s `found` count (`[0]`), so a query with no usable neighbour gets an all-zero warm-up vector, whereas `knn_impute` falls back to the raw embedding for the same case. Needs a bank smaller than `k` plus the excluded group to trigger.
- `evaluation/nirt/coldstart_eval.py:lomo_eval` draws probe rows from `obs` by row, not by query. If a model has more than one row per `query_id` (e.g. lm-harness `acc` + `acc_norm`), a probe row's twin lands in the eval set. No current build has multi-metric duplicates for one (query, model), so latent.
- `training/data/normalize.py:render_prompt` unpacks any prompt that is a `str(list[str])` literal into blank-line-joined turns, so a genuine prompt such as `"['b', 'a', 'c']"` becomes `"b\n\na\n\nc"` (reproduced). Deliberate for RouterBench's stored format; a heuristic false positive, not filed.
- `router/embeddings/encoder.py:EmbeddingStore.__init__` builds `_index` from a dict comprehension, so duplicate ids collapse silently (last wins) rather than raising, and `subset()` copies the parent's `ids_fingerprint` / `texts_fingerprint` into the child manifest.
- `training/data/response_matrix.py:collapse_duplicate_observations` groups with the default `dropna=True`, so a duplicate whose `query_id` / `model_id` / `metric_type` is null is dropped, not averaged. The schema forbids null keys, so latent.
- `training/data/splits.py:make_presplit` groups by `query_id`, not content hash, so two ids with identical text but different `dataset` labels could straddle train and test1/test2. **Checked on the built IRT-Router data: 0 of 38,031 content-hash groups straddle**, so no live leak; `check_leakage` only compares ids and would not catch it.

Found 2026-09-25 in prompt decomposition; not filed (untestable here, or a wart):

- `training/prompt_decomposition/data/cascade.py:_split_of` hashes `prompt` **unstripped**, while `arena_corpus.build_arena_corpus` strips before hashing (`_split_of` is otherwise identical: sha1, first 8 hex, 0.70 / 0.85 cut). The same prompt with trailing whitespace can land in different splits in the two corpora. Only matters if a head trained on one corpus is scored on the other's test rows; nothing does that today. Could not run: the 140k arena dump is not cached.
- `cascade.model_tiers` derives weak / strong tiers from win rates over **all** single-turn battles, including the ones later assigned to the test split, so the escalate label's tier definition has seen test outcomes. Model-level, so the effect should be small.
- Stale docstring: `sources.load_agent_tasks` says the agent labels are "On by default"; `build_task_dataset(include_agent=False)` and its own docstring say off, on the evidence.
- `dataset.build_real_only_dataset` takes the frozen eval rows' labels from the current hand-labelled pool (matched by prompt), not from `eval_frozen.parquet`, and does not assert every frozen row was matched. Checked today: all 400 matched, 0 label disagreements, so latent; a relabelling of the pool would silently change the "frozen" test.

**Resolved (not a bug filing needed):** `evaluation/routing/oracle.py:511`'s "hard oracle is a one-hot prediction, so `selected_predicted_score` is always 1.0" lead was the mechanism the ER-01 fix (2026-09-22 `code_review` consolidation) intentionally uses — the canonical `oracle_choice` + a one-hot matrix replaced the old hand-rolled perturbation. Checked against current source: the one-hot construction is now the documented, intended design, not a bug.

## Refuted: suspected, checked, not bugs

Kept so they are not re-investigated.

- **Ensemble temperature applied twice at serving** (`EnsembleClassifier.predict_proba` and `CalibratedHead._calibrated`). `training/prompt_decomposition/experiment.py` fits the head temperature on `model.predict_proba` output, which already includes the ensemble's own temperature, so serving composes them exactly as training did.
- **Length / attribute heads served a different encoder config than they were trained on.** Training and serving both use `EmbeddingEncoder(encoder_model, max_length=256)` with default pooling.

- **`BaselineNIRTTrainer` `profile_dim = ... or query_dim`** (a lead from the earlier review). `fit()` builds the model with exactly the same fallback ([baseline/train.py:131](../src/training/nirt/baseline/train.py#L131), [234](../src/training/nirt/baseline/train.py#L234)), so the payload records the dimension the model really has.
- **`ResponseHead.quantiles` hitting `torch.quantile`'s size limit.** `batched_forward` chunks at 16384 rows, and this torch build handled 40000 rows (`512 x 40000` samples) without error for the Beta and ZOIB heads.
- **`mlp_router_matrix` never calling `.eval()` (dropout stays on).** `fit_mlp_router` returns the model in eval mode ([mlp_router.py:125](../src/training/trainers/mlp_router.py#L125)) and `MLPRouter.from_artifact` calls `.eval()`.
- **`InteractionLayer` crashing when `relevance_dim > 0` but `r_q` is absent.** `BaselineNIRT` passes `relevance_dim` only when `use_relevance` is on, and fills a uniform `r_q` when it is missing.
- **Pool-expansion battery mixing a binarised accuracy with the frontier's accuracy.** `pool_description`, `derived_headline` and `_validation_frontier` all use `(true >= 0.5).mean()`, and `pareto` / `_policy_row` define `accuracy` the same way, so the `-Npt` thresholds compare like with like.
- **`fit()` early-stopping / model-selection defects** (best-state restore, `ReduceLROnPlateau` fed the negated higher-is-better metric, `_next_battles` closing over `gen` before it is assigned, `_predict_packed` leaving the model in eval mode before the collapse probe). Each traced; all are correct.
- **`QueryBank.search` group exclusion under-fetching.** It over-fetches `k + max_group_size`, which bounds the number of bank rows one excluded group can occupy, so `k` clean neighbours are always available when the bank is large enough.
- **`TrainingData.correctness_matrix` pooling metrics silently.** `combine_metrics` is opt-in and documented; the default is one metric.
- **`TrainingData.correctness` failing when `corrected_score` is absent** (`df.get(...)` returns `None`). `coerce_response_frame` and `apply_chance_correction` always create it.
- **`EmbeddingEncoder.encode_cached` serving stale vectors** (the forest experiment re-samples rows under one `tag`). The cache key is `signature + tag + sha1(all texts)`, so a different row set is a different file.
- **`external_eval.score`'s "top-2" being wider than 2.** `shortlist` is a fixed 2 unless the head is built with `shortlist_mass`, which the default is not.
- **Frozen eval vs pool mismatch in `build_real_only_dataset`** as a live defect: measured, none (see the lead above).
- **`AuditResult.permutation_p` / `passes_permutation` disagreeing, and `overfit.audit`'s near-duplicate check comparing a subsample to the full-set accuracy.** They gate on different quantities on purpose; the subsample only applies above 12,000 rows and the hand-labelled sets are 2,441.
- **`TrainingConfig.seed` not reaching `fit()`.** Both `fit()` functions read the top-level `seed` that `translate_training_config` sets.

## Repros

Run from the repo root with the project venv, e.g. `.venv\Scripts\python.exe known_bugs\_repros\repro_router.py`. They live in [_repros/](_repros).

