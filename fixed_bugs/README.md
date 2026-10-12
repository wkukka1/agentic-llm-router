# Fixed bugs

Bugs from the [`known_bugs/`](../known_bugs/README.md) hunt (started 2026-09-19) whose fix has landed in `src/`.
Each bug has its own folder, at the same `<Class>/` or `_module_level/<package>/` path it had under `known_bugs/`:

```
fixed_bugs/
  <Class>/<ID-slug>/
    bug.md    the report as filed: what, failure scenario, evidence
    fix.md    what changed, the fix suggested at filing time, the regression tests (code), and the diff
```

The diffs in `fix.md` run from `263e04d` (the last commit before the hunt) to the working tree as of
2026-09-27, filtered to the hunks each fix touched. Several fixes were uncommitted when they were written up,
so a diff is the reliable record rather than a commit hash. Where two fixes touch the same function the hunk
appears in both; the bug IDs in the code comments tell them apart.

Still open elsewhere:

- [AGENTICROUTER-003](../known_bugs/AgenticRouter/AGENTICROUTER-003-max-calls-and-cost-exclude-helper-llm-calls.md) is only partly fixed, so it stays in `known_bugs/`.
- [training.taxonomy MOD-001](_module_level/training.taxonomy/MOD-001-clusters-fit-on-all-splits-including-ood/fix.md) is fixed in code, but the on-disk taxonomy has not been rebuilt.

## Index

42 bugs, highest severity first.

| ID | Fix | Class | Sev | One line |
|---|---|---|---|---|
| [MOD-001](_module_level/training.prompt_decomposition.data.dataset/MOD-001-dataset-builders-import-a-module-and-a-loader-that-no-longer-exist/bug.md) | [fix](_module_level/training.prompt_decomposition.data.dataset/MOD-001-dataset-builders-import-a-module-and-a-loader-that-no-longer-exist/fix.md) | (module) training.prompt_decomposition.data.dataset | **high** | `build_real_only_dataset` / `build_task_dataset` always raise `ImportError`; the former also calls a deleted `load_handlabelled` |
| [ROUTER-001](Router/ROUTER-001-fallback-cost-scale-mismatch/bug.md) | [fix](Router/ROUTER-001-fallback-cost-scale-mismatch/fix.md) | Router | med | Unsupported candidates get a per-token cost against per-query costs; an unseen model can win |
| [ROUTERMODEL-002](RouterModel/ROUTERMODEL-002-cost-model-never-consumed/bug.md) | [fix](RouterModel/ROUTERMODEL-002-cost-model-never-consumed/fix.md) | RouterModel | med | `cost_model` is stored, never read |
| [MLPROUTER-001](RouterModel/MLPRouter/MLPROUTER-001-from-artifact-loses-pathway-and-data-config/bug.md) | [fix](RouterModel/MLPRouter/MLPROUTER-001-from-artifact-loses-pathway-and-data-config/fix.md) | MLPRouter | med | `from_artifact` serves with the default pathway and data config |
| [ROUTERMODELFACTORY-001](RouterModelFactory/ROUTERMODELFACTORY-001-cost-model-kwarg-overwritten/bug.md) | [fix](RouterModelFactory/ROUTERMODELFACTORY-001-cost-model-kwarg-overwritten/fix.md) | RouterModelFactory | med | `create(artifact, cost_model=cm)` drops `cm` |
| [MLPROUTERTRAINER-001](RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-001-hyperparameters-not-persisted/bug.md) | [fix](RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-001-hyperparameters-not-persisted/fix.md) | MLPRouterTrainer | med | Artifact stores no hyperparameters or pathway |
| [LANGCHAINADAPTER-001](ProviderAdapter/LangChainAdapter/LANGCHAINADAPTER-001-api-key-not-forwarded/bug.md) | [fix](ProviderAdapter/LangChainAdapter/LANGCHAINADAPTER-001-api-key-not-forwarded/fix.md) | LangChainAdapter | med | Explicit `api_key` is ignored |
| [LOCALARTIFACTSTORE-002](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-002-save-clobbers-fit-run-json/bug.md) | [fix](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-002-save-clobbers-fit-run-json/fix.md) | LocalArtifactStore | med | `save` overwrites `fit()`'s `run.json` (history, val_metrics, git sha) |
| [TRAININGHARNESS-003](TrainingHarness/TRAININGHARNESS-003-unpopulated-metrics-read-as-perfect/bug.md) | [fix](TrainingHarness/TRAININGHARNESS-003-unpopulated-metrics-read-as-perfect/fix.md) | TrainingHarness | med | Unreported metrics default to 0.0, the best value for loss/bce/regret |
| [AGENTICROUTER-001](AgenticRouter/AGENTICROUTER-001-result-reports-pool-best-quality/bug.md) | [fix](AgenticRouter/AGENTICROUTER-001-result-reports-pool-best-quality/fix.md) | AgenticRouter | med | Result reports the pool-best quality next to the cheaper model that answered (`lam > 0`); always optimistically biased |
| [AGENTICROUTER-002](AgenticRouter/AGENTICROUTER-002-unknown-cost-reported-as-zero/bug.md) | [fix](AgenticRouter/AGENTICROUTER-002-unknown-cost-reported-as-zero/fix.md) | AgenticRouter | med | Unknown call cost (`None`) is reported as `0.0` |
| [LANGCHAINTOOLORCHESTRATOR-001](LangChainToolOrchestrator/LANGCHAINTOOLORCHESTRATOR-001-tool-errors-abort-the-executor/bug.md) | [fix](LangChainToolOrchestrator/LANGCHAINTOOLORCHESTRATOR-001-tool-errors-abort-the-executor/fix.md) | LangChainToolOrchestrator | med | `handle_tool_error=True` only catches `ToolException`; `LookupError` aborts the run |
| [NAIVEDECOMPOSER-001](NaiveDecomposer/NAIVEDECOMPOSER-001-silently-drops-subtasks/bug.md) | [fix](NaiveDecomposer/NAIVEDECOMPOSER-001-silently-drops-subtasks/fix.md) | NaiveDecomposer | med | Dropped short sub-tasks and anything past `max_parts` without a signal; now merged / folded |
| [LLMCLIENTFACTORY-001](LLMClientFactory/LLMCLIENTFACTORY-001-cache-ignores-profile/bug.md) | [fix](LLMClientFactory/LLMCLIENTFACTORY-001-cache-ignores-profile/fix.md) | LLMClientFactory | med | Cache key omits the profile; a re-priced profile keeps old prices |
| [ROUTER-003](Router/ROUTER-003-no-registry-capability-filter-drops-everything/bug.md) | [fix](Router/ROUTER-003-no-registry-capability-filter-drops-everything/fix.md) | Router | med | No-registry mode + `required_capabilities` silently empties routing |
| [ROUTERMODEL-004](RouterModel/ROUTERMODEL-004-subclass-init-drops-cost-model-and-artifact/bug.md) | [fix](RouterModel/ROUTERMODEL-004-subclass-init-drops-cost-model-and-artifact/fix.md) | RouterModel | med | Subclass `__init__`s don't accept/forward `cost_model`/`artifact` |
| [MOD-001](_module_level/router.llm.adapters/MOD-001-guess-provider-missing-end-boundary/bug.md) | [fix](_module_level/router.llm.adapters/MOD-001-guess-provider-missing-end-boundary/fix.md) | (module) router.llm.adapters | med | `guess_provider` matches hints mid-token; `gpt4all-13b-snoozy` → wrongly `"openai"` |
| [LANGCHAINADAPTER-002](ProviderAdapter/LangChainAdapter/LANGCHAINADAPTER-002-partial-price-defaults-to-zero/bug.md) | [fix](ProviderAdapter/LangChainAdapter/LANGCHAINADAPTER-002-partial-price-defaults-to-zero/fix.md) | LangChainAdapter | med | A half-priced profile prices the unset side at $0, not an error |
| [DOMAINCLASSIFIER-001](DomainClassifier/DOMAINCLASSIFIER-001-merge-domains-not-set/bug.md) | [fix](DomainClassifier/DOMAINCLASSIFIER-001-merge-domains-not-set/fix.md) | DomainClassifier (prompt_heads) | med | Built without `merge_domains=True`; runs the unmerged 10-class head |
| [MLPROUTERTRAINER-003](RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-003-seed-precedence-inverted/bug.md) | [fix](RouterModelTrainer/MLPRouterTrainer/MLPROUTERTRAINER-003-seed-precedence-inverted/fix.md) | MLPRouterTrainer | med | `seed` precedence inverted: an explicit hyperparameter beats `config.seed` |
| [MOD-001](_module_level/training.nirt/MOD-001-runs-dir-resolution-asymmetric-vs-baseline/bug.md) | [fix](_module_level/training.nirt/MOD-001-runs-dir-resolution-asymmetric-vs-baseline/fix.md) | (module) training.nirt | med | Explicit relative `runs_dir` resolves differently than the baseline's `out_dir` |
| [MOD-001](_module_level/evaluation.nirt.routing/MOD-001-train-quality-silently-ignores-unknown-metric/bug.md) | [fix](_module_level/evaluation.nirt.routing/MOD-001-train-quality-silently-ignores-unknown-metric/fix.md) | (module) evaluation.nirt.routing | med | `train_quality` silently ignores an unrecognized `metric` value |
| [MOD-001](_module_level/evaluation.routing.oracle/MOD-001-compare-routers-no-missing-model-check/bug.md) | [fix](_module_level/evaluation.routing.oracle/MOD-001-compare-routers-no-missing-model-check/fix.md) | (module) evaluation.routing.oracle | med | `compare_routers` skips the missing-model guard its sibling `evaluate_router` has |
| [MOD-001](_module_level/training.taxonomy/MOD-001-clusters-fit-on-all-splits-including-ood/bug.md) | [fix](_module_level/training.taxonomy/MOD-001-clusters-fit-on-all-splits-including-ood/fix.md) | (module) training.taxonomy | med | Clusters / centroids / `r_q` are fit on val, test and held-out OOD queries (11,144 test, 19,850 math/code) |
| [MOD-001](_module_level/training.data.loaders/MOD-001-irt-router-zero-token-zero-cost-rows-priced-free/bug.md) | [fix](_module_level/training.data.loaders/MOD-001-irt-router-zero-token-zero-cost-rows-priced-free/fix.md) | (module) training.data.loaders | med | 10,565 IRT-Router rows with 0 tokens + $0 cost (real scores) ingested as free; 21% of `qwen25_72b_instruct` |
| [ORCHESTRATOR-001](Orchestrator/ORCHESTRATOR-001-release-then-reserve-window/bug.md) | [fix](Orchestrator/ORCHESTRATOR-001-release-then-reserve-window/fix.md) | Orchestrator | med | `run()` released a child's spawn-time hold before re-reserving; a sibling could take it |
| [LOCALARTIFACTSTORE-001](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-001-mlp-roundtrip-drops-version-fields/bug.md) | [fix](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-001-mlp-roundtrip-drops-version-fields/fix.md) | LocalArtifactStore | low-med | MLP reload loses `router_model_version`, `training_dataset_version` |
| [LLMDECOMPOSER-001](LLMDecomposer/LLMDECOMPOSER-001-prefix-strip-corrupts-decimal-task/bug.md) | [fix](LLMDecomposer/LLMDECOMPOSER-001-prefix-strip-corrupts-decimal-task/fix.md) | LLMDecomposer | low-med | `3.14 times 2` becomes `14 times 2` |
| [LOCALARTIFACTSTORE-006](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-006-read-run-json-swallows-errors/bug.md) | [fix](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-006-read-run-json-swallows-errors/fix.md) | LocalArtifactStore | low-med | A corrupt `run.json` reads back as "no metadata", not an error |
| [TEXTENCODER-001](TextEncoder/TEXTENCODER-001-encode-reseeds-global-rngs/bug.md) | [fix](TextEncoder/TEXTENCODER-001-encode-reseeds-global-rngs/fix.md) | TextEncoder | med-low | Every `encode()` reseeded the process-global `random` / numpy / torch RNGs |
| [ROUTER-002](Router/ROUTER-002-decision-artifact-id-never-set/bug.md) | [fix](Router/ROUTER-002-decision-artifact-id-never-set/fix.md) | Router | low | `RoutingDecision.artifact_id` is never set |
| [ROUTERMODEL-001](RouterModel/ROUTERMODEL-001-call-alias-bypasses-subclass-route/bug.md) | [fix](RouterModel/ROUTERMODEL-001-call-alias-bypasses-subclass-route/fix.md) | RouterModel | low | `__call__ = route` ignores subclass overrides |
| [ROUTERMODEL-003](RouterModel/ROUTERMODEL-003-resolve-costs-series-positional/bug.md) | [fix](RouterModel/ROUTERMODEL-003-resolve-costs-series-positional/fix.md) | RouterModel | low | Labelled `pd.Series` costs are treated as positional |
| [LOCALARTIFACTSTORE-003](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-003-artifact-id-escapes-root/bug.md) | [fix](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-003-artifact-id-escapes-root/fix.md) | LocalArtifactStore | low | `artifact_id` such as `../x` writes outside the root |
| [LOCALARTIFACTSTORE-004](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-004-load-reads-model-pt-twice/bug.md) | [fix](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-004-load-reads-model-pt-twice/fix.md) | LocalArtifactStore | low | `load` decodes `model.pt` two to three times |
| [LOCALARTIFACTSTORE-005](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-005-save-order-can-pair-new-weights-with-old-metadata/bug.md) | [fix](ArtifactStore/LocalArtifactStore/LOCALARTIFACTSTORE-005-save-order-can-pair-new-weights-with-old-metadata/fix.md) | LocalArtifactStore | low | Re-save can pair new weights with old `run.json` |
| [AGENTICROUTER-004](AgenticRouter/AGENTICROUTER-004-run-state-lives-on-the-instance/bug.md) | [fix](AgenticRouter/AGENTICROUTER-004-run-state-lives-on-the-instance/fix.md) | AgenticRouter | low | Per-run state on the instance; concurrent `run()` calls interfere |
| [CLIENTREGISTRY-001](ClientRegistry/CLIENTREGISTRY-001-echo-warning-skipped-after-register/bug.md) | [fix](ClientRegistry/CLIENTREGISTRY-001-echo-warning-skipped-after-register/fix.md) | ClientRegistry | low | Echo-fallback warning skipped for clients added via `register()` |
| [LLMCLIENT-001](LLMClient/LLMCLIENT-001-unpriced-profile-prices-calls-at-zero/bug.md) | [fix](LLMClient/LLMCLIENT-001-unpriced-profile-prices-calls-at-zero/fix.md) | LLMClient | low | Unpriced profile (default `0.0`) prices every counted call at zero |
| [BUDGETLEDGER-001](BudgetLedger/BUDGETLEDGER-001-commit-overshoots-total-silently/bug.md) | [fix](BudgetLedger/BUDGETLEDGER-001-commit-overshoots-total-silently/fix.md) | BudgetLedger | low (latent) | `commit` can push `spent` past `total` with no signal |
| [LLMRESPONSE-001](LLMResponse/LLMRESPONSE-001-raw-meta-fields-dropped/bug.md) | [fix](LLMResponse/LLMRESPONSE-001-raw-meta-fields-dropped/fix.md) | LLMResponse | low | `raw`/`meta` fields lost in the `LLMResponse` unification |
| [PLANFORECASTER-001](PlanForecaster/PLANFORECASTER-001-score-plan-resets-plan-status/bug.md) | [fix](PlanForecaster/PLANFORECASTER-001-score-plan-resets-plan-status/fix.md) | PlanForecaster | low | `score_plan` moved approved / executing plans back to `EVALUATED` |
