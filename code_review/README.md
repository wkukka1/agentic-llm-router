# Code review of `src/` — index

This is a review of all 140 Python files under `src/` (about 14,300 lines), in scope for correctness, design/duplication and performance. Style, naming, docstrings and type hints are out of scope. No source file was changed.

**Totals:** 215 findings (190 per-package, 25 cross-cutting). None is rated high. 57 are medium, 158 are low. 181 are confirmed and 34 are suspected.

- **Confirmed** means the path was traced end to end, and a probe was run where noted.
- **Suspected** means plausible but not provable from the code alone. Each one is listed below as a question.
- **Severity:** high is wrong results or a crash on a real path, medium is wrong under a plausible config, low is latent.

Each finding cites `path:start-end` and quotes the exact lines. `_contracts.md` is the interface map the reviewers worked from; it contains no findings.

---

## 1. Counts

| Doc | Corr. conf | Corr. susp | Design conf | Design susp | Perf conf | Perf susp | Total |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| [router.md](router.md) | 6 | 0 | 2 | 0 | 1 | 0 | 9 |
| [router.agentic.md](router.agentic.md) | 11 | 5 | 2 | 1 | 2 | 0 | 21 |
| [router.data.md](router.data.md) | 2 | 0 | 0 | 0 | 1 | 0 | 3 |
| [router.embeddings.md](router.embeddings.md) | 6 | 0 | 1 | 0 | 0 | 0 | 7 |
| [router.execution.md](router.execution.md) | 6 | 2 | 5 | 1 | 0 | 0 | 14 |
| [router.llm.md](router.llm.md) | 3 | 1 | 4 | 0 | 1 | 0 | 9 |
| [router.nirt.md](router.nirt.md) | 3 | 1 | 3 | 0 | 2 | 0 | 9 |
| [router.routing.md](router.routing.md) | 8 | 2 | 4 | 1 | 2 | 0 | 17 |
| [decompose.md](decompose.md) | 0 | 0 | 2 | 0 | 1 | 0 | 3 |
| [training.md](training.md) | 2 | 0 | 1 | 1 | 1 | 0 | 5 |
| [training.data.md](training.data.md) | 7 | 2 | 1 | 0 | 3 | 0 | 13 |
| [training.nirt.md](training.nirt.md) | 10 | 2 | 1 | 1 | 3 | 0 | 17 |
| [training.nirt.baseline.md](training.nirt.baseline.md) | 11 | 5 | 3 | 3 | 4 | 0 | 26 |
| [training.retrieval.md](training.retrieval.md) | 3 | 0 | 1 | 0 | 1 | 0 | 5 |
| [training.taxonomy.md](training.taxonomy.md) | 1 | 1 | 0 | 0 | 0 | 0 | 2 |
| [training.models.md](training.models.md) | 3 | 0 | 0 | 0 | 1 | 0 | 4 |
| [evaluation.md](evaluation.md) | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| [evaluation.judge.md](evaluation.judge.md) | 3 | 0 | 1 | 0 | 0 | 0 | 4 |
| [evaluation.nirt.md](evaluation.nirt.md) | 3 | 4 | 0 | 0 | 1 | 0 | 8 |
| [evaluation.pool_expansion.md](evaluation.pool_expansion.md) | 1 | 1 | 2 | 0 | 1 | 0 | 5 |
| [evaluation.routing.md](evaluation.routing.md) | 5 | 0 | 0 | 0 | 1 | 0 | 6 |
| [_cross_cutting_architecture.md](_cross_cutting_architecture.md) | 5 | 0 | 6 | 0 | 0 | 0 | 11 |
| [_cross_cutting_duplication.md](_cross_cutting_duplication.md) | 1 | 0 | 13 | 0 | 0 | 0 | 14 |
| **Total** | **103** | **26** | **52** | **8** | **26** | **0** | **215** |

---

## 2. Top findings to look at first

Sorted by how much they can distort results or cost you work. All are medium and confirmed.

1. **[XA-02](_cross_cutting_architecture.md)**: offline experiments use a different cost-aware rule than the router. `pareto`, `routing_report` and the pool-expansion battery route on each query's realised cost normalised by the max cell. The λ values and savings in the ledger therefore don't correspond to `Router.route`.
2. **[XA-01](_cross_cutting_architecture.md)**: `evaluate_router` and `compare_routers` pick with eval-split mean costs, while `Router.route` uses train-split costs. Cost-aware regret describes a router that doesn't exist.
3. **[EP-01](evaluation.pool_expansion.md)**: the headline "saving at −1 / −3 pt" numbers choose λ on the test split.
4. **[RE-01](router.embeddings.md)**: `build_store` returns the old vectors after an encoder config or text change. Probe-verified.
5. **[RE-02](router.embeddings.md)**: a killed rebuild leaves a store that loads with new ids, zero vectors and the old "complete" manifest. Probe-verified.
6. **[TM-01](training.models.md)**: edited or re-rendered model profiles are never re-embedded without `--restart`.
7. **[TD-01](training.data.md)**: anchor-judge rows take over `dataset` and `source` for the RouterBench queries they judge in `queries.parquet`. This breaks family labels, features and taxonomy. Probe-verified.
8. **[TD-05](training.data.md)**: substring family matching puts MMLU math subjects in `math`, so they are held out as OOD. Probe-verified.
9. **[TD-02](training.data.md)**: with `clip: true`, chance correction is an identity on 0/1 scores. `score_effective` equals raw for binary MC items.
10. **[TRV-01](training.retrieval.md)**: kNN self-exclusion is by id, so a query's 0-shot/5-shot twin (identical text) imputes it from itself in train but not in eval.
11. **[RN-01](router.nirt.md) / [XA-04](_cross_cutting_architecture.md)**: projected-mode `center_discrimination` centres on a batch mean. Inference results depend on 8192-row chunk boundaries, and the pairwise loss uses two different centres.
12. **[RR-04](router.routing.md)**: a NaN model cost always wins the argmax when `lam > 0`.
13. **[RR-05](router.routing.md)**: the hard constraints `max_cost`, `max_latency` and `min_quality` are never enforced.
14. **[XA-03](_cross_cutting_architecture.md) + [EJ-01](evaluation.judge.md)**: judge answers written as `+N` fail to parse and become ties in the training data.
15. **[EN-01](evaluation.nirt.md)**: `cold_start_eval` ignores `query_features` and `query_pathway`. It crashes on features runs and silently mis-scores kNN runs.
16. **[TN-03](training.nirt.md)**: the collapse alarm fires every epoch at `dim: 1`, and `abort_on_collapse` stops training after 3 epochs.
17. **[RA-03](router.agentic.md) / [RA-01](router.agentic.md)**: the live agentic path never records real-model cost, and it double-counts cost for nested orchestration.
18. **[ER-01](evaluation.routing.md)**: the oracle-classifier baseline labels ties by column order, so it learns pool ordering. *(Fixed 2026-09-14, with ER-02..06 and XD-02.)*

---

## 3. One line per doc

- **[router.md](router.md):** top-level pipeline, policy, config, provenance, tools and tracing. An empty registry routes over the whole pool, NaN scores break policy ranking, and seeding has process-wide side effects.
- **[router.agentic.md](router.agentic.md):** live agentic flow. Cost is not recorded or is double-counted, there is no error handling, unknown ids silently get an echo client, and the sentence splitter is broken.
- **[router.data.md](router.data.md):** `NIRTDataset`. An empty frame with features crashes, and id-type mismatches silently drop every row.
- **[router.embeddings.md](router.embeddings.md):** encoder and store. Stale and partial stores are served as complete, resume breaks on a device change, and per-pathway config keys are ignored.
- **[router.execution.md](router.execution.md):** unwired scaffolding. The budget ledger invariants and spawn/aggregate logic are wrong and should be fixed before wiring.
- **[router.llm.md](router.llm.md):** unwired scaffolding. Pricing is dropped, and the provider allow-list is too narrow.
- **[router.nirt.md](router.nirt.md):** model, predict and shrinkage. Batch-dependent centring, NaN with an empty bank, silently ignored config knobs, and an unsafe `torch.load`.
- **[router.routing.md](router.routing.md):** the `Router` interface. Eligibility masks fail open, NaN costs win, constraints are unenforced, and short embeddings are zero-padded.
- **[decompose.md](decompose.md):** signal extraction scaffolding. Signal names collide, and reading `.dimension` loads the model.
- **[training.md](training.md):** CLI, phase0 and the MLP trainer. The MLP baseline sees different inputs than the NIRT run it is compared with.
- **[training.data.md](training.data.md):** loaders, splits, families and quality. Anchor-judge rows take over query metadata, chance correction is a no-op, the family mapping is wrong, and a smoke run overwrites the full split store.
- **[training.nirt.md](training.nirt.md):** trainer and metrics. Ranking losses crash, the collapse alarm false-fires, and `val_metric` has broken selection logic.
- **[training.nirt.baseline.md](training.nirt.baseline.md):** the Phase 1/2 control arm. Metrics that can't be compared are shown side by side, `score_kind` is ignored, and thresholds are hard-coded.
- **[training.retrieval.md](training.retrieval.md):** FAISS bank and kNN imputation. Self-exclusion leaks text twins, and the store cache check is too weak.
- **[training.taxonomy.md](training.taxonomy.md):** clustering and `r_q`. There is no centroid/relevance consistency check, and clustering silently widens to all queries.
- **[training.models.md](training.models.md):** model profiles. Stale embeddings, an empirical-stats default that leaks, and a `$None` price.
- **[evaluation.md](evaluation.md):** classical IRT and the judge data prep. `main_effects` trains on the scored split, and duplicate prompts get mismatched labels.
- **[evaluation.judge.md](evaluation.judge.md):** judge client. `+N` margins are lost, a `None` content crashes, and non-transient errors are retried.
- **[evaluation.nirt.md](evaluation.nirt.md):** cold-start, OOD and routing reports. Cold-start ignores the run's input pathway, and the learning curve has survivorship bias.
- **[evaluation.pool_expansion.md](evaluation.pool_expansion.md):** ledger battery. Operating points are test-selected, provenance paths are wrong, and checkpoints are reloaded repeatedly.
- **[evaluation.routing.md](evaluation.routing.md):** oracle evaluation. The classifier's ties follow column order, and the per-query table disagrees with the summary under masks.
- **[_cross_cutting_architecture.md](_cross_cutting_architecture.md):** evaluated versus served decision rules, judge-to-training flow, checkpoint formats, and build-order dependence.
- **[_cross_cutting_duplication.md](_cross_cutting_duplication.md):** regret, utility, oracle tie-break, `effective_rank`, family maps, observation-table access, and the two LLM client stacks.

---

## 4. Needs your call (suspected findings as yes/no questions)

- [ ] **RA-09:** does hitting `max_iterations` in the LangChain executor return its stop message as the final answer in your installed LangChain version?
- [ ] **RA-10:** will you install LangChain 1.x, where `AgentExecutor` / `create_tool_calling_agent` are no longer in `langchain.agents`?
- [ ] **RA-11:** do any of your chat models return list-of-blocks content (which would be stringified as a Python repr)?
- [ ] **RA-19:** do pool model ids ever reach `guess_provider` as if they were provider model names?
- [ ] **RA-20:** can `AgenticResult.summary()` run with `predicted_quality=None`?
- [ ] **RA-21:** do real `LLMDecomposer` outputs include numbering or preamble lines?
- [ ] **RX-12:** will the budget ledger be used concurrently (check-then-act race)?
- [ ] **RX-13:** will one orchestrator instance run more than one task at a time?
- [ ] **RX-14:** should `ExecutionResult.success` default to `False`?
- [ ] **RL-02:** will the `router.llm` factory be pointed at non-OpenAI/Anthropic/Google models?
- [ ] **RN-04:** can a NIRT training batch contain exactly one row when `query_head_batchnorm: true`?
- [ ] **RR-06:** do you load runs trained against a non-default phase-0 config through `NIRTRouter.from_run`?
- [ ] **RR-16:** do you reload router modules in a live process (notebooks, hot reload)?
- [ ] **RR-17:** can an `MLPRouter` be constructed with a pathway different from the one it was fit on?
- [ ] **TR-05:** is the MLP baseline ever compared against NIRT runs that use `query_pathway` or `query_features`?
- [ ] **TD-07:** were your lm-harness logs produced by lm-eval ≥ 0.4 (`samples_<task>_<ts>.jsonl` layout)?
- [ ] **TD-09:** do IRT-Router's `train.csv` and `test1.csv` / `test2.csv` share any `(task, question)` pairs?
- [ ] **TN-06:** is the preference weight meant to be a loss weight rather than a separate Adam step?
- [ ] **TN-12:** can validation-only model ids appear in `model_params: free` runs?
- [ ] **TN-17:** is the cold-start probe meant to use a different score function from `NIRTModel`'s logit?
- [ ] **TB-05:** can `y_soft` contain NaN or out-of-range values when it reaches the continuous losses?
- [ ] **TB-20:** can boundary statistics receive NaN scores?
- [ ] **TB-21:** should cold-start labels be clipped to [0, 1] when training and eval labels are not?
- [ ] **TB-22:** should a missing `r_q` be uniform `1/C` (as in the data path) rather than zeros?
- [ ] **TB-23:** can the warm-up blend be inactive during training but active at eval in any config you run?
- [ ] **TB-24:** are the θ identifiability regularisers meant to act on per-observation θ in projected mode?
- [ ] **TB-25:** is `target: soft` ever used with the Bernoulli synthetic world (which trains on the true probability)?
- [ ] **TB-26:** is the Normal "well-specified" synthetic world meant to be censored to [0, 1]?
- [ ] **TX-01:** does the Phase 1 data join validate `r_q` against the current centroids?
- [ ] **EN-04:** is `lomo_eval` run on observations that include multi-metric sources (lm-harness `acc` + `acc_norm`)?
- [ ] **EN-05:** do you run OOD passes for models trained with `query_features`?
- [ ] **EN-06:** can the profile store lack any warm pool model when `cold_start_eval` runs?
- [ ] **EN-07:** do real ZOIB frontiers contain repeated cost points after clipping?
- [x] **EP-03:** can a battery's pinned checkpoint pool contain a model absent from the evaluated split? → Treated as possible: the battery now fails fast, listing the missing models (validation-only gaps mark λ selection `unavailable` instead).

---

## 5. Triage checklist

- [ ] Decide the served cost-aware rule, and make evaluation and experiments use it (XA-01, XA-02, EP-01, EN-03).
- [ ] Fix store staleness and partial writes before the next embedding rebuild (RE-01, RE-02, TM-01, TD-03).
- [ ] Fix `queries.parquet` metadata and families, then rebuild the data tables (TD-01, TD-05, TD-02 decision).
- [ ] Fix NaN handling in routing (RR-04, RR-02, RR-03, RT-02, RA-02).
- [ ] Enforce or remove the hard constraints (RR-05).
- [ ] Fix kNN self-exclusion by content hash and rebuild the `knn*` stores (TRV-01, TRV-02).
- [ ] Replace batch-mean centring with a pool-level buffer (RN-01, XA-04).
- [x] Fix the judge parser and filter `parsed_ok` before re-collecting judgments (EJ-01, EJ-02, XA-03). Also EJ-03 and EJ-04.
- [ ] Make `cold_start_eval` and `ood_datasets` honour the run's input config (EN-01, EN-05, TR-05).
- [ ] Fix the trainer crashes and selection logic (TN-01, TN-02, TN-03, TN-04).
- [ ] Fix the agentic cost accounting and error handling (RA-01, RA-03, RA-04, RA-05).
- [ ] Answer section 4, then promote or drop each suspected finding.
- [ ] Schedule the duplication cleanups (`_cross_cutting_duplication.md`), starting with XD-02, XD-05 and XD-07.

---

## Verification performed

- **Import linter** (`.venv/Scripts/lint-imports`): 3 contracts kept, 0 broken. The three sanctioned exemptions are not reported as violations.
- **Tests** (`pytest tests -q --import-mode=importlib`): 427 passed, 2 failed.
  - **Failing tests:** `tests/training/data/test_facade.py::test_query_embeddings_read_only_by_default_returns_none` and `::test_profile_embeddings_read_only_by_default_returns_none`. The toy config resolves `embedding.cache_dir` against the real repo root, where a real `query__irt` store exists. Note that `test_facade.py` and `facade.py` have uncommitted edits in the working tree.
  - **Collection:** the default import mode fails at collection, because `tests/router/embeddings/test_embeddings.py` and `tests/training/data/test_embeddings.py` share a basename.
- **Line citations:** a script re-read every quoted `NNN: code` line across all docs against the current files. It checked 516 lines. 2 mismatches were found (ER-05) and fixed.
- **Probes:** behaviour claims marked "probe" were run against the real functions with `.venv/Scripts/python.exe`. Covered: store staleness, the partial rebuild, family mapping, `_build_queries`, `render_prompt`, chance correction, and empty float indexing. One suspected issue (an `Int64` cast crash when collapsing duplicates) was disproved by probe and removed.
- **Working tree:** the review reads the current working tree. While the review ran, uncommitted changes appeared outside `code_review/`, in `facade.py`, `profiles.py`, `training/data/embeddings.py`, `scripts/embeddings/*` and several tests. The reviewers did not make them.
