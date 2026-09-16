# Cross-cutting review: architecture and boundaries

These findings are about how packages interact: which decision rule is evaluated versus served, what flows between pipelines, and interfaces that don't line up. They merge the raw notes from reviewers A2, A5, A6 and A8 with issues found while finishing the review.

**Import-linter ground truth:** `.venv/Scripts/lint-imports` analysed 117 files and 187 dependencies. All three contracts are KEPT: "Serving never imports training or evaluation" (1 ignored import), "Training never imports evaluation", and "Decompose stays dependency-free" (2 ignored imports). None of the findings below is an import-contract violation. The three sanctioned exemptions are not reported.

**Summary:** 11 findings: 5 correctness, 6 design · 11 confirmed

| ID | Topic | Severity | Category |
|----|-------|----------|----------|
| XA-01 | Evaluated router uses a different cost vector than the served router | medium | correctness |
| XA-02 | Two different cost-aware decision rules | medium | correctness |
| XA-03 | Judge parse failures enter training data as ties | medium | correctness |
| XA-04 | Batch-mean centring breaks pairwise training and inference | medium | correctness |
| XA-05 | Two incompatible `model.pt` formats and `load_run` functions | low | design |
| XA-06 | Trainers import private router helpers | low | design |
| XA-07 | The documented tool recursion guard isn't what the live tools do | low | design |
| XA-08 | Depth/budget limiting implemented twice, with different `max_depth` meaning | low | correctness |
| XA-09 | `RoutingPipeline` and `AgenticRouter` disagree on cost handling | low | design |
| XA-10 | Three provider vocabularies | low | design |
| XA-11 | Pipeline stages silently depend on stale outputs of earlier runs | low | design |

---

### XA-01: Cost-aware router evaluation uses a different cost vector than the served router (A5-X1)
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed
**Spans:** `src/evaluation/routing/oracle.py:73-91`, `src/evaluation/routing/oracle.py:217-220`, `src/evaluation/routing/oracle.py:510-532`, `src/router/routing/base.py:261-271`, `src/router/routing/routers.py:63-86`

```python
217:    C = (np.asarray(model_costs, np.float64) if model_costs is not None
218:         else (cost.mean(axis=0) if cost is not None else None))
220:    selected = routing_decision(pred, lam=lam, model_costs=C, eligible=eligible)
```

**What breaks:** The two paths pick with different costs:
- **Served:** `Router.route(lam)` selects with `router.default_model_costs`, the per-model mean cost on the **train** split.
- **Evaluated:** `evaluate_router` and `compare_routers` pass no `model_costs`, so `routing_evaluation` recomputes `C` as the column mean of the **evaluation split's** cost matrix.

For `lam > 0`, the evaluated selections differ from what `router.route(qids, lam=lam)` returns on the same queries. The reported cost-aware regret and hit rate describe a router that doesn't exist, built with eval-split costs (a mild label-side leak).
**Evidence:** Traced both paths. `test_compare_routers_adds_oracle_and_floor` (`tests/router/routing/test_routing_interface.py:186-195`) uses `lam=0.5` but never compares against `router.route(...).selected`.
**Direction:** Evaluate the selection `router.route(qids, lam=lam)` produces, or pass `model_costs=router.default_model_costs`.

---

### XA-02: Offline experiments use a different cost-aware decision rule than the router
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed
**Spans:** `src/evaluation/nirt/routing.py:87-91`, `src/router/nirt/routing_decision.py:35-45`, `src/evaluation/pool_expansion/battery.py:268-272,332`

```python
89:    cmax = cost.max()
90:    cost_norm = cost / cmax if cmax > 0 else cost
91:    return np.argmax(pred - lam * cost_norm, axis=1)
```

**What breaks:** `evaluation.nirt.routing.route` computes `pred - lam * cost[q, m] / max(cost)`:
- **Cost source:** the per-cell cost of the query being routed, which comes from the outcome and isn't known before the call.
- **Scale:** normalised by the single most expensive cell in the eval matrix.
- **No masking:** non-finite predictions or costs aren't masked.

The served `routing_decision` instead uses an unnormalised per-model cost vector and masks NaNs. `pareto`, `routing_report` and the whole pool-expansion battery (frontier, `minus3pt_lambda`, ledger savings) use `route`. Consequences:
- **Non-transferable λ:** "ZOIB saves X% at λ=0.5" corresponds to no λ for `Router.route`, and the scale changes whenever one outlier cell changes `cost.max()`.
- **Cost leak:** the offline policy sees each query's realised cost, so the frontier is optimistic compared with any deployable router.

**Evidence:** Traced `battery.routing_block` → `pareto(zoib_ey, true, cost)` → `route(pred, cost, lam)`, and compared it with `routing_decision`.
**Direction:** Make `route` call `routing_decision` with a per-model cost vector computed on train, and report the λ scale that goes with it.

---

### XA-03: Judge parse failures enter the training data as ties
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed
**Spans:** `src/evaluation/judge/client.py:40-54`, `scripts/data/collect_anchor_judgments.py:251`, `src/training/data/loaders.py:529-541`

```python
54:    return 0, raw[:400], False                              # client.py: unparsed -> margin 0
530:    judgments = judgments[~judgments["swapped"]]            # loaders.py: only swaps filtered
541:        outcome_anchor = 0.5 if margin == 0 else (1.0 if margin < 0 else 0.0)
```

**What breaks:** An unparseable verdict is returned as `margin=0, parsed_ok=False`. The collection script stores `parsed_ok` in `judgments.parquet`. `load_anchor_judge` filters only `swapped` rows, so every failed parse becomes a `0.5` tie observation in `responses.parquet` and flows into `TrainingData.pairwise()`. EJ-01 makes these failures systematic: every `+N` answer ("B is better") becomes a tie. That dilutes the candidate's wins, which is exactly the judge-versus-gold disagreement signal the pipeline measures.
**Evidence:** Traced client → script → loader. Grep for `parsed_ok` in `src/training` finds nothing.
**Direction:** Filter `parsed_ok` in `load_anchor_judge` and report the dropped count, and fix EJ-01.

---

### XA-04: Batch-mean discrimination centring breaks both pairwise training and inference (A2-X1, RN-01)
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed
**Spans:** `src/router/nirt/model.py:278-283`, `src/training/nirt/pairwise.py:95-103`, `src/training/nirt/train.py:504-517`, `src/router/nirt/predict.py:42-47`

```python
282:            r = a.mean(0, keepdim=True)            # model.py, projected mode
96:    a_a, b_a = model.model_parameters(e_a)       # pairwise.py
97:    a_b, b_b = model.model_parameters(e_b)
```

**What breaks:** With `center_discrimination: true` in projected mode, `r` is the mean over whatever rows share a forward call. The model's docstring says centring is lossless only when every compared candidate shares `r`. That doesn't hold in two places:
- **Training:** `pairwise_logit_diff` centres the `e_a` and `e_b` batches separately, so `z_a − z_b` picks up a spurious `−(r_a − r_b)·θ_q` term that biases the Bradley–Terry gradient.
- **Inference:** `predict_dataset` centres over arbitrary 8192-row chunks (RN-01).

The same model therefore trains and scores against moving reference points.
**Evidence:** `test_pairwise_logit_diff_matches_manual_bilinear` (`tests/training/nirt/test_pairwise.py:79-88`) runs with centring off. No test checks batch invariance of `predict_dataset`.
**Direction:** Compute projected-mode `r` once over the model pool and store it as a buffer. Both call sites then share it.

---

### XA-05: The two `model.pt` formats and two `load_run` functions are mutually incompatible (A2-X3)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/training/nirt/train.py:602-612`, `src/router/nirt/checkpoint.py:19-38`, `src/training/nirt/baseline/checkpoint.py:37-45`, `src/training/nirt/baseline/checkpoint.py:56-82`

```python
31:        blob["config"].get("model", {}),     # router/nirt/checkpoint.py
62:        blob["model_cfg"], n_models=len(blob["model_index"]), query_dim=blob["query_dim"],   # baseline/checkpoint.py
```

**What breaks:** Pointing either loader at the other kind of run fails with an opaque error:
- `router.nirt.checkpoint.load_run` on a baseline directory raises `KeyError: 'config'`.
- The baseline loader on a NIRT run raises `KeyError: 'model_cfg'`.

Both files are named `model.pt`, and the two functions share a name but have different signatures and return tuples. Callers that import both (`scripts/route_compare.py`, `evaluation/pool_expansion/battery.py:33,76`) must alias one of them.
**Evidence:** Read the writer and both readers. Neither reader checks a format tag.
**Direction:** Write a `format` key into each blob, assert it in each loader, and rename one of them (for example `load_baseline_run`).

---

### XA-06: `training.trainers.mlp_router` imports three private router helpers (A2-X4)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/training/trainers/mlp_router.py:16`, `src/training/trainers/mlp_router.py:68`, `src/router/nirt/baselines_infer.py:19-31,101-106`, `src/router/routing/routers.py:89-101`

```python
16:from router.nirt.baselines_infer import _build_mlp_router, _train_correctness_matrix
68:    from router.routing.routers import MLPRouter, _load_training_data
```

**What breaks:** The allowed training→router direction rests on underscore-private names. A router-side change to `_build_mlp_router` alters the checkpoint architecture used by both fit and inference, with no public contract, and import-linter can't flag it. `evaluation/nirt/ood.py:23` does the same with `training.models.profiles._family_of` (XD-03).
**Evidence:** Grep: none of these names is in an `__all__`.
**Direction:** Promote them to public names.

---

### XA-07: The documented recursion guard (`RoutingMode.MODEL_SELECTION` via `RouterTool`) isn't what the live tools do (A6-X02)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/router/tools/router_tool.py:1-14`, `src/router/tools/router_tool.py:39-52`, `src/router/context.py:30-36`, `src/router/agentic/orchestrator.py:45-79`, `src/router/agentic/router.py:121-133`

```python
71:        sc = agent._solve(query, depth + 1)
72:        steps.append(sc)
73:        return sc.answer
```

**What breaks:** `router_tool.py` and `context.py` state that tool-originated routing must never recurse into full agentic dispatch. The `route_and_answer` tool, which the default system prompt tells the agent to prefer, calls `agent._solve`. That re-triages and can build a fresh `AgentExecutor` at every level until `max_depth`. Anyone relying on the documented invariant underestimates fan-out and cost (RA-17).
**Evidence:** Traced `route_and_answer` → `AgenticRouter._solve` → `orchestrator.run` → `build_router_tools`. Grep finds `MODEL_SELECTION` only in `context.py` and `router_tool.py`.
**Direction:** Pass a mode flag into `_solve` so that tool-initiated calls can't orchestrate, or correct the documented invariant.

---

### XA-08: Depth and budget limiting are implemented twice, and `max_depth` means different things (A6-X03, A8-X3)
**Category:** correctness · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/router/agentic/router.py:58,121-140`, `src/router/agentic/orchestrator.py:29-39`, `src/router/execution/orchestrator.py:42-92`, `src/router/execution/budget.py:21`

```python
127:        if decision.mode == "single" or depth >= self.max_depth:     # agentic/router.py
76:            child.depth < self.limits.max_depth                        # execution/orchestrator.py
```

**What breaks:** The two implementations differ on budget and on depth:
- **Budget:** the live `router.agentic` path has only a bare `max_depth` and no spend cap. The budget check exists only in the unwired `router.execution` design, whose `can_spawn` also checks `ledger.remaining()`.
- **Depth:** both default `max_depth=2`. `AgenticRouter` allows nodes at depth 2. `Orchestrator.can_spawn` stops at depth 1.

Porting the live flow onto `router.execution` with the same config value silently makes trees one level shallower.
**Evidence:** Traced both. `execution/orchestrator.py:4-11` acknowledges that they diverge.
**Direction:** Pick one convention (the depth of the deepest allowed node), pin it with a test, and route the live path through a shared `BudgetLedger`.

---

### XA-09: `RoutingPipeline` and `AgenticRouter` route a live prompt with different cost handling (A6-X05)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/router/agentic/router.py:77-92`, `src/router/agentic/triage.py:124-130`, `src/router/router.py:66-105`

```python
87:        return self.router.route_text([prompt], encoder=self.encoder, lam=self.lam)
```

**What breaks:** `RoutingPipeline.route` calls `route_text` with `lam=0`, so cost is applied once, by the policy. `AgenticRouter.route_decision` applies `lam` inside the router and then uses the cost-adjusted pick as its difficulty signal (RA-06). The two prompt-to-model entry points produce different selections for the same `Router` and objective.
**Evidence:** Read both call sites.
**Direction:** Factor one helper that turns a prompt into (scores row, best-quality pick, cost-aware pick), and use it from both entry points.

---

### XA-10: Provider vocabularies differ between `LLMClientFactory`, `guess_provider` and the registry (A8-X2)
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/router/llm/client.py:54-68`, `src/router/agentic/llm_clients.py:130-145`, `configs/model_registry.yaml:105-133`

```python
54:    _PROVIDERS = {"openai", "anthropic", "google"}
133:    ("gemini", "google_genai"), ("palm", "google_genai"), ("bison", "google_genai"),
```

**What breaks:** Three components use three provider vocabularies:
- **`guess_provider`:** LangChain names (`google_genai`, `mistralai`, `groq`).
- **`LLMClientFactory`:** `openai` / `anthropic` / `google` only.
- **Registry YAML:** model-author names (`google`, `meta`, `zhipu`).

An `LLMProfile` built from `guess_provider` raises `ValueError` for every Gemini, Mistral, Llama, Cohere and Qwen model.
**Evidence:** Compared the three lists. No code maps between them.
**Direction:** Define one canonical provider mapping in `router`.

---

### XA-11: Pipeline stages silently depend on stale outputs of earlier runs
**Category:** design · **Severity:** low · **Confidence:** confirmed
**Spans:** `src/training/data/loaders.py:532-535`, `src/training/data/loaders.py:595-596`, `src/training/retrieval/query_bank.py:36-39`, `src/training/taxonomy/clustering.py:129-130`, `src/training/models/profiles.py:291-295`, `src/router/embeddings/encoder.py:439-443`

```python
532:    queries_path = cfg.path("processed") / "queries.parquet"     # loaders.py: reads the previous build
596:            warnings.warn(f"skipping source '{s}': {exc}")        # a missing input is only a warning
130:    col = "nirt_observations.parquet" if p.exists() else "queries.parquet"
```

**What breaks:** Several stages read whatever an earlier run left on disk and have a silent fallback when it's missing. Results therefore depend on build order and history:
- **Anchor judge:** skipped on a fresh build, included on the next, and it reads query text from the previous `queries.parquet` (TD-01).
- **Bank and clusters:** they widen to all queries when observations aren't built yet (TRV-03, TX-02).
- **Profiles and stores:** they never refresh after inputs change (TM-01, RE-01, RE-02).

None of these stages records its inputs in a manifest that a later stage checks.
**Evidence:** The spans above. The `phase0` pipeline (`training/phase0.py`) runs stages in sequence but doesn't pass freshness information between them.
**Direction:** Have each artefact manifest record the fingerprints of its inputs (source files, upstream manifests), and check them on load. Turn "input missing" fallbacks into errors, with explicit opt-outs.
