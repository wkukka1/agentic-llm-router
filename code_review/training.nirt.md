# Code review: training.nirt

**Files reviewed:** 4 (1065 lines) · **Context-only:** `src/router/nirt/model.py`, `src/router/data/nirt.py`
**Summary:** 17 findings: 12 correctness, 2 design, 3 performance · 14 confirmed, 3 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TN-01 | `src/training/nirt/train.py:115-144` | correctness | medium | confirmed |
| TN-02 | `src/training/nirt/train.py:376-379` | correctness | medium | confirmed |
| TN-03 | `src/training/nirt/train.py:543-557` | correctness | medium | confirmed |
| TN-04 | `src/training/nirt/train.py:561-564` | correctness | medium | confirmed |
| TN-05 | `src/training/nirt/coldstart.py:68-73` | correctness | medium | confirmed |
| TN-06 | `src/training/nirt/train.py:504-517` | design | medium | suspected |
| TN-07 | `src/training/nirt/train.py:442-454` | correctness | low | confirmed |
| TN-08 | `src/training/nirt/train.py:492-493` | correctness | low | confirmed |
| TN-09 | `src/training/nirt/train.py:213-257` | performance | low | confirmed |
| TN-10 | `src/training/nirt/train.py:280-298` | correctness | low | confirmed |
| TN-11 | `src/training/nirt/train.py:519-561` | correctness | low | confirmed |
| TN-13 | `src/training/nirt/train.py:613-630` | correctness | low | confirmed |
| TN-14 | `src/training/nirt/train.py:390-462` | performance | low | confirmed |
| TN-15 | `src/training/nirt/metrics.py:97-131` | design | low | confirmed |
| TN-16 | `src/training/nirt/metrics.py:148-155` | performance | low | confirmed |
| TN-12 | `src/training/nirt/train.py:381-387` | correctness | low | suspected |
| TN-17 | `src/training/nirt/coldstart.py:68-75` | correctness | low | suspected |

---

## src/training/nirt/train.py

### TN-01: `loss: pairwise` / `listwise` crash in `backward()` when a batch has no discordant pair `:115-144`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
122:        m = torch.triu(torch.ones(k, k, dtype=torch.bool, device=logit.device), 1) & (dY != 0)
123:        if not m.any():
124:            return torch.zeros((), dtype=logit.dtype, device=logit.device)
```

**What breaks:** When no pair in a batch has different targets, `_pairwise_within_group` returns a fresh `torch.zeros(())` with no grad. The fast path does this at `:123-124`. The slow path does too, because `total` is never added to when no block has `mm.any()` (`:129-144`). The loss `loss_fn(...)` at `:491` then has no `grad_fn`, so `loss.backward()` at `:495` raises `RuntimeError: element 0 of tensors does not require grad`. The realistic trigger is the final grouped batch: `_grouped_batches` yields `G mod batch_queries` leftover queries (`:222-224`). If those few queries are all zero-variance, which is common on RouterBench where many queries are solved by every model, the batch has no discordant pair. `_listwise_within_group` has the same failure on its slow path (`:159-166`) when every group has size < 2. `bce_pairwise` is safe because its BCE term carries grad. `theta_cov_weight > 0` also masks the crash.
**Evidence:** Traced `fit` → `_loss_fn("pairwise")` (`:201-203`) → `_pairwise_within_group` → `loss.backward()` (`:495`). No `requires_grad` guard exists between them. Tests `test_routing_aware_loss_learns_ranking` and `test_pairwise_weighting_default_none_unchanged` (`tests/router/nirt/test_nirt_train.py:192-204,314-322`) use synthetic Bernoulli worlds with 6 models and 64-128 queries per batch. Every batch there contains a discordant pair, so the zero-pair case is never exercised.
**Direction:** Return `logit.sum() * 0.0` (graph-connected) instead of a detached zero, or skip `backward`/`step` when `loss.requires_grad` is False.

---

### TN-02: Preference auxiliary feeds `e_q` without the `data.query_features` concat, so the first battle batch crashes `:376-379`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
376:        pairwise_arrays = build_pairwise_arrays(
377:            d, source=pcfg.get("source"), split="train", pathway=pathway,
378:            query_pathway=pcfg.get("query_pathway") or None,
```

**What breaks:** Setting `data.query_features: default` together with `train.preference.enabled: true` sizes the model with `query_dim = train_ds.query_dim`, which is `D_emb + D_f` (`:393`; `router/data/nirt.py:102-107`). The battle arrays, however, hold raw `q_store` rows of width `D_emb` (`pairwise.py:77`). `pairwise_loss` → `model.latent_query(e_q)` (`pairwise.py:95`) therefore calls a `Linear(D_emb+D_f, …)` on `[N, D_emb]`, and the call raises a shape-mismatch RuntimeError at the first auxiliary step (`:510`). The run fails only after a full correctness epoch has already been trained.
**Evidence:** `build_pairwise_arrays` never receives `query_features` and has no feature store parameter (`pairwise.py:45-52`). No test combines the two options. `test_fit_with_concatenated_query_features` (`test_nirt_train.py:216-235`) has preference off, and `test_preference_enabled_trains_with_explicit_arrays` (`:252-263`) has no features.
**Direction:** Pass `query_features` through and concatenate the feature rows, zero-filled for battle queries without features, just as `NIRTDataset` does. At minimum, assert `pairwise_arrays.e_q.shape[1] == model.query_dim` before training starts.

---

### TN-03: Collapse alarm fires on every epoch for `dim: 1` (the nirt.yaml default); with `abort_on_collapse` it stops training after 3 epochs `:543-557`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
544:            is_collapsed = (
545:                collapse_diag.get("theta_effective_rank", float("nan")) < collapse_rank_threshold
546:                or collapse_diag.get("bias_argmax_agreement", 0.0) > collapse_agreement_threshold
```

**What breaks:** `effective_rank` of a `[N, 1]` θ matrix is `λ²/λ² = 1.0` by definition (`metrics.py:173-178`). That is always below the default threshold 1.3 (`:426`). Every `query_latent` run with `model.dim: 1` (`configs/nirt.yaml:33`) therefore prints "possible routing collapse" every epoch. With `abort_on_collapse: true` it breaks at epoch `collapse_patience - 1` (epoch 2 by default), whatever the model is actually doing. The `break` at `:557` also happens before the score/best-state update at `:561-567`, so the abort epoch's validation score is never considered. With `collapse_patience: 1`, `best_state` stays `None` and `best_epoch` stays `-1`.
**Evidence:** Traced `_collapse_diagnostics` (`:288`) → `effective_rank`. `test_abort_on_collapse_stops_training` (`test_nirt_train.py:376-385`) uses `dim=2` with an artificial threshold of 1000, and no test covers `dim=1` with the alarm.
**Direction:** Skip the rank criterion when `K == 1`, or express the threshold relative to K. Run the best-state check before honouring the abort `break`.

---

### TN-04: `val_metric` is always minimised, silently falls back to BCE for unknown keys, and never improves on NaN `:561-564`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
561:        score = vm.get(val_key, vm["bce"])
562:        if sched is not None:
563:            sched.step(score) if lr_schedule == "plateau" else sched.step()
564:        if score < best_val - 1e-5:
```

**What breaks:**
- `val_key` is read unvalidated (`:409`), and `prediction_metrics` also returns higher-is-better keys: `auc`, `spearman_r`, `pearson_r`, `accuracy`, `acc@0.5` (`metrics.py:108-123`). With `val_metric: auc`, early stopping keeps the epoch with the lowest AUC, and `ReduceLROnPlateau(mode="min")` (`:434-436`) cuts the LR while AUC is rising.
- A typo such as `val_metric: regrett` silently trains on BCE.
- `spearman_r`/`pearson_r` are NaN when predictions are constant (`metrics.py:21-22,27-28`). `NaN < best_val` is False, so no epoch ever improves: `best_state` stays `None` and the run returns the last-epoch weights with `best_epoch = -1`.

**Evidence:** The contract map says `val_metric` accepts "any metric key, falls back to bce". The yaml comment lists only `{bce, mse, regret}` (`configs/nirt.yaml:92`), but nothing enforces that list. Tests only use `bce` and `regret` (`tests/helpers/nirt.py:144`, `test_nirt_train.py:200,210`).
**Direction:** `require_choice(val_key, ("bce","mse","brier","mae","log_loss","ece","regret"))`, or negate the higher-is-better keys. Treat a NaN score as "no improvement" explicitly.

---

### TN-06: The preference weight is not a loss weight: auxiliary battles take separate Adam steps after the correctness pass `:504-517`
**Category:** design · **Severity:** medium · **Confidence:** suspected

```python
510:                loss = pairwise_loss(model, pw["e_q"][idx], pw["e_a"][idx], pw["e_b"][idx], pw["y"][idx])
511:                opt.zero_grad()
512:                (pw_weight * loss).backward()
```

**What breaks:** `configs/nirt.yaml:116` documents `preference.weight` as the "aux-loss weight added to the correctness loss". The code does something else. After the full correctness epoch it runs a separate block of about `N_battles/4096` consecutive `opt.step()` calls on the battle loss alone. The model is then validated immediately after that block (`:519`). Adam divides each update by `sqrt(v)`. During a run of consecutive auxiliary steps, the first moment is dominated by the scaled battle gradient, so multiplying the loss by `weight` changes the step size only through the partly shared `v` history. That is far weaker than a true loss weight. Two further side effects follow: each auxiliary step applies `weight_decay` again, and the alternating blocks mean the epoch-end checkpoint is biased toward the battle objective. The `train.preference.weight` sweep referenced in `scripts/nirt/anchor_judge_analysis.py:268-276` may have measured less than intended.
**Evidence:** Read `:504-517`. There is one shared `opt` (`:429`). Adam's approximate scale invariance is a known property, but its actual size here depends on how the two gradient populations mix in `v`, which I have not measured. Hence `suspected`.
**Direction:** Draw a battle mini-batch inside each correctness step and backprop `loss + pw_weight * pw_loss` in one `opt.step()`, so `weight` is a genuine relative weight.

---

### TN-07: Preference enabled with zero joinable battles silently trains without the auxiliary loss `:442-454`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
443:    if pairwise_on and pairwise_arrays is not None and len(pairwise_arrays) > 0:
```

**What breaks:** When `build_pairwise_arrays` drops every battle because no query or profile embeddings exist, `pw` stays `None`. The run then proceeds as a plain correctness run with no log line, since the only print sits inside the `if` (`:452-454`). The run's `config` still records `preference.enabled: true`. A concrete trigger with phase0-built stores is described in TR-01 (`training.md`): the `irt` bert query store holds only correctness queries.
**Evidence:** `pairwise.py:72-76` returns an empty `PairwiseArrays`, and `n_dropped` is never surfaced when the arrays are empty.
**Direction:** Raise, or at least warn with `n_dropped`, when preference is enabled and zero battles survive the join.

---

### TN-08: `theta_cov_weight` runs a second query-head forward per step (double BatchNorm update, different dropout mask) `:492-493`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
492:            if theta_cov_weight > 0 and hasattr(model, "latent_query"):
493:                loss = loss + theta_cov_weight * _theta_cov_penalty(model.latent_query(qt[idx]))
```

**What breaks:** `model(qt[idx], ref)` at `:486` has already computed θ through `latent_query` (`model.py:316`). Calling it again doubles the query-head cost. With `query_head_batchnorm: true`, `BatchNorm1d` running statistics get two momentum updates per step (`model.py:189-190,256-257`), so the eval-mode statistics drift faster than configured. With `query_head_dropout > 0`, the penalty is computed on a different dropout sample than the one the loss used.
**Evidence:** `NIRTModel.forward` does not expose θ, so the second call is the only way the trainer can get it. Tests (`test_nirt_train.py:305-311`) check only finiteness.
**Direction:** Compute θ once: call `latent_query` and `model_parameters` explicitly, or add a forward variant that returns θ.

---

### TN-09: Per-group helpers scan the full row array once per group (O(G·N)), and `_grouped_batches` repeats it every epoch `:213-257`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
219:    g_ids = np.unique(qgroup)
220:    order = torch.randperm(len(g_ids), generator=gen).numpy()
221:    rows_of = {g: np.where(qgroup == g)[0] for g in g_ids}
```

**What breaks:** `_grouped_batches` rebuilds `rows_of` with one full `qgroup == g` scan per query, every epoch. `_query_variance_mask` (`:253-256`) and `_val_regret` (`:234-236`) follow the same pattern. `_val_regret` runs every epoch under `val_metric: regret` and always once at the end (`:583`). At RouterBench scale, roughly tens of thousands of train queries × about 10 models each (rough estimate), that is on the order of 10^9–10^10 element comparisons per epoch of pure Python-driven bookkeeping.
**Evidence:** Read the code. `qgroup` is a dense 0..G-1 inverse from `np.unique` (`:66`), so a single `argsort` plus split is enough.
**Direction:** Compute `order = np.argsort(qgroup, kind="stable")` and `np.split(order, np.cumsum(np.bincount(qgroup))[:-1])` once in `fit`, and reuse them. Use `np.maximum.reduceat`/`bincount` for the variance mask and regret.

---

### TN-10: `bias_argmax_agreement` ignores the interaction residual and weights queries by model coverage `:280-298`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
289:    if (getattr(model, "orientation", None) == "query_latent"
290:            and getattr(model, "model_params", None) == "free"
291:            and getattr(model, "difficulty", None) == "scalar"):
```

**What breaks:** The "full router" logit at `:296` is `θ·a − b`. With `model.interaction: true` the real logit also adds `γ·MLP([θ,a,θa])` (`model.py:322-324`), and the gate above does not exclude interaction. Once γ moves off 0, the collapse agreement is measured on a different router than the one being trained, and that number can abort training (TN-03). Separately, `theta_val` is computed per observation row (`:531`), so each query is repeated once per scored model. Both `theta_effective_rank` and the agreement rate are therefore coverage-weighted rather than per-query, and the head runs on every row each epoch.
**Evidence:** Traced `:529-533` → `_collapse_diagnostics`. `test_collapse_diagnostics_logged_for_query_latent` (`test_nirt_train.py:359-366`) uses `interaction` off.
**Direction:** Add `not model.interaction` to the gate, or score through `model.forward`. Deduplicate `va["q"]` by `qgroup` before calling `latent_query`.

---

### TN-11: An empty validation split crashes with `KeyError: 'bce'` `:519-561`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
520:        vm = prediction_metrics(va["y"], val_prob)
```

**What breaks:** `prediction_metrics` returns `{"n": 0}` for empty input (`metrics.py:100-102`). The next reads of `vm['bce']` at `:538` (verbose print) and `vm["bce"]` at `:561` raise `KeyError`. `split.validation_fraction` defaults to 0 (`training/data/splits.py:151`). `configs/phase0.yaml:211` sets 0.1, so this only hits configs that omit the key, or callers passing an empty `datasets=` tuple.
**Evidence:** Traced `_pack` on an empty dataset → `_predict_packed` returns an empty array → `prediction_metrics`.
**Direction:** Check `len(va["y"]) > 0` up front and raise a clear error, or skip validation-based early stopping.

---

### TN-12: Validation-only model ids get untrained rows in `free` mode and are scored and saved `:381-387`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
381:    all_ids = sorted(set(map(str, train_ds.model_ids)) | set(map(str, val_ds.model_ids)))
382:    model_index = {m: i for i, m in enumerate(all_ids)}
```

**What breaks:** A model that appears in validation but not train gets an `nn.Embedding` row that never receives gradient: `a ~ N(0, 0.1)`, `b = 0` (`model.py:202-205`). Its predictions still count toward `val_metrics` and early stopping, and the row is checkpointed into `n_models`/`model_index`. `NIRTRouter` then offers it as a candidate (`routers.py:184`).
**Evidence:** Only train rows are checked for `midx < 0` (`:386`). `build_nirt_observations(models="warm")` normally yields the same model set in every split, so whether this is reachable is unconfirmed. **Maintainer question:** can a warm model have observations only in validation or test queries (e.g. after dedup)?
**Direction:** Build `model_index` from train ids only, and drop or flag validation rows for unknown models.

---

### TN-13: `run.json` is written with bare `NaN` tokens (invalid strict JSON) `:613-630`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
613:        (out / "run.json").write_text(
614:            json.dumps(
```

**What breaks:** `history` and `val_metrics` routinely contain `float("nan")`: `spearman_r`/`pearson_r`/`auc` on degenerate predictions, and `theta_effective_rank` (`metrics.py:21-22,41-42,170-177`). `json.dumps` defaults to `allow_nan=True` and emits `NaN`, which `jq`, `JSON.parse`, and other strict parsers reject.
**Evidence:** No Python reader of `run.json` exists (contracts §1), so nothing in-repo breaks today, only external tooling.
**Direction:** Map NaN to `None` before dumping, or pass `allow_nan=False` behind a sanitiser.

---

### TN-14: `fit` is CPU-only and has no device option `:390-462`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
456:    qt = torch.from_numpy(tr["q"])
457:    mt = torch.from_numpy(tr["m"])
```

**What breaks:** The model and every tensor stay on CPU, and no `train.device` key exists. The phase1/2 baseline trainer does expose `train.device` (contracts §5), so the two trainers are inconsistent, and NIRT sweeps (`knn_impute_sweep`, `complexity_search`, `ab_orientation`) cannot use an available GPU.
**Evidence:** No `.to(` or `device` anywhere in `train.py`. `_collapse_diagnostics` (`:296-297`) and `_predict_packed` (`:312`) call `.numpy()` directly, which would also need `.cpu()` once a device is added.
**Direction:** Add `train.device`, move the model and packed tensors once, and `.cpu()` before any numpy conversion.

---

## src/training/nirt/metrics.py

### TN-15: BCE, Brier, and accuracy are each implemented two or three times `:97-131`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
104:    p = np.clip(y_prob, _EPS, 1 - _EPS)
105:    bce = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())
```

**What breaks:** The BCE formula exists three times: `log_loss` (`:63-67`), inline in `prediction_metrics` (`:104-105`), and `_bce_mse` (`:126-131`). Brier exists twice (`brier_score` `:56-60` and `:107`). `accuracy` and `acc@0.5` compute the same expression twice (`:115,120`). A change to clipping or eps in one copy would make `prediction_metrics["bce"]` disagree with `log_loss`, which `baseline/calibration.py` uses.
**Evidence:** Grep `from training.nirt.metrics import`: `brier_score`/`log_loss`/`reliability_curve` are used by `training/nirt/baseline/calibration.py:15`, and `prediction_metrics` by more than 15 modules and scripts. All copies are live.
**Direction:** Have `prediction_metrics` and `_bce_mse` call `log_loss`/`brier_score`, and compute accuracy once.

---

### TN-16: `per_group_metrics` builds its small-group mask with a Python loop over every row `:148-155`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
149:    counts = {lbl: int((g == lbl).sum()) for lbl in np.unique(g)}
150:    small = np.array([counts[lbl] < min_n for lbl in g])
```

**What breaks:** `:150` is a per-row Python loop, and `:149`/`:154` add a full scan per label. Both are slow on the full observation table used by `scripts/nirt/capacity_diagnostics.py:46`.
**Evidence:** Read the code.
**Direction:** Use `labels, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)` and `small = cnt[inv] < min_n`.

---

## src/training/nirt/pairwise.py

No findings in this doc. See `_cross_cutting_raw/A2.md`: centering inconsistency in `pairwise_logit_diff` (medium) and duplication of `NIRTModel.forward`. TN-02 covers the missing feature concatenation from the `train.py` side.

---

## src/training/nirt/coldstart.py

### TN-05: Prior strength does not shrink as the probe set grows (mean NLL + summed ridge) `:68-73`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
71:        nll = torch.nn.functional.binary_cross_entropy_with_logits(logit, y_t)
72:        reg = ridge * ((a - pa).pow(2).sum() + (b - pb).pow(2))
73:        (nll + reg).backward()
```

**What breaks:** `binary_cross_entropy_with_logits` defaults to `reduction="mean"`, so the data term is an average over the `k` probes while the prior term is not divided by `k`. That is equivalent to a Gaussian prior whose precision grows linearly with `k`. As `k → ∞` the minimiser tends to `argmin E[nll] + ridge‖Δ‖²`, a fixed shrunk point, not the MLE. The docstring's claim "as `k` grows … converges toward the pure-probe MLE" (`:48-49`) only holds as `ridge → 0`. `lomo_eval` uses the default `ridge=1.0` (`evaluation/nirt/coldstart_eval.py:34,72-73`), so its learning curve is strongly shrunk toward the prior at every `k` and understates few-shot cold start.
**Evidence:** Derived from the objective. Tests miss it: `test_fit_probe_recovers_true_params…` uses `ridge=1e-3` (`tests/training/nirt/test_nirt_coldstart_fewshot.py:38-39`), and `test_fit_probe_more_data_moves_closer…` (`:55-71`) compares k=10 with k=300 at `ridge=0.1`, where the gap comes from estimation noise and does not test convergence.
**Direction:** Use `reduction="sum"`, or divide `ridge` by `k`, so the prior is a fixed-strength MAP prior.

---

### TN-17: The probe score function differs from `NIRTModel`'s logit `:68-75`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
70:        logit = theta_t @ a - b
```

**What breaks:** `NIRTModel` applies `softplus(a)` when `constrain_discrimination` is on (the default at `dim: 1`), mean-centres `a` under `center_discrimination`, supports `difficulty: vector`, and adds an interaction residual (`model.py:302-325`). `fit_probe` fits and returns an unconstrained `a` in plain `θ·a − b` form. If `prior_a` comes from `a_head` before softplus, or the result is plugged back into a constrained or centred model, predictions will not match.
**Evidence:** The only caller is `evaluation/nirt/coldstart_eval.lomo_eval`, which also scores with the plain form (`coldstart_eval.py:75`), so it is internally consistent. Grep of `scripts/` finds no `lomo_eval`/`prior_a` producer. **Maintainer question:** are priors meant to be post-softplus, post-centring `a_m` from `model.model_parameters`, and is `fit_probe` intended only for scalar-difficulty, non-interaction models?
**Direction:** Document or enforce the supported model configuration (raise on vector difficulty or interaction), or score through the model's own parameterisation.

---

## Checked and ruled out
- `_grouped_batches` → `_group_blocks`: each chunk holds distinct group ids with their rows concatenated contiguously, so boundary detection and `view(G, k)` are valid.
- `_weighted_bce`/`mse` with `weight=None` reproduce the unweighted mean. `hard_bce` binarises only the target.
- `_pack`: `np.unique(..., return_inverse=True)` gives a dense qgroup. The `midx < 0` check on train only is sufficient because the index is built from train ∪ val.
- `_theta_cov_penalty`: guards K<2 and B<2, and normalises by K(K−1). The `/N` vs `/(N−1)` difference from `effective_rank` does not matter because the participation ratio is scale-invariant.
- `_collapse_diagnostics` with `center_discrimination`: subtracting a shared vector does not change the per-query argmax.
- `build_pairwise_arrays`: `Series.isin(dict)` iterates keys, so the membership test against `_index` is correct.
- `_resolve_runs_dir` calls `Config.resolve`, which exists (`router/config.py:70-72`).
- `listwise` on zero-variance queries pushes logits toward uniform: this is intended ListNet behaviour.
- `marginal_baselines`: models seen only in the eval set fall back to the global mean, as intended.
- `reliability_curve`: the `np.digitize` on interior edges plus clip covers p=1.0 in the last bin.
