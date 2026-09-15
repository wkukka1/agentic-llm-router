# Cross-package interface contracts (src/)

Factual map for the reviewers. No severity judgements. Line refs are `path:line` against the working tree at review time.
Conventions: `B` = batch rows, `N` = observations, `K` = latent dim (`model.dim`), `D_q` = query dim (includes concatenated feature dim when `data.query_features` is set), `D_m` = profile-embedding dim, `C` = relevance dim, `Q`/`M` = #queries / #models.

---

## 1. Checkpoint formats

There are **two unrelated `model.pt` formats with the same filename**, and **two functions both named `load_run`** that have different signatures and return tuples.

| | NIRT / IRT-Router run | Phase 1/2 baseline (Bernoulli/Normal/Beta/ZOIB) |
|---|---|---|
| Writer | `training/nirt/train.py:602-630` (`fit`, `save=True`) | `training/nirt/baseline/checkpoint.py:27-53` (`save_checkpoint`), called from `training/nirt/baseline/train.py:210-215` |
| Location | `<runs_dir>/<name>/model.pt` + `run.json`. `runs_dir` defaults to `nirt_cfg["runs_dir"]` or `DEFAULT_NIRT_RUNS_DIR="data/processed/nirt_runs"` (`router/config.py:21`, `train.py:638-647`). Default name is `f"{irt|nirt}-{dim}d-{model_params}"` (`train.py:585-586`) | `out_dir` or `cfg["out_dir"]` or `"artifacts/phase1/baseline"` (`baseline/train.py:29,208-209`). Files: `model.pt`, `config.yaml`, `metrics.json`, `training_history.json`, `provenance.json` |
| Loader | `router/nirt/checkpoint.py:19-38` `load_run(name, runs_dir=None) -> (model, config, model_index)` | `baseline/checkpoint.py:56-66` `load_checkpoint(dir) -> (model, blob)`; `:69-82` `load_run(dir) -> (model, blob, settings)` |
| Model builder | `router.nirt.model.build_model` (`model.py:432-440`) | `baseline/model.py:159-162` `build_baseline_model` |

### model.pt keys: writer vs reader

| Key | NIRT writer (`train.py`) | NIRT reader (`router/nirt/checkpoint.py`) | Baseline writer (`baseline/checkpoint.py`) | Baseline reader (`baseline/checkpoint.py`) |
|---|---|---|---|---|
| `state_dict` | `:604` | `:36` | `:38` | `:64` |
| `config` (full nirt cfg dict) | `:605` | `:31` (`.get("model", {})`), `:38` returned | — | — |
| `model_cfg` (the `model` subsection) | — | — | `:40` (`config.get("model", {})`; train bakes `ablation`, `response_model`, `response_cfg` into it first, `baseline/train.py:115-119`) | `:62` |
| `n_models` | `:606` `len(model_index)` | `:32` | — | derived from `len(blob["model_index"])` `:62` |
| `model_index` `{model_id: row}` | `:607` | `:38` returned | `:41` | `:62`; also `blob["model_index"]` in `baseline/data.py:236`, `baseline/eval.py:93,179`, `continuous_eval.py:96`, `scripts/route_compare.py:145`, `evaluation/pool_expansion/battery.py:536`, `scripts/nirt/baseline/inspect_baseline.py:37` |
| `query_dim` | `:608` `train_ds.query_dim` (includes feature concat) | `:33` | `:42` | `:62` |
| `profile_dim` | `:609` `train_ds.model_dim` | `:34` | `:42` (`profile_dim or query_dim`, `baseline/train.py:212`) | `:63` |
| `relevance_dim` | — | — | `:43` | `:63` |
| `arch` | — | — | `:39` (`getattr(model,"arch","baseline")`) | not read anywhere (grep `blob[`) |
| `dim` | — | — | `:44` | not read anywhere |
| `model_params` | — | — | `:44` | not read anywhere |

### Side files

| File | Writer | Reader |
|---|---|---|
| `run.json` keys `name, git_sha, elapsed_sec, config, n_models, model_index, best_epoch, history, val_metrics, baselines` | `train.py:613-630` | no Python reader in src/ or scripts/ (only mentioned in docstrings: `router/tracing/traces.py:5`, `router/models/artifacts.py:4`) |
| `config.yaml` (full phase1/2 cfg) | `baseline/checkpoint.py:46` | `baseline/checkpoint.py:74-81` reads `data.pathway` (default `"irt"`), `response.binary_threshold` (0.5), `response.score_kind` (`"effective"`) |
| `metrics.json` = `{validation, best_epoch, train_imbalance}` | `baseline/checkpoint.py:47-49`, payload `baseline/train.py:213-214` | not read by the checkpoint loaders |
| `training_history.json` | `baseline/checkpoint.py:47-49` | not read by the checkpoint loaders |
| `provenance.json` = `{git_sha, seed, phase1_config, phase0_artifacts{…sha1}, taxonomy_version, dataset_stats}` | `baseline/checkpoint.py:50-52`, `baseline/train.py:240-267` | not read by the checkpoint loaders |

### Consumers of each `load_run`
- `router.nirt.checkpoint.load_run`: `router/routing/routers.py:198-210` (`NIRTRouter.from_run`), `evaluation/pool_expansion/battery.py:76-78`, `scripts/route_compare.py:46-49`, `scripts/nirt/{ab_orientation,capacity_diagnostics,compare,coldstart,knn_impute_sweep,eval_nirt,complexity_search,train_test_gap,shrinkage_eval,route_eval,theta_collapse}.py`. It is re-exported from `router/nirt/__init__.py:26`.
- `training.nirt.baseline.checkpoint.load_run`: `baseline/data.py:230-232` (`checkpoint_matrix`), `baseline/eval.py:92`, `baseline/continuous_eval.py:95,215`, `evaluation/pool_expansion/battery.py:33,535`, `scripts/route_compare.py:30,144`, `scripts/nirt/baseline/{inspect_baseline,evaluate}.py`.
- `NIRTRouter.from_run` reads the stored config's `data.pathway` (default `"irt"`), `data.query_pathway`, and `data.query_features` (`routers.py:201-208`).

### Mismatches observed
- Baseline `model.pt` keys `arch`, `dim`, `model_params` are written (`baseline/checkpoint.py:39,44`) and never read.
- `run.json` is written (`train.py:613`) and no code reads it.
- The two formats are not interchangeable. `router.nirt.checkpoint.load_run` requires `config` and `n_models`, which the baseline format lacks. `baseline.checkpoint.load_checkpoint` requires `model_cfg` and `relevance_dim`, which the NIRT format lacks.
- The two `load_run` functions differ in signature. The router version is `(name, runs_dir)` and joins `runs_dir/name`. The baseline version is `(directory)`. Their 2nd and 3rd return elements also differ: `(config, model_index)` versus `(blob, settings)`. `route_compare.py` and `battery.py` import both and alias one as `nirt_load_run`.
- NIRT `fit` stores `nirt_cfg` after CLI mutation. `scripts/nirt/train_nirt.py:88` injects `data.ood_holdout_families` into it. `fit` never reads that key (`train.py:338-341`), and `evaluation/nirt/ood.py:29-30` reads `evaluation.ood_holdout_families` instead.

---

## 2. NIRTDataset batch contract (`src/router/data/nirt.py`)

Constructor `NIRTDataset(observations, query_store, profile_store, *, feature_store=None, return_ids=False)` (`:56-64`).
- Rows are dropped unless `query_id in query_store._index`, `model_id in profile_store._index`, and `target` is not null (`:70-76`). The count is kept in `self.dropped`. It reads the **private** `EmbeddingStore._index` (`router/embeddings/encoder.py:267`).
- Required obs columns: `query_id, model_id, target, cost, metric_type, source` (`:70-83`). The full schema `NIRT_OBS_COLUMNS` is `query_id, model_id, target, score_raw, metric_type, source, split, cost, is_multiple_choice, n_choices` (`training/data/nirt.py:27-30`).
- When a feature store is given, `_feat` is `[N, D_f]` float32. Queries missing from the feature store get zero rows (`:90-96`).
- Properties: `query_dim = D_emb + D_f` (`:102-107`); `model_dim = D_m` (`:109-111`); `targets` `[N]` float32; `query_ids`, `model_ids` are object ndarrays (`:113-123`).

| Access | Key | Shape | dtype |
|---|---|---|---|
| `__getitem__(i)` (`:126-141`) | `query_embedding` | `[D_q]` | float32 numpy (feat concatenated) |
| | `model_embedding` | `[D_m]` | float32 numpy |
| | `target` | scalar | `np.float32` |
| | `metric` | scalar | str (from obs column `metric_type`) |
| | `source` | scalar | str |
| | `cost` | scalar | `np.float32` (NaN if non-numeric) |
| | `query_id`, `model_id` | scalar | only if `return_ids` |
| `gather(index=None)` (`:143-170`) | `query_embedding` | `[N, D_q]` | float32 |
| | `model_embedding` | `[N, D_m]` | float32 |
| | `target` | `[N]` | float32 |
| | `cost` | `[N]` | float32 |
| | `metric`, `source` | `[N]` | str ndarray |
| | (no ids; use `.query_ids`/`.model_ids`) | | |
| `collate(batch)` (static, `:173-192`) | `query_embedding` | `[B, D_q]` | torch.float32 |
| | `model_embedding` | `[B, D_m]` | torch.float32 |
| | `target`, `cost` | `[B]` | torch.float32 |
| | `metric`, `source` | list[str] len B | — |
| | `query_id`, `model_id` | list, if present in items | — |

- `dataloader(batch_size=256, shuffle=True, **kw)` (`:194-198`) has no caller in src/ or scripts/ (grep). `NIRTDataset.from_config(cfg, *, data, split, pathway="irt", query_pathway, query_features, observations, return_ids)` is at `:201-261`. With `observations=None` it reads `cfg.path("processed")/nirt_observations.parquet` and raises if that file is missing. It filters on `split`. The query store comes from `data.query_embeddings(query_pathway or pathway)` and the profile store from `data.profile_embeddings(pathway)`.
- Consumers use `gather()`, never `__getitem__`/`collate`: `training/nirt/train.py:61` (`_pack` → `q,m,y,midx,model_ids,qgroup`), `router/nirt/predict.py:23`, `training/nirt/baseline/data.py:98`.

### Training-side builders (`src/training/data/nirt.py`)
- `build_nirt_observations(cfg, *, data=None, score_kind="effective", metrics=None, models="warm", drop_unsplit=True)` (`:36-88`). `target` is `pd.to_numeric(long["score"])` from `TrainingData.correctness(...)` (`:55-56,69`). `split` maps each query to one of train/validation/test/ood via `d.splits[name]["query_ids"]` (`:58-64`), with the last matching name winning. `cost` is the per-(q,m) mean of `d.responses.cost`, and `source` is `first` (`:78-83`). Unsplit rows are dropped (`:85-86`).
- `write_nirt_observations` and `read_nirt_observations` use `processed/nirt_observations.parquet` (`:91-99`).
- `nirt_dataset_from_config(cfg, *, data, split, pathway, query_pathway, query_features, observations, return_ids, **build_kw)` (`:105-133`) loads `data` if absent, builds the obs table if the parquet is absent, and delegates to `NIRTDataset.from_config`.
- `TrainingData.nirt_dataset(...)` (`facade.py:303-316`) calls `nirt_dataset_from_config`.
- `router.routing.routers._cross_obs` (`routers.py:44-57`) builds a synthetic obs frame over the full Q×M grid with `target=0.0` and `cost=NaN` for serving.

---

## 3. NIRT model forward contract

### `router/nirt/model.py`, the serving/training model used by `training/nirt/train.py`
`build_model(model_cfg, *, n_models, query_dim=768, profile_dim=768)` dispatches on `model_cfg["orientation"]` (default `query_latent`) (`:432-440`).

Shared interface: `forward(e_q, model_ref)` returns a **logit** `[B]`. `predict_proba` is `sigmoid(forward)` under `no_grad` (`:327-331`, `:422-426`). Callers apply `torch.sigmoid(model(...))` themselves (`train.py:312`, `predict.py:47`, `routers.py:279`).
`model_ref` depends on `model_params`:
- `"projected"`: float `[B, D_m]` (profile embeddings).
- `"free"`: long `[B]` (model-index rows). numpy input is coerced (`:298-299`, `:403-404`).

| Class | Attrs exposed | Computation | Shapes |
|---|---|---|---|
| `NIRTModel` (`orientation="query_latent"`, `:128-331`) | `query_dim, dim, n_models, model_params, difficulty, constrain_discrimination, center_discrimination, query_head_batchnorm, interaction, …` | `latent_query(e_q)` (`:253-258`) gives θ `[B,K]` (optional `BatchNorm1d(affine=False)`). `model_parameters(ref)` (`:285-308`) gives `a [B,K]` (softplus if constrained, centered if enabled) and `b` (`[B]` scalar difficulty or `[B,K]` vector). Logit is `Σ a·θ − b` (scalar) or `Σ a(θ−b)` (vector), plus `γ·MLP([θ,a,θa])` if `interaction` (`:310-325`) | in `e_q [B,D_q]`; out `[B]` |
| `IRTRouterModel` (`orientation="model_latent"`, `:334-426`) | `query_dim, dim, n_models, model_params, bound_ability, constrain_discrimination` | `latent_ability(ref)` gives θ_m `[B,K]`: sigmoid(Linear) if projected and `bound_ability`, raw Embedding if free with no bound applied (`:398-405`). `query_parameters(e_q)` gives `a_q [B,K]`, `b_q [B]` (`:407-413`). Logit is `Σ a_q·θ_m − b_q` (`:415-420`) | out `[B]` |

- Query head `_query_head` (`:85-107`): a Linear when `hidden` is falsy; a legacy `Sequential(Linear,ReLU,Linear)` when all P2a knobs are default; otherwise `_MLPHead`. The choice changes state_dict key layout.
- Projected `a_head`/`b_head` use `components.mlp` (`:196-200`). Optional modules `query_bn`, `interaction_net`, `interaction_gamma` exist only when enabled (`:189-190,212-216`).
- `_center_discrimination` (`:260-283`) uses the free pool mean (`self.a.weight`) or, in projected mode, the **current batch mean**.
- Pairwise aux (`training/nirt/pairwise.py:84-117`) calls `model.latent_query` and `model.model_parameters` on profile embeddings, so it requires `NIRTModel`, projected mode. `train.py:397-402` enforces projected.
- `train.py` also uses `model.latent_query` (guarded by `hasattr`, `:492,529`), `model.model_parameters(torch.arange(n_models))` (free+scalar only, `:289-295`), and `model.orientation`, `.dim`, `.model_params` (`:585-586`).

### `router/nirt/components.py`, the building blocks for **`training.nirt.baseline.model.BaselineNIRT`**
The module docstring says `router.nirt.baseline.BaselineNIRT` (`components.py:1`); the class actually lives in `training/nirt/baseline/model.py`.

| Component | Signature → output |
|---|---|
| `mlp(in,out,hidden)` `:21-26` | Linear, or Linear-ReLU-Linear |
| `WarmupBlender(enabled, alpha, learnable)` `:29-53` | `forward(e_q [B,D_q], neighbor_mean [B,D_q] or None)` → `[B,D_q]`; all-zero neighbour rows pass through |
| `DiscriminationHead` `:56-79` | `forward(e_q, r_q [B,C] or None)` → `(a [B,K] softplus-if-constrain × sigmoid(W_r r_q) gate, gate [B,K] or None)`. `w_r` is init weight 0, bias 2.0 |
| `DifficultyHead` `:82-90` | → `b_q [B]` |
| `InteractionLayer` `:93-111` | `forward(a_q, theta_m, b_q, r_q)` → `(logit [B], base_irt [B])`; `γ` init 0; the net input is `3K + C` |
| `LengthHead` `:114-125` | → `[B]`, not connected to the loss |

`BaselineNIRT.forward(e_q, model_ref, r_q=None, neighbor_mean=None)` returns `Output` (`baseline/model.py:37-49,140-151`) with fields:
- `logit [B]`, `base_irt [B]`
- `a_q [B,K]`, `b_q [B]`, `theta_m [B,K]`
- `relevance_gate`, `length_pred`
- `response: ResponseOutput`

The response head receives `features=cat(e', θ_m) [B, D_q+K]`, or `None` for Bernoulli (`:148`). `predict_proba` is `response_head.as_probability(out.response)` (`:153-156`). `Output.proba()` is `sigmoid(logit)` (`:48-49`). θ_m is sigmoid-bounded only in projected mode (`:123-129`). A missing `r_q` with relevance on is replaced by zeros `[B,C]` (`:142-143`).

### Response heads (`training/nirt/baseline/response_head.py`, `continuous_*.py`)
Factory `build_response_head(name, cfg, *, feature_dim, hidden)` (`response_head.py:159-166`) accepts names `bernoulli|normal|beta|zoib` (`:156`). Output is `ResponseOutput(z, params: dict)` (`:36-45`).

| Head | `target` attr | `params` keys | `mean(out)` | `as_probability(out)` |
|---|---|---|---|---|
| Bernoulli `response_head.py:104-133` | `binary` | `logit` (=z), `p`=sigmoid(z) | `p` | `p` (probability) |
| Normal `continuous_normal.py:29-43` | `soft` | `mu` (= **z, the raw latent, not sigmoided**), `sigma`=softplus+eps | `mu` (=z) | base `mean().clamp(0,1)` (`response_head.py:75-76`), i.e. z clamped |
| Beta `continuous_beta.py:34-55` | `soft`, `interior_only=True` | `mu`=sigmoid(z) clamped, `kappa`, `alpha`, `beta` | `mu` | clamp(mean) |
| ZOIB `continuous_zoib.py:36-92` | `soft` | `mu, kappa, alpha, beta, log_pi0, log_pi1, log_pic, pi0, pi1, pic` | `pi1 + pic·mu` | clamp(mean) |

Other head methods: `nll(y,out)` gives `[B]`, `loss` its mean, `variance/stddev`, `interval(level)` / `lower_bound` (sample-based, `_sample(n=512)`), and `boundary_probs` (ZOIB only).
- The baseline trainer uses `bce_loss(out.logit, target)` for Bernoulli and `head.loss(target, out.response)` otherwise (`baseline/train.py:164-165`).
- Targets are `y_soft` for non-Bernoulli heads or `train.target=soft`, else binary `y` (`:155,161`).
- Beta trains on interior rows only (`:131-132,222-232`).
- `batched_forward(model, source, fields=…)` (`baseline/data.py:171-214`) is the shared inference path. Its fields are `proba|mean|std|lower|upper|nll|logit|base_irt|a_q|b_q|<param>`.

---

## 4. Router interface (`src/router/routing/`)

### `base.py`
- `UnsupportedCandidatePolicy` enum `ERROR|DROP|SCORE_WITH_PRIOR` (`:54-60`). The default is `DROP` (`:155`).
- `RoutingResult` dataclass (`:66-108`) has fields:
  - `query_ids: list[str]`, `model_ids: list[str]`
  - `scores: np.ndarray [Q,M]`, `selected: np.ndarray [Q]` (column idx)
  - `lam: float=0.0`, `model_costs: Optional[np.ndarray [M]]`
  - derived: `selected_model_ids`, `selected_scores`, `model_mix()`, `to_frame()` (cols `query_id, selected_model_id, selected_score`), `score_frame()`.
- `Router(abc.ABC)` (`:138-329`):
  - ClassVars `kind="router"`, `can_route_text=False`.
  - `__init__(model_ids, *, name=None, unsupported_policy=DROP)` raises on an empty pool (`:154-160`).
  - **Only abstract method:** `predict_scores(query_ids: Sequence[str]) -> pd.DataFrame [query_id × model_id]` (`:226-231`).
  - Overridable: `predict_scores_text(texts, *, encoder=None) -> DataFrame` with positional rows (default raises, `:233-244`); `default_model_costs -> Optional[np.ndarray]` (default None, `:218-223`); `save(path)` / `load(artifact)` both raise `NotImplementedError` (`:198-216`).
  - Concrete: `model_ids` (copy), `supported_models()`, and `filter_candidates(candidates, constraints)` (duck-typed `model_id` / `capabilities` / `constraints.required_capabilities`, `:171-195`).
  - `aligned_scores(query_ids)` reindexes to `(ids, self._model_ids)` as float64 (`:247-250`).
  - `route(query_ids, *, lam=0.0, model_costs=None, eligible=None) -> RoutingResult` (`:281-299`); `__call__ = route`.
  - `route_text(texts, *, encoder, lam, model_costs, eligible)` overwrites the index with `"0".."N-1"` (`:303-322`).
  - `_route_from_scores` (`:252-279`) resolves costs: an explicit `model_costs` (Mapping or len-M array via `_resolve_costs`, `:114-132`), else `default_model_costs`. It then calls `routing_decision`.
- `router/nirt/routing_decision.py:16-45` `routing_decision(pred [Q,M], *, lam, model_costs [M], eligible [Q,M]) -> [Q] int`:
  - Non-finite cells are set to −inf.
  - Cost is subtracted only if `lam` is truthy and costs are given.
  - Rows with no finite cell fall back to column 0.
  - Ties go to the lowest index.

### `registry.py`
`REGISTRY: dict[str, type[Router]]` (`:17`). `register(cls)` requires a `kind` other than `"router"` and rejects duplicate kinds (`:20-28`). `build_router(kind, /, **kwargs)` calls `cls.from_run(run, **kwargs)` if `run=` is passed and the class has `from_run`, else `cls(**kwargs)` (`:31-47`). There are no callers of `build_router` in src/ or scripts/ other than the docstring example (`routing/__init__.py:38`).

### `routers.py` concrete routers (all `@register`)

| kind / class | Constructor | Artifacts / data loaded | `predict_scores` path | text? | `default_model_costs` |
|---|---|---|---|---|---|
| `matrix` `MatrixRouter` `:137-156` | `(scores: DataFrame, *, name, model_costs)`; pool = `scores.columns` | none | `scores.reindex(index=ids)` | no | given `model_costs` |
| `nirt` `NIRTRouter` `:162-280` | `(model, model_index, *, data=None, pathway="irt", query_pathway, query_features, name)`; pool = `sorted(model_index, key=model_index.get)` (`:184`); `from_run(run_name, *, data, runs_dir, name)` `:194-210` | `runs_dir/<run>/model.pt` via `router.nirt.checkpoint.load_run`; `TrainingData` (loaded from `load_config()` if `data=None`, `:95-101`); query/profile/feature stores | `NIRTDataset(_cross_obs(...))` → `predict_dataset` (sigmoid) → `pivot_qm` (`:216-234`) | yes: `_encode_texts` (`load_encoder(data.cfg, query_pathway)`, fallback `pathway`, cached) → `_score_embeddings` zero-pads e_q up to `model.query_dim` (`:256-264`) and per model calls `sigmoid(model(q, ref))` | train-split mean `obs.cost` per model from `data.nirt_observations()`; NaN is filled with nanmax (`:63-86`) |
| `knn` `KNNRouter` `:286-350` | `(data=None, *, k=5, pathway="retrieval", model_ids=None, name)`; default pool = train-split model_ids in obs (`:304-306`) | `TrainingData`; `_fit_knn(data,k,pathway)` cached → `(nn, C_imp, model_ids)` (`baselines_infer.py:34-47`) | `knn_router_matrix(data, ids, k, pathway, fitted)` | yes: `knn_router_matrix_from_embeddings` | same train-mean cost |
| `mlp` `MLPRouter` `:356-384` | `(model, model_ids, *, data=None, pathway="irt", name)`; fit via `training/trainers/mlp_router.py:62-72` | `TrainingData`; in-memory `nn.Sequential(Linear,ReLU,Dropout,Linear)` (`baselines_infer.py:101-106`) | `mlp_router_matrix` → `sigmoid(model(X))`, rows only for ids in `q_store._index` (`baselines_infer.py:109-117`) | no | same train-mean cost |
| `random` `RandomRouter` `:390-407` | `(model_ids, *, seed=0, name)` | none | `default_rng(seed).random((Q,M))`, re-seeded per call | no | None |

Consumers of `Router`:
- `router/agentic/router.py:46-87` (`AgenticRouter` requires `can_route_text` and calls `route_text([prompt], encoder=self.encoder, lam=self.lam)`)
- `router/router.py:39-105` (`RoutingPipeline.route` calls `filter_candidates`, then `route_text([prompt])` with lam=0, then `score_frame().iloc[0]`)
- `router/tools/router_tool.py:29`
- `training/trainers/mlp_router.py:68` imports the private `_load_training_data` from `routers.py`.

---

## 5. Config keys

References: `configs/phase0.yaml` (data pipeline, loaded as `router.config.Config` by `load_config`, `config.py:118-124`), `configs/nirt.yaml` (plain dict to `training.nirt.train.fit`), `configs/phase1.yaml` / `phase2.yaml` (plain dict to `training.nirt.baseline.train.fit`), `configs/irt_router.yaml` (alternative phase0-shaped Config).
Helpers are in `router/config.py`:
- `section(cfg, dotted)` returns a dict or `{}` (`:75-87`).
- `coerce_hidden` maps `None/"none"/"None"/0` to None (`:90-102`).
- `coerce_auto_bool("auto"|None → auto)` (`:105-107`).
- `require_choice` (`:110-115`).
- `Config.path(key)` reads `paths.<key>` with **no default**, so a missing key raises KeyError (`:64-68`).

### `training.nirt.train.fit(nirt_cfg)`, from nirt.yaml (defaults in parens)
- top: `seed` (42) `:333`; `runs_dir` (`data/processed/nirt_runs`) `:639`
- `data.`: `pathway` ("irt") `:339`, `query_pathway` (None) `:340`, `query_features` (None) `:341`
- `train.`, read at `:404-428`:
  - optimisation: `lr` (1e-3), `weight_decay` (1e-5), `batch_size` (4096), `epochs` (50), `patience` (6), `val_metric` ("bce"; any metric key, falls back to bce `:561`), `grad_clip` (0), `lr_schedule` ("none"|cosine|plateau)
  - loss/sampling: `loss` ("soft_bce"|hard_bce|mse|pairwise|listwise|bce_pairwise), `sampler` ("cell"|query; forced to query for grouped losses `:414-415`), `pairwise_weighting` ("none"|abs_diff), `loss_aux_weight` (0.5), `listwise_tau` (0.1), `batch_queries` (512), `zero_variance_weight` (1.0), `theta_cov_weight` (0.0)
  - collapse alarm: `abort_on_collapse` (False), `collapse_rank_threshold` (1.3), `collapse_agreement_threshold` (0.95), `collapse_patience` (3)
- `train.preference.`: `enabled` (False), `source` (None), `query_pathway` (None) `:347-379`; `weight` (0.3), `batch_size` (4096) `:440-441`
- `model.` goes to `build_model` / `from_config`:
  - `orientation` ("query_latent") `model.py:436`
  - `dim` (1), `model_params` ("projected"), `query_hidden` (64), `constrain_discrimination` ("auto" → dim==1) `:230-239,383-392`
  - NIRTModel only: `difficulty` ("scalar"), `model_hidden` (None), `interaction` (False), `interaction_hidden` (32), `center_discrimination` (False), `query_head_batchnorm` (False) `:241-247`
  - IRTRouterModel only: `bound_ability` (True) `:393`
  - both, via `_head_kwargs` `:110-125`: `query_head_layers` (1), `query_head_norm` ("none"), `query_head_activation` ("relu"), `query_head_dropout` (0.0), `query_head_residual` (False)
- nirt.yaml keys **with no reader in src/ or scripts/** (grep): `data.score_kind`, `data.metrics`, `train.materialize` (mentioned only in the docstring `train.py:8`).
- nirt.yaml sections read only by scripts:
  - `classical_irt.*`: `scripts/nirt/compare.py:90`, `ab_orientation.py:67-70`
  - `routing.{reference_model,lambda_sweep,reward_alphas}`: `compare.py:91-96`, `ab_orientation.py:60`
  - `baselines.{knn_router{k,pathway},mlp_router{hidden,epochs,lr,pathway}}`: `compare.py:92,146-151`
  - `knn_impute.*`: `knn_impute_sweep.py:106`
  - `evaluation.ood_holdout_families`: `evaluation/nirt/ood.py:29-30` (default `("math","code")`)

### `training.nirt.baseline.train.fit(phase1_cfg)`, from phase1.yaml / phase2.yaml
- top: `seed` (42), `out_dir` (`artifacts/phase1/baseline`) `baseline/train.py:83,208`
- `train.` via `TrainCfg.from_dict` (`:58-71`): `lr`|`learning_rate` (1e-3), `weight_decay` (0.0), `batch_size` (4096), `epochs` (60), `patience` (8), `device` ("cpu"), `target` ("binary"), `class_weighting` (False), `grad_clip` (0.0)
- `ablation.` `use_relevance` (True) / `use_warmup` (False) / `use_interaction` (True), each falling back to the matching `model.` key (`:88-90`)
- `regularization.` via `RegConfig.from_dict`: `theta_l2` (1e-4), `theta_center_l2` (1e-3), `discrimination_l2` (0), `difficulty_l2` (1e-4) (`losses.py:19-28`)
- `response.`: `binary_threshold` (0.5), `score_kind` ("effective"), `model` (falls back to `model.response_model`, "bernoulli"), `cfg` (`model.response_cfg`) (`:102-118`)
- `data.pathway` ("irt") `:101`
- `model.` via `BaselineNIRT.from_config` (`baseline/model.py:96-121`):
  - shape: `dim` falling back to `theta_dim` (8), `model_params` ("free"), `query_hidden` (64), `constrain_discrimination` ("auto")
  - options: `bound_ability` (True), `use_length_head` (True), `interaction_hidden` (32)
  - relevance/warm-up: `use_relevance`/`use_interaction`/`use_warmup`, with `model.ablation.*` taking precedence; `warmup_alpha` falling back to `warmup.alpha` (0.5); `warmup.enabled`, `warmup.learnable`
  - response: `response_model`, `response_cfg`
- response-head `cfg`: `epsilon` (1e-6); Beta and ZOIB also read `min_concentration` (1e-4), `max_concentration` (1e4), `mu_min` (1e-4) (`continuous_zoib.py:43-46`, `continuous_beta.py:39-42`, `continuous_normal.py:36`)
- `phase2.yaml` `response_models.<name>` is merged into `response` by `scripts/nirt/baseline/train_continuous.py:40-44`, which also drops `evaluation`.
- phase1/phase2 keys with no reader found by grep: `response.use_arena`, `train.loss` (`bce_with_logits`). `evaluation.n_icc` is passed via CLI args instead (`scripts/nirt/baseline/evaluate.py:84`). `retrieval.k` in phase1/2 is not read from this dict; warm-up reads `retrieval.k` from the phase0 Config (`training/retrieval/warmup.py:33`).

### Phase-0 `Config` readers (phase0.yaml / irt_router.yaml)

| Entry point | Keys (default) |
|---|---|
| `Config.path` (many) | `paths.{raw,processed,taxonomy,splits,indexes}`, no defaults |
| `TrainingData._default_pathway` `facade.py:250-253` | `embedding.default_pathway`, else the first of `embedding.pathways` |
| `router.embeddings.encoder` `:185-225,497-499` | `seed` (42); `embedding.{batch_size(64),device("auto"),pathways{<pw>:{backend("sentence_transformer"),model_name,normalize(True),max_seq_length(256),pooling("mean")}},cache_dir("data/processed/embeddings")}`; if `pathways` is not a dict, `["default"]` with top-level `embedding.*` |
| `training.data.splits` `:47-61,118,151` | `split.{cold_start_model_fraction(0),cold_start_min_observations(0),force_warm,force_cold,dedup_on("normalized_query"),validation_fraction(0)}` |
| `training.data.chance_correction` `:66-69` | `chance_correction.{method("normalized"),clip(True),warn_on_missing_choices(True)}` |
| `training.data.loaders` `:46-51,69,278,288,300,417-421,518-519` | `multiple_choice.{by_task,by_prefix}`; `sources.routerbench`, `sources.arena.local_dir`, `sources.gpt4_judge.local_dir`, `sources.lm_harness` (attribute access, no default); `sources.irt_router.{local_dir("data/raw/irt_router"),pool}`; `anchor_judge.output_dir` |
| `training.models.profiles` `:47-58,218-219` | `profiles.{config_path("configs/model_profiles.yaml"),include_empirical(True),version(1)}`; `profiles.task_families` (`training/data/families.py:17`) |
| `ClusterConfig.from_config` `clustering.py:41-55` | `seed`; `clustering.{pathway("retrieval"),umap{n_neighbors(15),min_dist(0.1),n_components(10),metric("cosine")},hdbscan{min_cluster_size(20),min_samples,metric("euclidean"),cluster_selection_method("eom")}}`; `taxonomy.version` (1) |
| `training.retrieval.query_bank` `:58-86` | `retrieval.{pathway("retrieval"),index_type("flat_ip"),k(5)}`, `seed` |
| `training.retrieval.warmup` `:32-33,77` | `data.pathway` ("irt"; normally absent from phase0.yaml, so the default applies), `retrieval.k` (5) |
| `training.nirt.baseline.data.build_arrays` `:95` | `data.pathway` ("irt"), same situation |
| `NIRTDataset.from_config` `router/data/nirt.py:230` | `paths.processed` |

---

## 6. TrainingData facade (`src/training/data/facade.py`)

Dataclass fields (`:48-55`): `cfg: Config`, `responses`, `queries`, `models` (DataFrames), `splits: dict`, `profiles: Optional[DataFrame]`. It is built by `load_training_data(cfg)` (`:359-376`), which reads `processed/{queries,models}.parquet`, responses, splits (raising if any split is None), and optional `model_profiles.parquet`.

| Member | Signature / return | External users (non-facade) |
|---|---|---|
| `cfg` | Config | `routers.py:124,128`; `evaluation/nirt/ood.py:35,60` |
| `responses` | DataFrame | `training/data/nirt.py:78`; `evaluation/nirt/ood.py:36`; `evaluation/nirt/evaluate.py:168` |
| `splits` | `{name: {"query_ids": [...]}, "cold_start_models": {"model_ids": [...]}}` | `training/data/nirt.py:60`; `training/phase0.py:31`; `training/retrieval/query_bank.py:30`; `training/data/loaders.py:414` |
| `model_ids()` / `warm_model_ids()` / `cold_start_model_ids()` `:65-72` | sorted `list[str]` | `battery.py:95`; `evaluation/nirt/evaluate.py:149`; `baseline/eval.py:168` |
| `split_query_ids(split)` `:86-94` | `set[str]` or None; raises on unknown/unbuilt split | internal |
| `query_texts(split)` `:96-101` | `{query_id: text}` | scripts |
| `correctness(split, metrics, models="warm", score_kind="effective")` `:106-153` | long DF: `query_id, model_id, dataset, metric_type, is_multiple_choice, n_choices, score_raw, score_corrected, score_effective, score` | `training/data/nirt.py:55`; `evaluation/nirt/evaluate.py:151`; `baseline/{eval.py:169,data.py:106,continuous_eval.py:37}` |
| `correctness_matrix(...)` `:155-178` | `[query × model]` via `pivot_qm` | scripts |
| `pairwise(source, split, models)` `:194-233` | DF `query_id, model_a, model_b, score_a, source, pair_id, winner` | `training/nirt/pairwise.py:58` |
| `model_profiles()`, `profile_text()` `:238-245` | | scripts |
| `query_embeddings(pathway=None)` / `profile_embeddings(pathway=None)` `:264-274` | `EmbeddingStore` or None | `router/data/nirt.py:241-242`; `routers.py:222-248`; `baselines_infer.py:26,44,69,112`; `pairwise.py:59-60`; `evaluation/nirt/{ood.py:90-91,evaluate.py:143-144}`; `trainers/mlp_router.py:41`; `baseline/eval.py:40,167` |
| `profile_embedding(model_id, pathway)` `:276-280` | ndarray or None | |
| `query_features(name=None)` `:282-288` | `EmbeddingStore` (`load_query_features(cfg, name or "default")`) or None | `router/data/nirt.py:255`; `routers.py:229` |
| `nirt_observations(**build_kw)` `:293-301` | obs DF; reads parquet if present and there are no kwargs, else builds | `routers.py:68,305`; `baselines_infer.py:23`; `evaluation/nirt/routing.py:46,63`; `battery.py:86`; `evaluation/baselines/classical_irt.py:79` |
| `nirt_dataset(split="train", pathway="irt", query_pathway, query_features, **kw)` `:303-316` | `NIRTDataset` | `training/nirt/train.py:359-362`; `router/nirt/predict.py:72`; `evaluation/nirt/evaluate.py:76,82,164`; `baseline/data.py:97` |
| `summary()` `:321-350` | dict; indexes `splits["train"/"validation"/"test"]` directly | scripts |

`EmbeddingStore` API relied on (`router/embeddings/encoder.py:242-365`):
- data: `ids`, `_index` (private dict), `matrix` (memmap `[n, dim]`), `dim`
- lookup: `__contains__`, `row_of(id)`, `rows_of(ids)` (raises KeyError on missing), `get(id)`, `gather(ids)`
- classmethods: `load(dir, mmap)`, `exists(dir)`
- `_index` is accessed outside the class at `router/data/nirt.py:71-72` and `baselines_infer.py:113`.

Import boundary notes:
- `router/routing/routers.py:89-101` is the documented router→training crossing (`_load_training_data`).
- `router/data/nirt.py:32-33` imports `TrainingData` under TYPE_CHECKING only.
- `router.nirt.components` is imported by training (`baseline/model.py:24`, `continuous_zoib.py:31`).

---

## 7. Entry points

`[project.scripts] router-phase0 = training.phase0:main` (`pyproject.toml:40`) runs `build_tables`/`write_tables` (`training.data.response_matrix`), `check_responses`, `make_splits`/`check_leakage`/`write_splits`, profiles, the NIRT obs table, and optional embeddings/taxonomy/query-bank (`training/phase0.py:1-60`). Shared CLI helpers live in `training/cli.py` (`base_parser`/`get_config` load a phase0 Config via `--config`; also `raw_parser`, `resolve`, `write_json`, `zeroshot_only`, `float_table`).

| Script | Drives |
|---|---|
| `scripts/data/download_routerbench.py`, `download_arena.py`, `download_irt_router.py`, `run_lm_harness.py` | raw downloads / lm-harness (only `training.cli`) |
| `scripts/data/build_response_matrix.py` | `training.data.response_matrix.build_tables/write_tables`, `chance_correction.estimate_model_bias`, `quality.check_responses` |
| `scripts/data/build_splits.py` | `training.data.splits.make_splits/make_presplit/check_leakage/write_splits` |
| `scripts/data/build_nirt_dataset.py` | `training.data.nirt.build_nirt_observations/write_nirt_observations/nirt_dataset_from_config` |
| `scripts/data/encode_irt_router_pathway.py` | `router.embeddings.{EmbeddingStore,default_store_dir,load_encoder}`, `training.data.{model_registry,normalize}` |
| `scripts/data/import_irt_router_embeddings.py` | `router.embeddings.EmbeddingStore`, `training.taxonomy.relevance.relevance_store_dir` |
| `scripts/data/select_anchor_model.py` | facade, `evaluation.nirt.routing.train_quality` |
| `scripts/data/select_anchor_judge_queries.py` | facade, `evaluation.data.stratify.sample_stratified_queries` |
| `scripts/data/collect_anchor_judgments.py` | facade, `evaluation.judge.{client.JudgeClient,prompts.build_prompt}`, `evaluation.data.judge_responses` |
| `scripts/embeddings/build_query_embeddings.py` | `router.embeddings.{build_store,load_encoder,available_pathways,default_store_dir}` |
| `scripts/embeddings/build_profile_embeddings.py` | same + `training.models.profiles` |
| `scripts/embeddings/build_query_features.py` | `training.data.query_features.build_query_features` |
| `scripts/models/build_model_profiles.py` | `training.models.profiles.build/write_model_profiles` |
| `scripts/taxonomy/cluster_queries.py` / `generate_taxonomy.py` / `build_relevance.py` | `training.taxonomy.{clustering,taxonomy,relevance}` |
| `scripts/retrieval/build_query_bank.py` / `build_warmup.py` | `training.retrieval.{query_bank,warmup}`, `evaluation.nirt.ood` |
| `scripts/nirt/train_nirt.py` | `training.nirt.train.fit` (nirt.yaml + CLI overrides), `evaluation.nirt.ood.ood_datasets` for `--ood` |
| `scripts/nirt/eval_nirt.py` | `router.nirt.checkpoint.load_run`, `evaluation.nirt.evaluate.evaluate_split` |
| `scripts/nirt/coldstart.py` | `load_run`, `evaluation.nirt.evaluate.cold_start_eval` |
| `scripts/nirt/compare.py` | `load_run`, `predict_matrix_from_dataset`, `baselines_infer.{knn,mlp}_router_matrix`, `trainers.mlp_router.fit_mlp_router`, `evaluation.baselines.classical_irt` |
| `scripts/nirt/ab_orientation.py` | `training.nirt.train.fit`, `load_run`, `predict_matrix`, classical IRT |
| `scripts/nirt/route_eval.py` | `load_run`, `predict_matrix`, `baseline.data.checkpoint_matrix`, `evaluation.nirt.routing`, `evaluation.routing.oracle` |
| `scripts/nirt/misroute_analysis.py` | reads route_eval CSV/parquet outputs + nirt.yaml; only `training.cli`, `router.config` |
| `scripts/nirt/anchor_judge_analysis.py` | facade, `evaluation.routing.oracle.routing_evaluation` |
| `scripts/nirt/capacity_diagnostics.py` | `load_run`, `predict_matrix*`, `mlp_router_matrix`, `fit_mlp_router`, `training.nirt.metrics`, classical IRT |
| `scripts/nirt/complexity_search.py` | `load_run`, `predict_dataset/predict_matrix`, `evaluation.nirt.{routing,evaluate}`, `evaluation.routing.oracle` |
| `scripts/nirt/knn_impute_sweep.py` | `training.retrieval.knn_impute.build_knn_imputed_store`, `training.nirt.train.fit`, `load_run`, `predict_matrix*` |
| `scripts/nirt/shrinkage_eval.py` | `load_run`, `predict_matrix`, `evaluation.nirt.routing`, `router.nirt.shrinkage` |
| `scripts/nirt/theta_collapse.py` | `load_run`, facade |
| `scripts/nirt/train_test_gap.py` | `load_run`, `predict_dataset`, `training.nirt.metrics` |
| `scripts/nirt/baseline/train_baseline.py` | `training.nirt.baseline.train.fit` (phase1.yaml) |
| `scripts/nirt/baseline/train_continuous.py` | `baseline.train.fit` (phase2.yaml, merges `response_models.<name>`) |
| `scripts/nirt/baseline/evaluate.py` | `baseline.checkpoint.load_run`, `baseline.eval.evaluate_checkpoint`, `baseline.continuous_eval.evaluate_continuous` |
| `scripts/nirt/baseline/inspect_baseline.py` | `baseline.checkpoint.load_run`, `baseline.data.{build_arrays,batched_forward}`, `baseline.diagnostics`, `baseline.eval.theta_matrix` |
| `scripts/nirt/baseline/synthetic.py` | `baseline.train.fit`, `baseline.synthetic` |
| `scripts/nirt/baseline/compare_response_models.py`, `boundary_stats.py` | `baseline.continuous_eval.{compare_response_models,boundary_statistics}` |
| `scripts/pool_expansion/run_phase.py` | `evaluation.pool_expansion.run_battery` |
| `scripts/route_compare.py` | both `load_run`s, `baseline.data.checkpoint_matrix`, `router.nirt.predict`, `evaluation.nirt.routing`, facade |

No script constructs a `router.routing.*Router` or `AgenticRouter` directly, based on the import lists above.
