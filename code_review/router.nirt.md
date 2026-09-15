# Code review: router.nirt

**Files reviewed:** 9 (970 lines): `__init__.py`, `baselines_infer.py`, `checkpoint.py`, `components.py`, `frames.py`, `model.py`, `predict.py`, `routing_decision.py`, `shrinkage.py` · **Context-only:** `src/training/nirt/train.py`, `src/training/nirt/pairwise.py`, `src/training/trainers/mlp_router.py`, `src/router/routing/base.py`, `src/evaluation/routing/oracle.py`, `code_review/_contracts.md`
**Summary:** 9 findings: 4 correctness, 3 design, 2 performance · 8 confirmed, 1 suspected

`routing_decision.py` bugs are filed once in `router.routing.md`, where the caller lives: the NaN-cost argmax (RR-04) and the column-0 fallback for rows with nothing selectable (RR-03).

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| RN-01 | `src/router/nirt/predict.py:42-47` | correctness | medium | confirmed |
| RN-02 | `src/router/nirt/shrinkage.py:60-63` | correctness | low | confirmed |
| RN-03 | `src/router/nirt/components.py:38-42` | correctness | low | confirmed |
| RN-04 | `src/router/nirt/model.py:189-190` | correctness | low | suspected |
| RN-05 | `src/router/nirt/model.py:380-396` | design | low | confirmed |
| RN-06 | `src/router/nirt/model.py:398-405` | design | low | confirmed |
| RN-07 | `src/router/nirt/checkpoint.py:29` | design | low | confirmed |
| RN-08 | `src/router/nirt/predict.py:23-29` | performance | low | confirmed |
| RN-09 | `src/router/nirt/shrinkage.py:57-59` | performance | low | confirmed |

---

## src/router/nirt/predict.py

### RN-01: With `center_discrimination` in projected mode, predictions depend on arbitrary batch boundaries `:42-47`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
45:        for i in range(0, len(out), batch_size):
46:            sl = slice(i, i + batch_size)
47:            out[sl] = torch.sigmoid(model(q[sl], ref_src[sl])).numpy()
```

**What breaks:** In projected mode, `_center_discrimination` subtracts `r = a.mean(0)` over the rows of the current forward call (`model.py:281-282`). `predict_dataset` cuts the long observation table into fixed 8192-row chunks that ignore query boundaries. Two consequences:
- **Ranking:** a query whose candidate rows straddle a chunk boundary is scored against two different `r` vectors. The docstring promises "exactly lossless for within-query ranking" (`model.py:263-267`), which does not hold at inference, and the routing argmax can change.
- **Calibration:** the probability for the same `(q, m)` cell depends on which other rows share its chunk. BCE, ECE and calibration results therefore change with row order and `batch_size`.

`NIRTRouter.predict_scores` reaches this code through `predict_matrix_from_dataset`.
**Evidence:** Traced `predict_matrix` → `predict_matrix_from_dataset` → `predict_dataset` → `model.forward` → `model_parameters` → `_center_discrimination`. The same root cause breaks training-side pairwise scoring (XA-04). `tests/router/nirt/test_nirt_model.py` is the only test mentioning `center_discrimination`, and `predict_dataset` has no batch-invariance test.
**Direction:** For projected mode, compute `r` once over the whole pool (the mean of `a_head` over all profile embeddings), store it as a buffer after training, and use it at inference. At minimum, chunk by query.

### RN-08: `predict_dataset` materialises every embedding row of the split `:23-29`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
23:    g = dataset.gather()
29:        ref_src = torch.from_numpy(np.ascontiguousarray(g["model_embedding"], dtype=np.float32))
```

**What breaks:** `gather()` copies `n_obs × (D_q + D_m) × 4` bytes, and projected mode duplicates the same profile vector for every observation of a model. At 36k queries × 20 models × (768 + 768) dims, that is about 4.4 GB before the model runs. `NIRTDataset.gather`'s own docstring (`router/data/nirt.py:146-148`) warns against doing this for large sets.
**Evidence:** Read `:19-48` and `router/data/nirt.py:143-170`.
**Direction:** Iterate `dataset.dataloader(shuffle=False)`, or project each model's profile once and index it.

---

## src/router/nirt/shrinkage.py

### RN-02: An empty training bank yields NaN weights instead of an error `:60-63`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
60:    kk = min(k, sims.shape[1])
61:    topk = np.partition(sims, -kk, axis=1)[:, -kk:]
62:    sim = topk.max(axis=1) if kk == 1 else topk.mean(axis=1)
```

**What breaks:** With `train_emb` of shape `(0, d)`, `kk = 0`. The slice `[:, -0:]` selects all zero columns, so `.mean(axis=1)` is NaN, and every weight is NaN. `shrink_predictions` then returns an all-NaN matrix, which routes to column 0 (RR-03) instead of failing loudly.
**Evidence:** Traced numpy's `-0:` slice (it equals `0:`). The `k < 1` guard at `:55` does not cover an empty bank.
**Direction:** Raise when `train_emb` is empty.

### RN-09: Brute-force cosine allocates a dense `(Nq, Ntr)` float64 matrix `:57-59`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
57:    e = _unit(np.asarray(eval_emb, dtype=np.float64))
59:    sims = e @ t.T                                   # (Nq, Ntr)
```

**What breaks:** The docstring calls brute force "fine up to ~10^5 train queries". At that size, 10k eval queries need 8 GB for `sims`, plus the copy `np.partition` makes. RouterBench (about 29k train queries against a 7k test split) already needs about 1.6 GB per call, and `scripts/nirt/shrinkage_eval.py:89` calls it inside a midpoint sweep.
**Evidence:** Read `:55-63` and the caller.
**Direction:** Compute the top-k once outside the sweep, in eval-row chunks, in float32, or use `training.retrieval.QueryBank`.

---

## src/router/nirt/components.py

### RN-03: A fixed warm-up `alpha` is not clamped, while the learnable one is `:38-42`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
38:        a = min(max(float(alpha), 1e-3), 1 - 1e-3)
41:        else:
42:            self.register_buffer("_alpha", torch.tensor(float(alpha)))
```

**What breaks:** The clamped `a` is used only for the learnable logit. A config value such as `alpha: 1.5` or `-0.2` is stored raw, and `(1-a) e_q + a * neighbor_mean` then extrapolates away from both points instead of blending them.
**Evidence:** Read `:34-53`.
**Direction:** Validate `0 <= alpha <= 1` and use the same value in both branches.

---

## src/router/nirt/model.py

### RN-04: θ-BatchNorm fails on a one-row training batch `:189-190`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
189:        if self.query_head_batchnorm:
190:            self.query_bn = nn.BatchNorm1d(self.dim, affine=False)
```

**What breaks:** In train mode, `BatchNorm1d` raises `ValueError: Expected more than 1 value per channel` for a single-row batch. Under `train.sampler: query`, which grouped losses force (`training/nirt/train.py:413-415`), a batch is a query's candidate group. A query with only one observed model, or a one-row final batch, would crash the epoch.
**Evidence:** Not traced end to end. It depends on whether `train.py` packs several query groups per batch and drops the last partial batch.
**Question for you:** can a training batch contain exactly one row when `query_head_batchnorm: true`? If so, use `drop_last` or skip BN when `B == 1`.

### RN-05: `IRTRouterModel.from_config` silently ignores NIRT-only knobs `:380-396`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
381:    def from_config(cls, model_cfg: dict, *, n_models: int,
393:            bound_ability=bool(model_cfg.get("bound_ability", True)),
```

**What breaks:** A sweep that sets `difficulty: vector`, `model_hidden`, `interaction`, `center_discrimination` or `query_head_batchnorm` together with `orientation: model_latent` trains the same model under different run names. The saved config then records a variant that never existed, so orientation A/B comparisons (`scripts/nirt/ab_orientation.py`) can be mislabelled.
**Evidence:** Compared the keys read by `NIRTModel.from_config` (`:230-248`) with those read by `IRTRouterModel.from_config` (`:383-396`).
**Direction:** Raise or warn when a key the chosen orientation doesn't support has a non-default value.

### RN-06: `bound_ability` has no effect with `model_params: free` `:398-405`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
400:        if self.model_params == "projected":
401:            theta = self.theta_head(model_ref)
402:            return torch.sigmoid(theta) if self.bound_ability else theta
405:        return self.theta(model_ref)
```

**What breaks:** In free mode, ability never passes through the sigmoid, whatever the flag says. Projected and free IRT-Router runs therefore differ in more than their parameterisation, and `bound_ability: true` in a free config does nothing.
**Evidence:** Read `:398-405`. The code does not document whether the unbounded free mode is intentional (paper M-IRT).
**Direction:** Apply the flag in both branches, or reject it for free mode.

---

## src/router/nirt/checkpoint.py

### RN-07: `torch.load(weights_only=False)` executes arbitrary pickles from a run directory `:29`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
29:    blob = torch.load(base / name / "model.pt", map_location="cpu", weights_only=False)
```

**What breaks:** Loading a run from a shared or downloaded `runs_dir` executes any code embedded in the pickle. `NIRTRouter.from_run` is the serving entry point, so this is on the serving path.
**Evidence:** The blob holds tensors, ints, a `model_index` dict and a plain `config` dict (`_contracts.md` §1), all of which `weights_only=True` can load.
**Direction:** Use `weights_only=True`.

---

## src/router/nirt/__init__.py, frames.py, baselines_infer.py, routing_decision.py

No new findings. See RR-03 and RR-04 for `routing_decision.py`.

---

## Checked and ruled out
- **`mlp_router_matrix` scoring with dropout active:** `fit_mlp_router` calls `model.eval()` before returning (`training/trainers/mlp_router.py:58`).
- **`_fit_knn` on an all-NaN training column:** `pivot_table` drops all-NaN columns by default, so `nanmean` never sees one.
- **`knn_router_matrix` / `mlp_router_matrix` dropping ids that are missing from the store:** callers reindex, and `router.routing.md` covers the effect.
- **`routing_decision` breaking exact ties toward the lowest column:** documented as the live decision rule.
- **The legacy `nn.Sequential` branch in `_query_head`:** kept so old state_dicts load, as documented.
- **Free-mode centring reference:** it uses the full pool (`self.a.weight`), so it does not depend on the call.
