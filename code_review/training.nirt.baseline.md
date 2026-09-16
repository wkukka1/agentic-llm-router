# Code review: training.nirt.baseline

**Files reviewed:** 16 (2,220 lines) · **Context-only:** `src/router/nirt/components.py`, `src/training/nirt/metrics.py` (read to check metric semantics), `src/training/data/facade.py`, `src/training/data/nirt.py`
**Summary:** 26 findings: 16 correctness, 6 design, 4 performance · 18 confirmed, 8 suspected

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| TB-01 | `src/training/nirt/baseline/continuous_eval.py:104-111` | correctness | medium | confirmed |
| TB-02 | `src/training/nirt/baseline/continuous_eval.py:216-228` | correctness | medium | confirmed |
| TB-03 | `src/training/nirt/baseline/data.py:105-110` | correctness | medium | confirmed |
| TB-04 | `src/training/nirt/baseline/continuous_normal.py:57-59` | correctness | medium | confirmed |
| TB-05 | `src/training/nirt/baseline/data.py:105-110` | correctness | medium | suspected |
| TB-06 | `src/training/nirt/baseline/eval.py:133-136` | correctness | low | confirmed |
| TB-07 | `src/training/nirt/baseline/data.py:204-206` | correctness | low | confirmed |
| TB-08 | `src/training/nirt/baseline/continuous_zoib.py:68-70` | correctness | low | confirmed |
| TB-09 | `src/training/nirt/baseline/eval.py:65-83` | correctness | low | confirmed |
| TB-10 | `src/training/nirt/baseline/diagnostics.py:72-77` | correctness | low | confirmed |
| TB-11 | `src/training/nirt/baseline/eval.py:36-42` | correctness | low | confirmed |
| TB-12 | `src/training/nirt/baseline/synthetic.py:159` | correctness | low | confirmed |
| TB-13 | `src/training/nirt/baseline/model.py:48-49` | design | low | confirmed |
| TB-14 | `src/training/nirt/baseline/response_head.py:53` | design | low | confirmed |
| TB-15 | `src/training/nirt/baseline/train.py:177-180` | design | low | confirmed |
| TB-16 | `src/training/nirt/baseline/continuous_eval.py:104-132` | performance | low | confirmed |
| TB-17 | `src/training/nirt/baseline/data.py:71-77` | performance | low | confirmed |
| TB-18 | `src/training/nirt/baseline/eval.py:98-99` | performance | low | confirmed |
| TB-19 | `src/training/nirt/baseline/model.py:83` | performance | low | confirmed |
| TB-20 | `src/training/nirt/baseline/continuous_eval.py:52-55` | correctness | low | suspected |
| TB-21 | `src/training/nirt/baseline/eval.py:190-192` | correctness | low | suspected |
| TB-22 | `src/training/nirt/baseline/model.py:142-143` | correctness | low | suspected |
| TB-23 | `src/training/nirt/baseline/model.py:140-141` | correctness | low | suspected |
| TB-24 | `src/training/nirt/baseline/train.py:166-167` | design | low | suspected |
| TB-25 | `src/training/nirt/baseline/synthetic.py:126-128` | design | low | suspected |
| TB-26 | `src/training/nirt/baseline/synthetic.py:84-87` | design | low | suspected |

---

## src/training/nirt/baseline/continuous_eval.py

### TB-01: Beta interval coverage is scored against boundary rows the Beta can never cover `:104-111`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
107:    interior = (ev.y_soft > 0) & (ev.y_soft < 1)
108:    imask = interior if getattr(head, "interior_only", False) else None
```

**What breaks:** For the Beta head, `imask` only applies to the NLL (`continuous_metrics:68-70`). `coverage_90`, `coverage_50`, `interval_width_90`, MAE/RMSE and mean-calibration ECE (`:71-87`) are all computed over every row. Beta samples lie strictly inside (0, 1). The interval `[lower, upper]` from `_sample`/`torch.quantile` therefore never contains an exact 0 or 1, so every boundary row counts as a miss. With about 78% of rows at the boundary (docs: "interior 22%"), Beta coverage is capped near 0.22 whatever the head's calibration. The published numbers match that cap: `docs/baseline_control_arm.md:275,282` reports Beta coverage_90 = 0.20 and says "its 50% interval covers 9%". `compare_response_models` then puts this number beside ZOIB/Normal coverage and tells the reader to compare coverage across rows (`:244-246`). The Beta head looks badly calibrated on uncertainty, but the metric's population causes that, not the head. The NLL uses interior rows and the coverage uses all rows, so the Beta row mixes two populations.
**Evidence:** `continuous_beta.py:74-75` samples from `Beta`, whose support is (0,1). `evaluate_continuous` passes `fw["lower"]/["upper"]` built over all of `ev` (`:105-106`). `tests/training/nirt/baseline/test_phase2_evaluation.py:21-42` covers `continuous_metrics` only with synthetic interior arrays and no mask, so this case is untested.
**Direction:** When the head is `interior_only`, apply `imask` to coverage/width (and report MAE/ECE both on interior rows and on all rows), or report the Beta row's coverage as "interior-conditional" and keep it out of the cross-row coverage comparison.

### TB-02: `compare_response_models` puts Bernoulli MAE/RMSE/ECE against binary labels next to continuous heads scored against graded `y_soft` `:216-228`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
223:                "nll": pred["bce"], "mae": pred["mae"], "rmse": pred["mse"] ** 0.5,
224:                "mean_calibration_ece": r["calibration"]["ece"],
```

**What breaks:** For the Bernoulli row, `pred` comes from `evaluate_checkpoint`, which computes `prediction_metrics(ev.y, p)` (`eval.py:104`) and `calibration_report(ev.y, p)` (`eval.py:107`), with `ev.y` = `1[y_soft >= threshold]`. For the continuous rows, MAE/RMSE/ECE are computed against `ev.y_soft` (`:65-81`). The output note (`:244-246`) says to read MAE/RMSE/calibration across rows. Those columns do not measure the same target, so the Bernoulli-vs-ZOIB comparison of "expected score error" is wrong. The two targets differ on every graded row with 0<y<1 (about 22% of rows).
**Evidence:** Traced `compare_response_models -> evaluate_checkpoint(write=False) -> prediction_metrics(ev.y, p)`; `training/nirt/metrics.py:107,114` computes brier/mae against `y_true` as given. No test calls `compare_response_models` (grep over tests/).
**Direction:** For the Bernoulli row, compute MAE/RMSE/mean-calibration against `ev.y_soft` using `proba` as the expected score (for example, reuse `continuous_metrics` with Bernoulli NLL), so every row shares one target.

### TB-16: `evaluate_continuous` runs 5+ full forward passes, plus 20 sampled passes for one plot `:104-132`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
105:    fw = batched_forward(model, ev, fields=tuple(fields), y=ev.y_soft, level=level)
106:    fw50 = batched_forward(model, ev, fields=("lower", "upper"), level=0.5)
129:    a_all = batched_forward(model, ev, fields=("a_q", "b_q"))
```

**What breaks:** The same `ev` goes through the model separately for `fw`, `fw50`, `bf` (`:117`), `a_all` (`:129`), and `_response_param_summary` (`:171`). `_plot_uncertainty_calibration` (`:282-284`) then runs 10 more full passes. Each pass draws 512 samples twice per batch for the sampling heads (see TB-07). One `batched_forward` call can return all fields, and all quantile levels can come from a single sample tensor.
**Evidence:** Read `:104-132` and `:276-290`; `batched_forward` recomputes `model(...)` on each call (`data.py:191-195`).
**Direction:** Merge the fields into one `batched_forward` call, and compute quantiles for every requested level from one draw of samples per batch.

### TB-20: `_frac` / `boundary_statistics` silently mis-bucket NaN scores `:52-55`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
54:    return {"n": int(len(y)), "y==0": float((y <= 0).mean()), "y==1": float((y >= 1).mean()),
55:            "0<y<1": float(((y > 0) & (y < 1)).mean()), "mean": float(y.mean())}
```

**What breaks:** `correctness()` coerces unparsable scores to NaN (`facade.py:139`), and `np.clip` keeps NaN. A NaN row counts in `n` but falls in none of the three buckets, so the fractions sum to less than 1 and `mean` is NaN. An empty split gives NaN plus a RuntimeWarning (`n = max(len(y),1)` is computed and never used).
**Evidence:** Suspected because it depends on whether `responses.score` ever holds non-numeric values for correctness metrics. Maintainer question: can `score_effective` be NaN for a warm correctness row?
**Direction:** Drop non-finite `y` before bucketing and report the count dropped.

---

## src/training/nirt/baseline/data.py

### TB-03: `score_kind` values other than `"raw"` are silently ignored `:105-110`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
105:    if score_kind == "raw":
106:        key = d.correctness(split=split, models="warm", score_kind="raw").set_index(
110:    y = (y_soft.astype(np.float64) >= float(binary_threshold)).astype(np.float32)
```

**What breaks:** `y_soft` is the `target` column of `nirt_observations.parquet`. That column was built once with `build_nirt_observations(score_kind="effective")` by default (`training/data/nirt.py:36,55,69`). Only `"raw"` is re-derived. `response.score_kind: corrected` (a valid `ScoreKind`, `facade.py:148-152`), or any typo, trains on effective scores. Meanwhile `score_kind` is written to `meta`, to provenance (`train.py:263`) and to `config.yaml`, so the artifact records a label definition it did not use. The same applies if the parquet was built with a non-default score kind: `score_kind: effective` then silently means whatever the parquet holds.
**Evidence:** Traced `fit -> build_arrays(score_kind=resp["score_kind"]) -> d.nirt_dataset -> NIRTDataset.from_config` (reads the parquet `target`). No test exercises `score_kind="corrected"` through `build_arrays` (grep `score_kind` in tests: only facade/nirt-data tests).
**Direction:** Validate `score_kind` with `require_choice` and re-derive `y_soft` from `correctness(score_kind=...)` for every kind except the one the parquet was built with, or raise.

### TB-05: NaN / out-of-range `y_soft` reaches the continuous losses unfiltered `:105-110`
**Category:** correctness · **Severity:** medium · **Confidence:** suspected

```python
108:        y_soft = np.array([key.get((q, m), np.nan) for q, m in zip(query_ids, model_ids)], np.float32)
110:    y = (y_soft.astype(np.float64) >= float(binary_threshold)).astype(np.float32)
```

**What breaks:** Under `score_kind="raw"`, a (q, m) pair missing from the raw correctness frame, or with a NaN `score_raw`, gives `y_soft = NaN`. `y` becomes 0 because `NaN >= t` is False. The trainer then feeds `y_soft` to `head.loss` for Normal/ZOIB, and to `bce_loss` when a Bernoulli run uses `train.target: soft` (`train.py:155-165`). One NaN makes the batch loss NaN and poisons every parameter through Adam. For ZOIB, NaN fails both `is0` and `is1` and takes the interior branch (`continuous_zoib.py:74-77`). Beta drops the row silently (`train.py:226`). Separately, the effective path never clips `y_soft` to [0,1]. With `chance_correction.clip: false` (a config option, `chance_correction.py:68,97`), negative targets go into soft BCE, which is not a valid Bernoulli likelihood. ZOIB counts them as exact zeros.
**Evidence:** `NIRTDataset` drops rows with a null *effective* target only (`router/data/nirt.py:70-76`). The raw lookup is a second, unaligned join, and no finiteness check exists anywhere in `build_arrays` or `fit`. Maintainer question: can a warm correctness row have non-null `score_effective` but NaN `score_raw`, or can `responses` hold duplicate (q, m) correctness rows (the `to_dict` would then pick one arbitrarily)?
**Direction:** After building `y_soft`, mask out non-finite rows (report the count in `meta`), and either clip to [0,1] or assert the range for the continuous heads.

### TB-07: Sampled `lower`/`upper` come from independent, unseeded draws `:204-206`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
204:                elif f in ("lower", "upper"):
205:                    lo, hi = head.interval(r, level)
206:                    v = lo if f == "lower" else hi
```

**What breaks:** When `fields` contains both `lower` and `upper`, `head.interval` runs twice per batch. For the Bernoulli/Beta/ZOIB heads each call draws a fresh 512-sample tensor (`response_head.py:79-90`), so `lower` and `upper` come from different Monte Carlo draws, and with no RNG seeding, `coverage_90`/`coverage_50` and route_eval's `lower`-field (LCB) routing change from run to run. It also doubles the sampling cost. Normal has analytic intervals and is unaffected.
**Evidence:** Read `batched_forward` loop; no `torch.manual_seed`/generator in `continuous_eval.py`, `data.py`, or `response_head.py`.
**Direction:** Compute `interval` once per batch and cache both ends; pass a seeded `torch.Generator` for evaluation.

### TB-17: `_join` does a Python-level `row_of` per observation `:71-77`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
73:    rows = np.array([store.row_of(q) if q in store else -1 for q in query_ids])
74:    out = np.full((len(query_ids), store.dim), fill, dtype=np.float32)
```

**What breaks:** `query_ids` is per observation (N = Q×M), so the loop runs a dict lookup twice per row for relevance and again for warm-up, even though queries repeat M times. On an empty `query_ids`, `np.array([])` is float64, and `matrix[rows[hit]]` raises `IndexError: arrays used as indices must be of integer type`. That could happen with an empty split after the `model_index` filter.
**Evidence:** `EmbeddingStore._index` is a dict (`_contracts.md` §6); `build_arrays:132,138` pass per-observation ids.
**Direction:** Map unique query ids once (`np.unique(..., return_inverse=True)`), build the rows with `dtype=np.int64`, then gather.

---

## src/training/nirt/baseline/continuous_normal.py

### TB-04: Probability diagnostics hard-code a 0.5 threshold instead of `binary_threshold` `:57-59`
**Category:** correctness · **Severity:** medium · **Confidence:** confirmed

```python
57:    def as_probability(self, out):
58:        # P(Y >= 0.5) under the predictive Normal
59:        return 0.5 * torch.erfc((0.5 - out["mu"]) / (out["sigma"] * _SQRT2))
```

**What breaks:** Binary labels are `y = 1[y_soft >= binary_threshold]` (`data.py:110`), and `binary_threshold` is configurable (`phase1.yaml:25`, `phase2.yaml:23`). With a threshold other than 0.5:
- The Normal head's `proba` stays P(Y≥0.5).
- `continuous_metrics` builds its diagnostic labels as `(y >= 0.5)` (`continuous_eval.py:76`) and ignores the checkpoint setting.
- The trainer's `val_*` history (`train.py:178`) and the cold-start metrics (`eval.py:192,199`) score that P(Y≥0.5) against labels thresholded elsewhere.

All BCE/accuracy/AUC "diagnostics" for such a run are then computed against a different event from the one predicted. Beta/ZOIB return E[Y] as `proba`, which is not an event probability at any threshold, so their `bce_diag_*` are not proper scores either.
**Evidence:** Traced `evaluate_continuous -> continuous_metrics(ev.y_soft, ..., fw["proba"])` → `prediction_metrics((y>=0.5), proba)`. Tests only use threshold 0.5. Note: `_contracts.md` §3 lists Normal `as_probability` as clamp(mean); the code overrides it to P(Y≥0.5).
**Direction:** Pass `binary_threshold` into `as_probability`/`continuous_metrics` (store it on the head or thread it from `load_run` settings). For Beta/ZOIB, compute P(Y≥t) from the distribution instead of E[Y].

---

## src/training/nirt/baseline/continuous_zoib.py

### TB-08: `clamp_min(min_conc)` on alpha/beta silently changes the distribution and stops the gradient `:68-70`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
68:    def _beta(self, out):
69:        return torch.distributions.Beta(out["alpha"].clamp_min(self.min_conc),
70:                                        out["beta"].clamp_min(self.min_conc))
```

**What breaks:** `alpha = mu*kappa` can fall as low as `mu_min*min_conc = 1e-8`. Whenever `mu*kappa < min_conc` (for example mu at its 1e-4 floor with kappa < 1), the density used for NLL and sampling has a different mean from `mu`, which is what `mean()`/`variance()` report (`:80-86`). The clamp also zeroes the gradient to kappa and mu for those rows. The Beta head has the same issue (`continuous_beta.py:57-59`). Its docstring claims the clamps on mu/kappa alone keep alpha and beta above the floor, which they do not. This is limited to extreme logits, but ZOIB is the selected production head.
**Evidence:** Arithmetic from the defaults `mu_min=1e-4`, `min_concentration=1e-4` (`:43-45`). `test_boundary_probs_and_finite_extremes` checks only finiteness.
**Direction:** Enforce the floor through parameterisation (for example `kappa >= min_conc / mu_min`, or `alpha = mu*kappa + min_conc`) so `mean()` and the density agree.

---

## src/training/nirt/baseline/continuous_beta.py

No separate findings; see TB-08 (same clamp at `:57-59`) and TB-01 (interval support).

---

## src/training/nirt/baseline/eval.py

### TB-06: Evaluating on `split="validation"` overwrites the trainer's `validation` block in `metrics.json` `:133-136`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
134:        prev = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
135:        (d / "metrics.json").write_text(json.dumps({**prev, split: result}, indent=2, default=float),
```

**What breaks:** `fit` writes `metrics.json = {validation: <best-epoch val metrics>, best_epoch, train_imbalance}` (`train.py:213-214`). `evaluate_checkpoint(split="validation")` and `evaluate_continuous(split="validation")` (`continuous_eval.py:151-153`) overwrite the `validation` key with a differently shaped eval dict, losing the training-time record. A Bernoulli eval followed by a continuous eval of the same split in one dir would also overwrite each other.
**Evidence:** Key collision traced between `train.py:213` and `eval.py:135`.
**Direction:** Namespace eval results (for example `{"eval": {split: ...}}`) or write `eval_<split>.json`.

### TB-09: ICC curves drop the interaction residual that is on by default `:65-83`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
81:        curves[label] = icc_curve(a[i], float(b[i]), dim_index=dim, theta_hold=theta_mean,
82:                                  sweep=np.linspace(lo - pad, hi + pad, 81))
```

**What breaks:** `icc_curve` plots `sigmoid(θ·a − b)` (`diagnostics.py:112`). The model's logit is `base + γ·MLP([a, θ, aθ, r_q])` whenever `use_interaction` is true, which is the default (`train.py:90`, `model.py:113`). After γ trains away from 0, the "item characteristic curves" and their `monotone_increasing` flag describe the base IRT term, not the fitted model.
**Evidence:** `InteractionLayer.forward` (`components.py:104-111`); `_iccs` receives only `a_q`, `b_q`, θ.
**Direction:** Build the ICC by running `model.interaction` on the swept θ (and the query's `r_q`), or label the output as "base IRT ICC" and add γ to the report.

### TB-11: `profile_matrix` silently drops models missing from the store, so the θ rows no longer line up with `model_index` `:36-42`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
41:    order = [m for m in sorted(model_index, key=model_index.get) if m in store]
42:    return np.ascontiguousarray(store.gather(order), dtype=np.float32)
```

**What breaks:** For projected runs, if the profile store is missing any checkpoint model (for example after profiles were rebuilt), `theta_matrix` returns `(M', K)` with M' < M and no warning. The spectrum and ICC `theta_mean` then silently use a subset, and a caller that indexes rows by `model_index` would get the wrong models.
**Evidence:** Callers: `eval.py:108`, `continuous_eval.py:127` (spectrum only today). Grep for `profile_matrix`/`theta_matrix` in scripts: `scripts/nirt/baseline/inspect_baseline.py` also uses them.
**Direction:** Raise, or at least warn, when `len(order) != len(model_index)`.

### TB-18: One evaluation loads `TrainingData` up to five times `:98-99`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
98:    tr = build_arrays(cfg, split="train", **common)
99:    ev = build_arrays(cfg, split=split, **common)
```

**What breaks:** `build_arrays` calls `load_training_data(cfg)` when `data` is None (`data.py:94`). `evaluate_checkpoint` makes that happen twice. `profile_matrix` loads again (`:40`). `cold_start_baseline` loads again (`:166`) and calls `build_arrays` a fourth time (`:176-179`). `evaluate_continuous` (plus `compare_response_models` looping over dirs) repeats the pattern. Each load reads the processed parquets and splits. The train-split arrays (with relevance/warm-up joins) are rebuilt only to get `y` and `model_ids` for `marginal_baselines`.
**Evidence:** Grep of `load_training_data` in `baseline/`: `data.py:94`, `eval.py:40,166`, `continuous_eval.py:34`, `train.py:100`.
**Direction:** Load `TrainingData` once per entry point and pass `data=` through `build_arrays`, `profile_matrix` and `cold_start_baseline`; build train arrays with `use_relevance=False, use_warmup=False`.

### TB-21: Cold-start labels are clipped to [0,1]; training/eval labels are not `:190-192`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
190:        y_soft = np.clip(pd.to_numeric(sub[sub["query_id"].isin(set(qids))]["score"],
191:                                       errors="coerce").to_numpy(np.float64), 0, 1)
192:        y = (y_soft >= binary_threshold).astype(np.float64)
```

**What breaks:** `build_arrays` uses the obs `target` unclipped (`data.py:101,110`), while the cold-start path clips. With `score_kind="raw"` or `chance_correction.clip: false`, warm and cold metrics (and the continuous-head NLL at `:201`) are computed on different label definitions. NaN scores become `y=0` and give a NaN `nll` for that model. The pooled block also omits `nll` entirely.
**Evidence:** Suspected because the difference only shows when scores fall outside [0,1]. Maintainer question: are effective scores guaranteed to lie in [0,1] under every supported config?
**Direction:** Share one label-construction helper (clip + finiteness mask) between `build_arrays` and `cold_start_baseline`.

---

## src/training/nirt/baseline/diagnostics.py

### TB-10: A fully collapsed θ is reported as `collapsed: False` `:72-77`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
72:        "explained_variance_ratio": (var / var.sum()).tolist() if var.sum() else [0.0] * len(sv),
77:        "collapsed": bool(len(sv) > 1 and var[0] / var.sum() > 0.98),
```

**What breaks:** If every θ row is identical (total collapse, the exact pathology the flag exists for), the centred SVD gives all-zero singular values. `var[0]/var.sum()` is `0/0 = nan` (numpy float64, RuntimeWarning), `nan > 0.98` is False, so `collapsed=False`. At the same time `effective_rank` returns 0.0. `explained_variance_ratio` guards the zero case; `collapsed` does not.
**Evidence:** Read the two lines; the `var.sum()` guard exists at `:72` only.
**Direction:** Treat `var.sum() == 0` (or below eps) as collapsed.

---

## src/training/nirt/baseline/synthetic.py

### TB-12: Recovery reports assume a free-mode θ table `:159`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
159:    theta_hat = model.theta.weight.detach().numpy()
```

**What breaks:** `_recovery_bernoulli` and `_recovery_continuous` (`:210`) access `model.theta`, which exists only when `model_params="free"` (`model.py:89-93`). A projected-mode run raises `AttributeError`. `to_arrays` also supplies `e_m = zeros((n,1))` (`:133`), so a projected synthetic model would map every model to the same θ.
**Evidence:** `tests/helpers/baseline.py:18` pins `model_params: "free"`, so tests never reach this.
**Direction:** Guard with an explicit error ("synthetic recovery requires model_params='free'"), or use `model.theta_table(...)`.

### TB-25: Bernoulli synthetic `y_soft` is the true probability, so a `target: soft` run trains on the oracle `:126-128`
**Category:** design · **Severity:** low · **Confidence:** suspected

```python
127:        y = ys if bernoulli else (ys >= 0.5).astype(np.float32)
128:        y_soft = syn.p[qi, mi].astype(np.float32) if bernoulli else ys
```

**What breaks:** For Bernoulli worlds, `fit` with `train.target: soft` uses `y_soft` (`train.py:155,161`), which is the generating `p`, not a noisy graded score. The recovery gate would then trivially approach Bayes BCE and would not show whether the model can recover the parameters from `{0,1}` data.
**Evidence:** Suspected misuse only: `baseline_cfg` uses `target: binary` for Bernoulli (`tests/helpers/baseline.py:22`).
**Direction:** Set `y_soft = ys` for Bernoulli worlds (keep `p` on `Synthetic` for the Bayes reference), or document that soft target is invalid there.

### TB-26: The Normal "well-specified" world is censored to [0,1] `:84-87`
**Category:** design · **Severity:** low · **Confidence:** suspected

```python
86:        y = z + sigma[:, None] * rng.standard_normal(z.shape)
108:        y=np.clip(y, 0.0, 1.0).astype(np.float32), pi_true=pi_true,
```

**What breaks:** `z` ranges over several logits (K=4, a≈softplus+0.1, θ~N(0,1)). Clipping to [0,1] puts most mass at the boundaries, so the Normal head is fit to a censored likelihood it does not model, and `mean_corr` compares its mean to unclipped `z` (`:195`). The gate therefore tests robustness to misspecification, not recovery of the head's own parameters, contrary to the module docstring.
**Evidence:** Suspected: whether the gate still discriminates depends on the empirical fraction clipped, which the maintainer should check (for example `(y<=0)|(y>=1)` mean on `make_synthetic_continuous("normal")`).
**Direction:** Do not clip the Normal world (the Normal head has real-line support), or scale `z` into [0,1] before adding noise.

---

## src/training/nirt/baseline/model.py

### TB-13: `Output.proba()` returns `sigmoid(logit)`, which is not the predictive probability for continuous heads `:48-49`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
48:    def proba(self) -> torch.Tensor:
49:        return torch.sigmoid(self.logit)
```

**What breaks:** For the Normal head `mu = z` on the raw scale (`continuous_normal.py:43`), so `sigmoid(z)` means nothing. For ZOIB, E[Y] = `pi1 + pic·sigmoid(z)`, which is not `sigmoid(z)`. `predict_proba` (`:153-156`) routes correctly through the head. A caller reaching for `out.proba()` on a continuous checkpoint silently gets a different quantity.
**Evidence:** Grep `\.proba\(\)`: only `tests/training/nirt/baseline/test_baseline.py:33` and `test_response_heads.py:64` (Bernoulli). Latent trap, no src caller.
**Direction:** Delegate `proba()` to the response head (store it on `Output`), or rename it `bernoulli_proba()`.

### TB-19: The length head runs on every forward and never feeds the loss `:83`
**Category:** performance · **Severity:** low · **Confidence:** confirmed

```python
83:        self.length_head = LengthHead(self.query_dim, query_hidden, enabled=self.use_length_head)
151:                      length_pred=self.length_head(e) if self.use_length_head else None)
```

**What breaks:** `use_length_head` defaults to True (`:61,117`). Every training step and every `batched_forward` chunk therefore runs a `query_dim→hidden→1` MLP whose output nothing reads (`components.py:114-117`: "NOT connected to the correctness logit"). The cost is small but applies to every pass, and its parameters are stored in every checkpoint.
**Evidence:** Grep `length_pred`: no reader besides `Output` construction.
**Direction:** Default `use_length_head` to False, or skip the call outside a length-loss path.

### TB-22: A missing `r_q` becomes zeros, while the data path fills unclustered queries with uniform `1/C` `:142-143`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
142:        if self.use_relevance and r_q is None:
143:            r_q = e_q.new_zeros(e_q.shape[0], self.relevance_dim)
```

**What breaks:** Training feeds `r_q` from `_join(rel, ..., fill=1/C)` (`data.py:132`). If a relevance-trained checkpoint is evaluated or served while `load_relevance` returns None, `to_tensors` passes `r_q=None`. The gate then becomes `sigmoid(bias)` and the interaction input gets zeros, both out of distribution, with no warning.
**Evidence:** `batched_forward` passes `t["r_q"]`, which is None whenever `arr.r_q` is None (`data.py:166,194`). Suspected because it needs the relevance artifact to be missing at eval time.
**Direction:** Use the same `1/C` fill as the data path, or raise when a relevance model receives no `r_q`.

### TB-23: The warm-up blend can switch on at eval after being inactive during training `:140-141`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
140:    def forward(self, e_q, model_ref, r_q=None, neighbor_mean=None) -> Output:
141:        e = self.warmup(e_q, neighbor_mean)
```

**What breaks:** `use_warmup` is stored on the model from config (`:73`) whether or not a warm-up store existed. Relevance is disabled when `relevance_dim == 0` (`:71`), but warm-up has no equivalent guard. If `load_warmup` returned None during training (all steps pass through), the checkpoint still has `use_warmup=True`. Once the warm-up artifact exists, `build_arrays(use_warmup=model.use_warmup)` supplies `nbr` and the model blends neighbour means it was never trained with.
**Evidence:** `train.py:139,142` (nbr None → `has_n=False`); `WarmupBlender.forward` passes through on None (`components.py:49`). `meta["warmup_available"]` is recorded in provenance but never checked on load.
**Direction:** Bake `use_warmup = use_warmup and tr_arr.nbr is not None` into the model cfg before building, mirroring relevance.

---

## src/training/nirt/baseline/response_head.py

### TB-14: `ResponseHead.target` is declared and never read; the trainer derives the target independently `:53`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
53:    target: str = "soft"          # which target column the trainer feeds: {soft, binary}
```

**What breaks:** Each head declares which target column it expects (`binary` for Bernoulli, `soft` for the others). `fit` ignores that and uses `soft_target = (not bernoulli) or tcfg.target == "soft"` (`train.py:155`). The declaration is dead metadata, and a new head declaring `target="binary"` would still receive soft targets.
**Evidence:** Grep `\.target\b` in src: only `tcfg.target` at `train.py:155`.
**Direction:** Have the trainer read `head.target` (with `train.target` as an explicit Bernoulli-only override), or remove the attribute.

---

## src/training/nirt/baseline/train.py

### TB-15: A Bernoulli `target: soft` run early-stops on BCE against hard labels `:177-180`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
178:        vm = prediction_metrics(va_arr.y, val_prob)
179:        val_nll = _mean_nll(model, va, va_arr, device=tcfg.device) if not bernoulli else vm["bce"]
180:        stop = vm["bce"] if bernoulli else val_nll
```

**What breaks:** With `train.target: soft`, the optimizer minimises soft BCE on `y_soft` but model selection uses BCE on thresholded `va_arr.y`, a different objective. The best epoch and the reported `val_nll` do not match the trained loss. Class-weighted runs likewise stop on unweighted BCE, which may be intended.
**Evidence:** `soft_target` at `:155`; `bce_loss(..., target)` at `:164`.
**Direction:** When `soft_target`, stop on `log_loss(va_arr.y_soft, val_prob)`.

### TB-24: θ identifiability regularisers act on per-observation, sigmoid-bounded θ in projected mode `:166-167`
**Category:** design · **Severity:** low · **Confidence:** suspected

```python
166:            theta_all = model.theta.weight[train_rows] if model_params == "free" else out.theta_m
167:            reg = regularization(rcfg, theta_all=theta_all, a_q=out.a_q, b_q=out.b_q).to(tcfg.device)
```

**What breaks:** In projected mode `out.theta_m` is `[B, K]` with one row per observation, so the "centre mean_m θ at 0" term (`losses.py:46`) is weighted by observation frequency rather than per model. With `bound_ability=True` (the default), θ = sigmoid(·) lies in (0,1), so a zero mean is unreachable. Both `theta_center_l2` and `theta_l2` then push every pre-activation toward −∞, shrinking the ability range instead of fixing a shift convention.
**Evidence:** `model.py:128-129`, `losses.py:42-46`. The coefficients are small (1e-3/1e-4), so the practical effect is unmeasured. Maintainer question: are projected runs meant to use these regularisers at all?
**Direction:** In projected mode apply the regularisers to `theta_proj` over the unique profile embeddings (or to the pre-sigmoid values), or turn them off when `bound_ability` is set.

---

## src/training/nirt/baseline/losses.py

No findings (see TB-24 for how the trainer applies `regularization`).

## src/training/nirt/baseline/calibration.py

No findings.

## src/training/nirt/baseline/checkpoint.py

No findings in scope (unread `arch`/`dim`/`model_params` keys already recorded in `_contracts.md` §1).

## src/training/nirt/baseline/continuous_synthetic.py

No findings. The back-compat shim is still imported by `tests/training/nirt/baseline/test_continuous_{beta,normal,zoib}.py:12-15`, so it is not dead.

## src/training/nirt/baseline/__init__.py

No findings.

---

## Checked and ruled out
- **ZOIB `torch.where` gradient NaN** (`continuous_zoib.py:75-77`): the interior branch is evaluated at y clamped to 1e-6 for boundary rows. `log_prob` and its digamma gradients stay finite even at alpha = 1e-4, so the masked branch contributes 0·finite = 0.
- **Beta NLL at exact 0/1:** unreachable in training because `_interior` filters both train and val (`train.py:131-132`), and eval masks NLL with `imask`.
- **`torch.quantile` size limit** (16,777,216 elements): 512 samples × default batch 16384 = 8.39M, under the limit.
- **`class_balance_pos_weight`** divide-by-zero: both numerator and denominator are clamped ≥1.
- **`regularization` empty terms:** returns `torch.zeros(())` and is moved to the device; no stack-of-empty error.
- **Free-mode `train_rows` index** is CPU long while θ may be on CUDA: advanced indexing with a CPU index tensor is accepted by PyTorch.
- **`long["query_id"].isin(q_store._index)`** (`eval.py:170`): pandas treats a dict as list-like over its keys, so it is correct.
- **`cold_start_baseline` y/qids ordering:** `long` is already filtered to the store, so `sub[isin(qids)]` keeps the order of `qids`.
- **`build_arrays` `midx = -1`:** impossible after the `keep` filter when `model_index` is given, and every id is present when it is derived.
- **`_iccs` "easy"/"hard" ordering:** larger `b` = harder (subtracted), so the ascending argsort is correct.
- **`effective_rank` on singular values** (Roy & Vetterli) is a valid definition. The clash with `training.nirt.metrics.effective_rank` is filed as cross-cutting (A3).
- **`continuous_synthetic.py` dead-code check:** grep `continuous_synthetic` over src/scripts/tests/configs/docs → 3 test imports; kept.
- **Normal `as_probability` vs `_contracts.md`:** the contract row says clamp(mean); the code returns P(Y≥0.5). This is a doc discrepancy, and the threshold issue is covered in TB-04.
