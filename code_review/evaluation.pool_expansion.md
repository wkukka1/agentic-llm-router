# Code review: evaluation.pool_expansion

**Files reviewed:** 2 (695 lines): `__init__.py`, `battery.py` · **Context-only:** `src/evaluation/nirt/routing.py`, `src/router/provenance.py`, `configs/irt_router.yaml`
**Summary:** 5 findings: 2 correctness, 2 design, 1 performance · 4 confirmed, 1 suspected

**Status (2026-09-14): EP-01..EP-05 fixed on `cleanup/consolidate-nirt`.**
- **EP-01:** headline λ is chosen on validation and measured on test. Test-chosen values are kept as `*_test_ceiling`. The ledger's E0–E2 rows need a re-run.
- **EP-02:** provenance hashes `cfg.source_path`, the resolved config, and `cfg.path("splits")` (including `ood.json`).
- **EP-03:** fails fast.
- **EP-04:** `routing.nirt_error` is recorded and shown in the ledger notes.
- **EP-05:** prediction frames are computed once per battery.

XA-02 (the decision rule itself) is still open.

Cross-package note: the frontier and savings numbers here come from `evaluation.nirt.routing.route`, which is a different cost-aware rule from the served router's. That issue is filed as XA-02.

| ID | Location | Category | Severity | Confidence |
|----|----------|----------|----------|------------|
| EP-01 | `src/evaluation/pool_expansion/battery.py:206-218` | design | medium | confirmed |
| EP-02 | `src/evaluation/pool_expansion/battery.py:478-483` | correctness | low | confirmed |
| EP-03 | `src/evaluation/pool_expansion/battery.py:111-114` | correctness | low | suspected |
| EP-04 | `src/evaluation/pool_expansion/battery.py:318-323` | design | low | confirmed |
| EP-05 | `src/evaluation/pool_expansion/battery.py:311-317` | performance | medium | confirmed |

---

## src/evaluation/pool_expansion/battery.py

### EP-01: Headline savings pick their operating point on the test split `:206-218`
**Category:** design · **Severity:** medium · **Confidence:** confirmed

```python
208:    q = frontier[frontier["accuracy"] >= best_acc - margin]
211:    r = q.loc[q["cost_per_1k_queries"].idxmin()]
271:    lam3 = float(frontier.loc[(frontier["accuracy"] - target).abs().idxmin(), "lam"])
```

**What breaks:** `_frontier_saving` scans the λ frontier computed on the evaluation split (`split="test"` by default). It returns the cheapest point whose test accuracy is within 1 or 3 points of the best model's test accuracy. `derived_headline` likewise picks `minus3pt_lambda` from test accuracy. The ledger columns `zoib_saving_at_minus1pt` / `zoib_saving_at_minus3pt` and `docs/pool_expansion_results.md` therefore report the best λ chosen with test labels, not a λ a deployed router could have chosen in advance. The numbers are optimistic, and phase-to-phase deltas partly reflect selection noise.
**Evidence:** Read `:198-295` and `routing_block` (`:331-340`). No validation-split frontier is computed anywhere in the battery.
**Direction:** Choose λ on the validation split, report test quality and cost at that λ, and keep the test-optimal number as a labelled ceiling.

### EP-02: Provenance hashes hard-coded config and split paths, ignoring `cfg` `:478-483`
**Category:** correctness · **Severity:** low · **Confidence:** confirmed

```python
479:        f"config/{p}": _sha256(root / "configs" / p)
480:        for p in ("phase0.yaml", "phase1.yaml", "phase2.yaml")
482:    for name in ("train.json", "validation.json", "test.json", "cold_start_models.json"):
483:        artefacts[f"splits/{name}"] = _sha256(root / "data" / "splits" / name)
```

**What breaks:** A battery run under `configs/irt_router.yaml` (whose `paths.splits` is `data/splits/irt_router`) records hashes of the RouterBench split files, or `None`, not the splits it actually used. It also never hashes `ood.json` or the config that was passed in. The provenance block exists to make ledger rows reproducible, and for any non-default config it points at the wrong artefacts.
**Evidence:** Read `:474-504`. `configs/irt_router.yaml:35` sets `splits: data/splits/irt_router`.
**Direction:** Hash `cfg.path("splits") / name` for every split file present, plus the config file `cfg` was loaded from.

### EP-03: A pool model with no evaluation observations empties the matrices, and `costs.max()` then crashes `:111-114`
**Category:** correctness · **Severity:** low · **Confidence:** suspected

```python
111:    costs = np.array([r["mean_cost_per_1k"] for r in rows])
113:    max_cost = costs.max()
```

**What breaks:** The pool is pinned from the ZOIB checkpoint's `model_index` (`:534-536`). `dense_matrices` keeps only queries where every pool model has a target and a cost (`evaluation/nirt/routing.py:40`). If any pinned model has no observations on the split (for example after a data rebuild or a source change), every row is dropped. Then `true_df[m] >= 0.5` averages an empty column to NaN, and `max()` fails. On RouterBench, `costs` is never empty because the columns survive, but every value is NaN, so `max_cost` is NaN, `cheap_mask` is all `False`, and the pool description is silently all NaN.
**Evidence:** Traced `pool_description` → `eval_matrices` → `dense_matrices`. Whether a current checkpoint's pool lacks test data was not checked.
**Question for you:** can the pinned checkpoint pool contain a model absent from the current split? If so, fail fast with the list of missing models.

### EP-04: A failing NIRT run is swallowed with no record `:318-323`
**Category:** design · **Severity:** low · **Confidence:** confirmed

```python
321:        except Exception as exc:  # noqa: BLE001 - NIRT run is optional context
322:            preds[f"NIRT ({nirt_run})"] = None
323:            _nirt_err = str(exc)
```

**What breaks:** `_nirt_err` is assigned but never used. When the NIRT checkpoint fails to load or score (for example the query-features dimension mismatch in RR-01 or EN-01), its policy row simply disappears from `routing.policies` and `battery.json`. Nothing says it was attempted or why it failed, and the ledger looks complete.
**Evidence:** Grep for `_nirt_err` finds only this assignment.
**Direction:** Store the error in the routing block (`"nirt_error": str(exc)`) and surface it in the ledger notes.

### EP-05: Every routing and ablation pass reloads three checkpoints and rescores the whole split `:311-317`
**Category:** performance · **Severity:** medium · **Confidence:** confirmed

```python
312:        "baseline NIRT (Bernoulli)": align(_pred_matrix(bernoulli, cfg, split, "proba"), true_df),
313:        "ZOIB  E[Y]": align(_pred_matrix(zoib, cfg, split, "mean"), true_df),
```

**What breaks:** Prediction matrices don't depend on the pool, because `align` only selects columns. Yet `routing_block` recomputes all of them: Bernoulli, ZOIB mean, ZOIB lower bound, and optionally NIRT. Each involves a checkpoint load and full-split inference. `run_battery` calls `routing_block` once, plus once for the all-removed ablation, plus once per added model for leave-one-out (`:446-452`), and `prediction_quality` and `cold_start_block` load the same checkpoints again. With 5 added models, that is 7 × (3–4) checkpoint loads and full-split forwards.
**Evidence:** Read `:298-349`, `:387-456` and `:528-552`.
**Direction:** Compute the prediction matrices once in `run_battery`, and pass them into `routing_block` and the ablations as data.

---

## src/evaluation/pool_expansion/__init__.py

No findings.

---

## Checked and ruled out
- **`read_text()` without an encoding (`:491`, `:619`):** the ledger is written with `json.dumps` (ASCII-escaped by default), and the phase configs contain no non-ASCII bytes (grep).
- **"Accuracy" as `true >= 0.5` on the chance-corrected graded target:** a definitional choice, applied the same way everywhere in the battery.
- **`ablation_pools` rebuilding `set(added)` per element:** negligible at pool sizes of about 20.
- **`gpt4_cheapest_cost_ratio` with a zero-cost model:** guarded, returns `None`.
