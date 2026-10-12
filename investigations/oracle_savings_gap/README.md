# Oracle savings gap

Find the prompts where the oracle is much cheaper than our router (or the router loses quality the oracle keeps), see which models are involved, and read the prompts for patterns.

The oracle is the baseline. It is **evaluation-only**: it is built from observed outcomes and must never feed back into training, features or candidate selection.

## Definitions

Per test query, over the RouterBench pool (11 models, 0-shot items only, so every model is observed on every query):

- **Oracle pick**: the cheapest model whose observed score is within `--tol` (default 0.02) of the best score. Cost ties go to the smallest model id.
- **Router pick**: `argmax_m pred - lam * C(m)`, with `C(m)` the per-model mean cost (`evaluation.nirt.routing.route`).
- **Failure type**
  - `overspend`: the router's pick is within `tol` of the best, but the oracle's pick is at least `--min-savings` (default 25%) cheaper than it.
  - `quality_loss`: the router's pick scores more than `tol` below the best, whatever it cost.
  - `ok`: neither.
- `savings_gap` = router cost - oracle cost (negative on a `quality_loss` where the router picked the cheaper model). `savings_gap_frac` is that over the router's cost.
- `n_good_models` = how many models scored within `tol` of the best. High means the prompt is easy (or lucky, see caveats); 1 means only one model got it.
- `failed_by` = `none`, a router name, or names joined with `+`.

## Run

```
python investigations/oracle_savings_gap/build_dataset.py --zoib artifacts/phase2/zoib_retrained                 # lam=0, quality-only
python investigations/oracle_savings_gap/build_dataset.py --zoib artifacts/phase2/zoib_retrained --lam 50 --out investigations/oracle_savings_gap/output/lam50
pytest investigations/oracle_savings_gap                                   # unit tests for gap.py
```

Options: `--nirt-run` (default `nirt-2d-projected`), `--zoib <checkpoint>` (adds a ZOIB E[Y] router; omit for NIRT only), `--tol`, `--min-savings`, `--include-multishot`, `--split`, `--per-group`, `--max-prompts`.

## Output (`output/`, and `output/lam50/`)

| file | what |
|---|---|
| `gap_all.parquet` | every query, including those no router failed (use as the contrast set) |
| `gap_dataset.parquet` / `.csv` | queries at least one router failed on; prompt text, dataset, metric, both picks, costs, scores, failure type |
| `by_model.csv` | oracle model x router model x failure type, with counts and total savings gap |
| `failure_rates_by_dataset.csv`, `failure_rates_by_metric.csv` | failure rates per group |
| `reading_sample.md` | failed prompts grouped by router / failure / oracle model, biggest groups first, both picks shown |
| `summary.json` | settings, oracle vs router quality and cost, pick mixes, failure counts |

## Caveats

- **`lam` barely matters below ~5.** Costs are ~1e-4 to 3e-3 dollars per query, so `pred - lam * C` stays a quality argmax. At `lam=0` the NIRT run picks gpt-4-1106-preview for 96% of queries and 97% of queries are flagged; at `lam=50` the mix is diverse (yi-34b-chat 57%, claude-instant-v1 26%, gpt-4 14%) and 79% are flagged. Check the printed pick mix before reading anything into a failure rate.
- **The oracle is a hindsight upper bound and includes luck.** On multiple-choice items a cheap model can score a correct guess, and the oracle takes it. Compare failures against `n_good_models` and the `ok` rows in `gap_all.parquet` rather than reading every oracle saving as achievable.
- **Two routers: `nirt` and `zoib`.** `nirt` is `nirt-2d-projected`; `zoib` is `artifacts/phase2/zoib_retrained` (E[Y]), retrained 2026-09-19 on the current taxonomy and the same 29-model pool as the NIRT run. The original `artifacts/phase2/zoib` was trained on the old taxonomy and cannot be scored any more (relevance dim 154 vs 39); it is left untouched. The retrained head is not the same model as the original Phase-2 ZOIB (different pool and validation set), so its metrics are not comparable to the old ones.
- The two routers pick the same model on 80% of queries at `lam=0` and 74% at `lam=50`, and fail on largely the same queries (`failed_by == nirt+zoib`: 3,554 of 3,677 at `lam=0`, 2,819 at `lam=50`). The `nirt`-only and `zoib`-only rows are the small set where they differ.
- The router's `C(m)` is the eval-split per-model mean cost, the same convention as `route_eval.py` / `route_compare.py`.
