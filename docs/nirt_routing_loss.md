# Routing-aware training (P5)

Part of the capacity workstream. The P1 diagnostic (`docs/nirt_capacity.md`) and the
P2 sweeps established that **per-cell BCE capacity does not reach the routing
arg-max** — every architecture change left routing regret flat. P5 trains the
*decision* instead of the cell.

## What it adds (`src/router/nirt/train.py`, `configs/nirt.yaml`)

- **`train.sampler: query`** — batches are whole query groups (`_grouped_batches`):
  every model row for a sampled query is in the same batch, so a within-query loss
  sees the complete per-query model list. `cell` (default) is the legacy
  per-observation shuffle, bit-identical.
- **`train.loss`** gains:
  - `pairwise` — RankNet / Bradley-Terry: `BCE(z_m − z_m', 1[y_m > y_m'])` over all
    model pairs within a query (ties skipped). Vectorised dense `(G, k, k)` fast
    path when every group has `k` models (the usual dense matrix).
  - `listwise` — ListNet top-1: `CE(softmax(z_q), softmax(y_q / listwise_tau))`.
  - `bce_pairwise` — `soft_bce + loss_aux_weight · pairwise` (keeps calibrated
    probabilities *and* sharpens the ranking).
  A grouped loss auto-forces `sampler: query`.
- **`train.val_metric: regret`** — early-stop on mean per-query routing regret
  (`_val_regret`) instead of BCE, so a ranking loss is not penalised for raising BCE.
- `train.batch_queries` (default 512), `train.loss_aux_weight` (0.5),
  `train.listwise_tau` (0.1).

All default off; `sampler: cell` + `loss: soft_bce` reproduces the legacy path
(`test_sampler_default_unchanged`).

## Results — IRT-Router 20-model suite

`model_params=free`, K=16, 2-layer head, `val_metric=regret`. Test / OOD are the
IRT-Router splits.

| run | test BCE | **test regret** | test oracle-hit\* | ood regret |
|---|--:|--:|--:|--:|
| baseline `irtrouter-nirt-2d-projected` (per-cell BCE, projected) | 0.511 | 0.146 | 0.843 | 0.122 |
| `+ model_params=free` (per-cell BCE) | 0.494 | 0.134 | 0.856 | 0.116 |
| IRT-Router paper K=25 (projected) | 0.499 | 0.136 | 0.854 | 0.124 |
| **`loss=pairwise`** | 0.56 | **0.127** | **0.863** | 0.123 |
| `loss=listwise` | 0.58 | 0.126 | 0.864 | 0.128 |
| **`loss=bce_pairwise`** (w=1) | **0.498** | 0.132 | 0.859 | **0.121** |

- **Pairwise / listwise give the lowest routing regret (0.126–0.127) and the
  highest oracle-hit (0.863–0.864)** — ~2 points better than the baseline and
  better than the paper's K=25 model. This is the first lever that moves the
  routing decision.
- Pure ranking losses **wreck calibration** (BCE 0.56–0.58) — expected, the loss
  has no probability-scale term.
- **`bce_pairwise` is the practical pick**: BCE 0.498 (as good as the best
  per-cell run) *and* regret 0.132 / best OOD regret 0.121.

## Recommended configs

| goal | config |
|---|---|
| best routing decision | `model_params: free`, `dim: 16`, `loss: pairwise`, `sampler: query`, `val_metric: regret` |
| routing + calibrated P(correct) | as above but `loss: bce_pairwise`, `loss_aux_weight: 1–2` |

## RouterBench

As predicted by P1, routing regret is representation-independent on the
GPT-4-dominated 9-model pool: `bce_pairwise` gives test regret 0.138 vs 0.140 for
per-cell BCE — flat (`optimal_rate` nudged 0.806 → 0.820). P5 is an IRT-Router
lever, not a RouterBench one.

## Summary of the whole workstream

| lever | RouterBench BCE | IRT-Router test regret | verdict |
|---|--:|--:|---|
| baseline (K=2 projected, per-cell BCE) | 0.612 | 0.146 | — |
| P2 architecture (deeper head, `model_hidden`, interaction) | 0.595 | 0.146 | prediction only, **no routing effect** |
| P4a encoder swap (`bge-small`) | 0.605 | 0.138 | **null** on both suites |
| **`model_params: free`** | 0.598 | 0.134 | biggest single win, both suites |
| **P5 `loss: pairwise` / `listwise`** | 0.60 (BCE ↑) | **0.126** | best routing, calibration lost |
| **P5 `loss: bce_pairwise`** | 0.60 | 0.132 | best routing *with* calibration |

**Recommended IRT-Router routing model:** `model_params: free`, `dim: 16`,
`loss: bce_pairwise`, `sampler: query`, `val_metric: regret`. The "increase model
complexity" hypothesis (P2/P4a) did not pan out; the wins were the
parameterization (`free`) and the objective (P5).
