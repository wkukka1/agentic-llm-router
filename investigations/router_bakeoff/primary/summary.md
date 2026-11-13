# Router bake-off, step 1: primary split, 1 seed (2026-11-12)

Design: `docs/router_benchmark_experiment.md`. Data: IRT-Router, 20 models; train 22 011 queries.
ID test = 10 458 queries (8 datasets). OOD = 3 159 queries (ceval, commonsense_qa, gsm8k, humaneval).
Every router uses the same BERT query embeddings and the same train rows.
External routers are llmrouter-lib 0.4.0 run in `.venv-llmrouter`. Each is retrained at
λ ∈ {0, 100, 300, 1000, 3000, 10000} on the label `argmax(y − λ·C_train)`.
Our routers sweep the same λ values over one fit.

Headline metric: **macro log-cost AIQ**. Take each router's cost/quality points, add an anchor
at always-glm_4_flash (the cheapest model), and form the upper concave frontier. AIQ is the
area under that frontier on a log-cost axis, averaged over datasets.
Δ columns are paired-bootstrap 95 % CIs (300 resamples, resampled within dataset) of
the router minus `nirt (K=2, BERT profile)`. q@1 % is the mean graded score at 1 % of
gpt_4o's spend.

| router | ID AIQ | ID Δ CI | ID q@1% | OOD AIQ | OOD Δ CI | OOD q@1% |
|---|---|---|---|---|---|---|
| oracle (cost-aware) | 0.920 | | 0.913 | 0.984 | | 0.984 |
| zoib E[Y] | **0.759** | [+0.000, +0.009] | 0.702 | 0.850 | [−0.003, +0.009] | 0.862 |
| bernoulli | 0.758 | [−0.001, +0.008] | 0.707 | 0.853 | [+0.003, +0.014] | 0.859 |
| nirt K=2 (anchor) | 0.754 | — | 0.701 | 0.847 | — | 0.875 |
| irt-router K=25 (ours) | 0.754 | [−0.004, +0.005] | **0.711** | 0.851 | [−0.000, +0.010] | 0.870 |
| mlp (ours) | 0.754 | [−0.004, +0.004] | 0.696 | 0.850 | [−0.002, +0.009] | 0.873 |
| nirt elo-manifold + Δ | 0.753 | [−0.006, +0.003] | 0.699 | **0.854** | [−0.001, +0.014] | 0.873 |
| **query-blind (train mean per model, swept over λ)** | 0.751 | [−0.006, −0.001] | 0.690 | 0.843 | [−0.007, −0.001] | 0.869 |
| litellm: adaptive (bandit) | 0.751 | [−0.006, −0.001] | 0.690 | 0.843 | [−0.007, +0.000] | 0.869 |
| ext: mlp | 0.731 | [−0.027, −0.020] | 0.638 | 0.795 | [−0.062, −0.043] | 0.669 |
| knn k=5 (ours) | 0.724 | [−0.037, −0.025] | 0.678 | 0.805 | [−0.054, −0.030] | 0.779 |
| ext: mf | 0.722 | [−0.036, −0.029] | 0.611 | 0.780 | [−0.079, −0.059] | 0.628 |
| ext: knn | 0.722 | [−0.037, −0.029] | 0.617 | 0.797 | [−0.061, −0.041] | 0.675 |
| ext: svm | 0.722 | [−0.037, −0.029] | 0.608 | 0.777 | [−0.081, −0.061] | 0.622 |
| ext: graphrouter | 0.716 | [−0.042, −0.035] | 0.592 | 0.777 | [−0.081, −0.061] | 0.622 |
| ext: elo | 0.716 | [−0.042, −0.035] | 0.592 | 0.777 | [−0.081, −0.061] | 0.622 |
| always deepseek_coder | 0.716 | [−0.042, −0.035] | 0.592 | 0.777 | [−0.081, −0.061] | 0.622 |
| ext: routerdc (frozen encoder) | 0.634 | [−0.130, −0.112] | 0.588 | 0.689 | [−0.170, −0.147] | 0.662 |
| per-dataset best | 0.625 | | 0.549 | 0.777 | | 0.622 |
| always gpt_4o | 0.577 | | 0.546 | 0.572 | | 0.531 |
| random | 0.568 | | 0.547 | 0.597 | | 0.536 |

Full columns (linear AIQ, q at 0.5/2/5/100 %, regret, any-best, Spearman, NDCG@3, MSE) are in
`summary_seed0.csv`. Frontier points are in `points_seed0.csv`, per-dataset results in `per_dataset_seed0.csv`.

## Reading

**Most important: the 20-model AIQ barely measures per-query routing.** A query-blind policy
gives every query the same per-model quality (the train mean) and sweeps λ. It reaches 0.751 ID
and 0.843 OOD, within 0.004–0.011 of the best learned router. That small margin is significant
but small. So most of what separates "ours" from the llmrouter-lib routers below is that ours are
cost-aware score models while those classifiers collapse to one label, not better per-query
choices. The per-query signal shows up more clearly in the 2-model sub-study
(`routellm_summary.md`), where query-blind routing is random (APGR 0.5) and our best reach 0.75.
LiteLLM's adaptive router lands exactly on the query-blind line: its request-type classifier puts
95 % of these queries in "general".


* **Our six score-based routers are statistically tied.** On ID and OOD every Δ CI
  lies within about ±0.015 AIQ of NIRT. The only clear separation is between this
  group and everything else.
* **All learned score-based routers of ours beat every external router**, by 0.02–0.04 AIQ
  on ID and 0.04–0.08 on OOD. Almost all of the gap is at low budgets: q@1 % is
  0.70 vs 0.59–0.64 on ID and 0.87 vs 0.62–0.67 on OOD.
* **The external classifiers collapse to the majority label.** At λ=0, deepseek_coder is the best
  model on most train queries. SVM, MLP, MF, Elo and GraphRouter predict it for 100 % of
  queries; kNN does so for 98.8 %. At high λ they collapse to glm_4_flash in the same way.
  Training on "best model" labels discards the graded per-model signal our routers fit.
  This is a property of these methods' training objective, not a bug in the port.
* GraphRouter at λ ≥ 1000 picks glm_4_plus (an expensive model) on 98–100 % of queries,
  even though its train label there is mostly glm_4_flash. The scoring path is identical to
  the library's `route_batch`/`predict`. Likely cause: test edges enter the graph with zero
  performance as their edge weight. Kept as is.
* RouterDC with a frozen encoder plus a projection head is worse than always-deepseek_coder.
  This is not a fair test of RouterDC, which fine-tunes its encoder; that needs a GPU.
* At full spend (q@100 %), nothing clearly beats always-deepseek_coder (ID 0.774 vs ≤ 0.785).
  The routers' value is in the cost/quality trade-off, not in peak quality.

## Caveats

1 seed; library-default hyperparameters (no validation tuning yet); 6 λ points for the
external routers. RouterDC uses a frozen encoder. Next steps (need go-ahead): 3 seeds,
validation tuning, LODO, the RouteLLM 2-model sub-study.
