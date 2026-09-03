# Candidate-pool expansion -- results ledger

Auto-rendered from `artifacts/pool_expansion/ledger.json` by `router.pool_expansion.render_ledger_md`. One row per phase; the predictor (frozen ZOIB head + Phase 1 Bernoulli control) is never retuned.

`oracle_cost_saving_ceiling` = `1 - oracle_cost / best-single-model_cost` (oracle = cheapest model attaining each query's max true score). `zoib_saving_at_minus{1,3}pt` = largest frontier cost-saving vs the reference model whose accuracy is within {1,3} points of the best single model (`n/a` = no frontier point qualifies).

| phase | n_warm_models | sources | gpt4_cheapest_cost_ratio | zoib_test_nll | zoib_test_mae | zoib_mean_cal_ece | oracle_cost_saving_ceiling | zoib_saving_at_minus1pt | zoib_saving_at_minus3pt | zoib_aiq_improvement | oracle_offbest_fraction | coldstart_beats_global_mean | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 9 | routerbench | 73.0802 | 0.3883 | 0.3368 | 0.0116 | 0.9108 | 0.1473 | 0.4856 | 0.2132 | 0.9728 | n/a | frozen 9-warm-model RouterBench reference; no retraining; oracle=cheapest-at-max; pool pinned to checkpoint (9); legacy Phase-0 NIRT not re-fit; cold-start: E1 reassigned the RB cold arm warm, so re-run skips it -- E0's original 9-warm projected cold-start did NOT beat global-mean BCE (0.757 vs 0.722), per Phases 1-2 |
| E1 | 11 | routerbench | 73.0802 | 0.3722 | 0.3208 | 0.0185 | 0.9257 | 0.1123 | 0.4609 | 0.2222 | 0.9791 | n/a | un-hold the 2 RouterBench cold-start models; RB test queries bit-identical to E0; RB cold-start eval paused (remaining cold models Arena-only); legacy Phase-0 NIRT not re-fit |
| E2 | 11 | routerbench | 61.6377 | 0.4126 | 0.3509 | 0.0211 | 0.9346 | 0.1556 | 0.4666 | 0.1799 | 0.9827 | n/a | RouterBench 0shot+5shot; 5shot items = separate query_ids (:5shot) carrying the question text, shared split group; 0shot test set bit-identical to E0/E1; shot NOT featurised; legacy Phase-0 NIRT not re-fit; RB cold-start paused |
