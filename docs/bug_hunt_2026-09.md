# Bug hunt, 2026-09-24 / 25: findings report

Written for: engineers who will fix these bugs or rely on the affected code. The per-bug detail
(evidence, failure scenario, suggested fix) lives in `known_bugs/`; this is the overview, the
reasoning about what matters most, and the record of what was and was not checked. The fix plan is
[`docs/superpowers/plans/2026-09-25-bug-hunt-fixes.md`](superpowers/plans/2026-09-25-bug-hunt-fixes.md).

## Summary

Two passes over the layers `known_bugs/README.md` marked "not started" found **14 bugs**: 1 high,
3 medium, 4 medium-low or low-medium, 6 low. Twelve were reproduced by running code, one measured on the built data, and one is an
analytic error in a published conclusion. Two (IRT-Router zero-cost rows, and the taxonomy fit) were fixed in code by the user the same day, uncommitted.

| # | ID | Sev | Status | What |
|---|---|---|---|---|
| 1 | [training.prompt_decomposition.data.dataset MOD-001](../fixed_bugs/_module_level/training.prompt_decomposition.data.dataset/MOD-001-dataset-builders-import-a-module-and-a-loader-that-no-longer-exist/bug.md) | **high** | **fixed 2026-09-25** (uncommitted) | `build_real_only_dataset` / `build_task_dataset` always raise `ImportError`; the first also calls a deleted loader |
| 2 | [training.taxonomy MOD-001](../fixed_bugs/_module_level/training.taxonomy/MOD-001-clusters-fit-on-all-splits-including-ood/bug.md) | med | **fixed in code 2026-09-25** (uncommitted; data not rebuilt) | Taxonomy clusters, centroids and `r_q` are fit on val, test and held-out OOD queries |
| 3 | [training.nirt.pairwise MOD-001](../known_bugs/_module_level/training.nirt.pairwise/MOD-001-auxiliary-battles-include-cold-start-models.md) | med | open | Auxiliary Arena loss trains on battles with cold-start models |
| 4 | [training.data.loaders MOD-001](../fixed_bugs/_module_level/training.data.loaders/MOD-001-irt-router-zero-token-zero-cost-rows-priced-free/bug.md) | med | **fixed 2026-09-25** (uncommitted) | IRT-Router rows with 0 tokens and $0 cost ingested as free |
| 5 | [TEXTENCODER-001](../known_bugs/TextEncoder/TEXTENCODER-001-encode-reseeds-global-rngs.md) | med-low | open | Every `encode()` reseeds process-global RNGs, including per routed request |
| 6 | [router.embeddings MOD-001](../known_bugs/_module_level/router.embeddings/MOD-001-build-store-resume-ignores-vector-fingerprint.md) | low-med | open | `build_store` won't resume a partial build after a device or batch-size change |
| 7 | [evaluation.prompt_decomposition.length_audit MOD-001](../known_bugs/_module_level/evaluation.prompt_decomposition.length_audit/MOD-001-target-ceiling-is-a-reliability-not-a-correlation-ceiling.md) | low-med | open | A reliability (0.742) is compared against a correlation; the head is at ~67%, not ~77% |
| 8 | [training.data.pricing MOD-001](../known_bugs/_module_level/training.data.pricing/MOD-001-missing-tokens-priced-at-zero.md) | low-med | open, **needs a decision** | `fill_costs` prices unknown token counts at $0 (an existing test pins this) |
| 9 | [QUERYBANK-001](../known_bugs/QueryBank/QUERYBANK-001-similarity-weighting-inverted-on-flat-l2.md) | low | open, latent | Similarity weighting on an L2 bank weights by distance |
| 10 | [JUDGECLIENT-001](../known_bugs/JudgeClient/JUDGECLIENT-001-max-retries-zero-makes-no-attempt.md) | low | open | `max_retries=0` makes no attempt and raises |
| 11 | [training.data.chance_correction MOD-001](../known_bugs/_module_level/training.data.chance_correction/MOD-001-n-choices-below-two-gives-nan-but-flagged-corrected.md) | low | open, latent | `n_choices <= 1` gives NaN but is flagged corrected |
| 12 | [evaluation.data.stratify MOD-001](../known_bugs/_module_level/evaluation.data.stratify/MOD-001-unanimous-graded-rows-counted-discriminative.md) | low | open | A unanimous graded row is counted as discriminative |
| 13 | [training.prompt_decomposition.experiment MOD-001](../known_bugs/_module_level/training.prompt_decomposition.experiment/MOD-001-run-all-records-no-failures-when-nothing-succeeds.md) | low | open | `run_all` records no failures when nothing succeeds |
| 14 | [training.prompt_decomposition.data.sources MOD-001](../known_bugs/_module_level/training.prompt_decomposition.data.sources/MOD-001-missing-synthetic-file-raises-instead-of-warning.md) | low | open | `load_synthetic_tasks` raises where it means to warn |

## What matters most, and why

**1. The dataset builders are dead code (high).** Nothing fails at import time, no test calls the
builders, and the only caller is a CLI subcommand, so the breakage sat unnoticed through two refactors
(the subsystem was nested, then a "deslop" commit deleted `load_handlabelled` while its caller stayed).
Consequence: the shipped domain and task heads cannot be retrained from source. The rest of
`build_task_dataset` is healthy; with the import aliased it reproduces the documented split sizes.

**2. Three leaks that make evaluation numbers optimistic (medium).** The taxonomy fit sees test and
OOD queries (11,144 test and 19,850 math/code queries, 18% of the clustered points), the auxiliary
pairwise loss trains on the cold-start models' battles (4,617 battles), and the IRT-Router cost table
priced 10,565 answered calls at $0. All three are unsupervised or cost-side leaks rather than label
leaks, but each sits under a headline result: the OOD experiment, the cold-start comparison, and
cost-aware routing savings.

**3. A library that resets the host's random state (medium-low).** `TextEncoder.encode()` calls
`seed_everything`, which reseeds `random`, numpy and torch globally, on the serving path. After the
first routed request, an embedding application's own random draws repeat. It is silent, and no
routing test would ever show it.

**4. A wrong published conclusion (low-medium).** `target_ceiling` returns a variance share
(Spearman-Brown reliability, 0.742) and five docs compare it to Spearman rho. The correlation ceiling
is its square root (0.861). The head is at ~67% of the achievable correlation, or ~54% on the variance
scale (its documented R-squared is 0.399), not ~77%. This changes the "is the length head finished?"
judgement, not any runtime behaviour.

**5. The rest are small** and mostly latent: an inverted weighting nothing exercises, off-by-one
retry semantics, an unguarded divide, and two robustness gaps in the prompt-decomposition sweep.

## How they were found

- **Top-down coverage**: `known_bugs/README.md` keeps a layer-by-layer coverage table; both passes
  went through the rows marked "not started" (training data layer, taxonomy, retrieval, pairwise,
  profile building, judge, evaluation batteries, prompt decomposition).
- **Reproduce before filing**: every entry carries a status (`reproduced`, `confirmed`) and, where
  possible, a script in `known_bugs/_repros/`. Leads that could not be verified stay in the README's
  "Leads" list rather than becoming bug entries.
- **An AST scan for stale imports** (`_repros/scan_stale_imports.py`) resolved every first-party
  import in `src/`, `scripts/` and `tests/`, including function-local ones. It found bug 1 and nothing
  else. It cannot see attribute misses on a module, which is how the deleted `load_handlabelled`
  was found (by reading the call sites).

## Refuted, so nobody re-checks them

Content-hash leakage across the IRT-Router presplit (0 of 38,031 groups straddle); the pool-expansion
battery mixing accuracy scales; the `fit()` early-stopping and scheduler logic; `QueryBank`
group-exclusion under-fetching; a stale embedding cache in the forest experiment (the cache is keyed by
a hash of the texts); the frozen eval set diverging from the hand-labelled pool (all 400 rows match,
0 label disagreements). Details in the README's "Refuted" section.

## Corrections made while writing this

Two statements in the first drafts were wrong and are corrected in the entries: `tests/prompt_decomposition/`
does exist (it covers the helpers but no test calls either builder), and the builders' CLI subcommands
are `build-real` / `build-task`, not `build-data`. The ceiling entry's R-squared comparison was rebuilt
from the docs' own R-squared (0.399 -> ~54%) instead of squaring Spearman.

## Not covered

`evaluation/nirt/evaluate.py`, `training/nirt/baseline/{data,eval,continuous_eval,diagnostics,calibration}.py`,
`training/cli.py` and `scripts/`. The arena and cascade corpus builders were read but not run (only the
55k arena dump is cached; they read the 140k one), and the LLMRouterBench loader was read but not
exercised (no raw results on disk). Six weaker leads are listed in the README, among them a cross-corpus
split-hash inconsistency (`cascade._split_of` hashes unstripped text where `arena_corpus` strips) that
matters only if a head is ever scored across the two corpora.
