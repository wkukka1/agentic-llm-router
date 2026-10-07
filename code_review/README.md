> **2026-09-22: this review has been consolidated into [`known_bugs/`](../known_bugs/README.md), which is now the single source of truth for open bug status.** Every finding below was re-verified against current source (not just trusted from commit titles): most of the high/notable-medium findings in §2 and the §5 triage checklist are now **fixed** (see recent git log); every finding still open has either been cross-linked to a matching pre-existing `known_bugs/` entry or filed as a new one there. This document is kept as the narrative record (file:line evidence, severity rationale) of the 2026-09-16 delta review — check `known_bugs/` for current status, not this file.

# Code review of `src/` — index (2026-09-16)

This is a **delta review**: it covers the 11 packages/files touched since the
2026-09-14 full review (commit `ecf4548`, whose 215 findings were triaged and
largely fixed the same day — see git log). Packages untouched since then
(`router.data`, `router.embeddings`, `router.execution`, `router.nirt`,
`router.tracing`, `training.data`, `training.models`, `training.retrieval`,
`training.taxonomy`, `training.prompt_decomposition`, `evaluation` top-level,
`evaluation.baselines`, `evaluation.data`, `evaluation.judge`,
`evaluation.pool_expansion`, `evaluation.prompt_decomposition`,
`decompose.classifiers`/`decompose.embedders`) were **not** re-reviewed here;
their 2026-09-14 findings still apply as of that date. Each reviewer worked
against this repo's `docs/*.md` (architecture, routing_interface,
agentic_router, nirt_model, baseline_control_arm, prompt_decomposition) as the
intent reference, cited alongside file:line evidence. No source file was
changed by this review.

**Totals:** 52 findings. 5 are high severity, 22 medium, 25 low. 30 are
confirmed and 22 are plausible.

- **Confirmed** means the path was traced end to end (several probe- or
  test-verified, noted per finding).
- **Plausible** means the defect is real given the code as read, but wasn't
  independently executed to prove the triggering input reaches it.
- **Severity:** high is wrong results or a crash on a real path, medium is
  wrong under a plausible config, low is latent/cosmetic.

---

## 1. Counts

| Doc | Corr. conf | Corr. susp | Design conf | Design susp | Perf conf | Perf susp | Conv. | Total |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| [router.md](router.md) | 2 | 0 | 1 | 1 | 0 | 0 | 0 | 4 |
| [router.llm.md](router.llm.md) | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 4 |
| [router.models.md](router.models.md) | 2 | 2 | 2 | 0 | 1 | 0 | 0 | 7 |
| [router.routing.md](router.routing.md) | 2 | 0 | 1 | 2 | 0 | 0 | 0 | 5 |
| [router.agentic.md](router.agentic.md) | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 3 |
| [training.md](training.md) | 3 | 0 | 1 | 0 | 0 | 0 | 2 | 6 |
| [training.trainers.md](training.trainers.md) | 2 | 0 | 2 | 1 | 0 | 0 | 0 | 5 |
| [training.nirt.md](training.nirt.md) | 1 | 1 | 1 | 2 | 0 | 0 | 0 | 5 |
| [decompose.md](decompose.md) | 1 | 0 | 1 | 1 | 2 | 0 | 0 | 5 |
| [evaluation.nirt.md](evaluation.nirt.md) | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 4 |
| [evaluation.routing.md](evaluation.routing.md) | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 4 |
| **Total** | **18** | **7** | **12** | **8** | **3** | **1** | **3** | **52** |

---

## 2. Top findings to look at first

Sorted by how much they can distort results or silently misbehave in
production. The five high-severity ones first, then notable mediums.

1. **[TP-01](training.md)** *(high, confirmed)*: `TrainingRun.training_metrics` is deterministically `loss=0.0, accuracy=0.0, bce=0.0, auc=0.0, spearman=0.0` for every real trainer — `_extract_metrics` looks for bare `"loss"` but both `NIRTTrainer` and `BaselineNIRTTrainer` only ever write `train_loss` (and never compute train-side accuracy/bce/auc/spearman at all). Not an edge case — this is the primary path, and the test suite only checks `training_time > 0`.
2. **[TT-01](training.trainers.md)** *(high, confirmed)*: `MLPRouterTrainer` persists a `RouterModelArtifact(router_model_name="mlp")`, but `MLPRouter` has no `from_artifact` classmethod, so `RouterModelFactory.create()` raises `TypeError` on load. The artifact-driven path this trainer exists to enable is a dead end; both its own test and the pipeline test dodge this by hand-rebuilding the model instead of going through the factory.
3. **[RL-01](router.llm.md)** *(high, confirmed)*: `LangChainAdapter.complete`/`.stream` — "the one `ProviderAdapter` that actually reaches a network today" per its own docstring — accept `**generation_kwargs` but never forward them to LangChain's `.invoke()`/`.stream()`. Any caller passing `temperature=`, `max_tokens=`, or `stop=` has it silently swallowed on a real, billed API call. Pre-existing gap, not a new regression, but it survived the adapter migration untouched.
4. **[EN-01](evaluation.nirt.md)** *(high, confirmed)*: `oracle_choice`'s canonical tie-break can select a below-max-score model as "oracle" when cost is `NaN` for every tied-max model, because `np.lexsort`/`np.sort` treat `NaN` as larger than `inf`. Reproduced directly (`true=[[0.9,0.9,0.5]], cost=[[nan,nan,0.01]]` → picks the 0.5-scoring model). This corrupts ground-truth oracle labels used elsewhere, not just a display bug — see ER-02 below for the caller-side half.
5. **[RM-01](router.models.md)** *(high, plausible)*: `LocalArtifactStore.save()` writes `model.pt` then `run.json` non-atomically, in two separate operations, with no temp-file/rename step. A process killed between the writes leaves a directory that `exists()`/`load()` report as valid but with every provenance field silently defaulted to empty — the same bug class as the 2026-09-14 `router.embeddings` finding ("a killed rebuild leaves a store that loads... with the old complete manifest"), recurring in a sibling store.

**Notable mediums:**

6. **[RC-01](router.md)**: `Router.route()`'s own docstring says `max_latency`/`latency_weight` are applied to scored candidates, but `ModelScore.expected_latency` is never set anywhere on the live path — it's a permanent silent no-op. No latency signal source exists anywhere in the package.
7. **[RR-01](router.routing.md)**: `RoutingResult.fallback` — the flag the class's own docstring says callers "must check before dispatching" — is computed from a cost-blind copy of the utility matrix, separate from the one `routing_decision` actually selects on. A non-finite `lam` can force a meaningless argmax pick while `fallback` still reports `False`.
8. **[RC-02](router.md) / [RR-03](router.routing.md)**: two independent reviewers flagged the same breakage — this branch's `Router` → `RouterModel` rename in `routing/base.py` left `src/router/tools/router_tool.py` importing the old `Router` name (harmless at runtime under `from __future__ import annotations`, but breaks static type-checking and is now ambiguous with the new top-level `router.router.Router`).
9. **[DC-01](decompose.md)**: `default_prompt_decomposer`'s `artifact_paths` dict is checked with four bare `if "key" in paths` branches and nothing else — a misspelled key (e.g. `"domains"`) is silently dropped, returning a valid-looking decomposer missing that classifier.
10. **[TN-01](training.nirt.md)**: the shared `_nirt_cfg_from_config`/`_phase1_cfg_from_config` helpers use `setdefault` on a sub-dict, so an explicit `"train": None` in `hyperparameters` (a real YAML shape both wrapped `fit()` functions already defend against with `... or {}`) crashes with `TypeError`/`AttributeError` instead of applying the override.
11. **[ER-01](evaluation.routing.md) / [ER-02](evaluation.routing.md)**: `compare_routing_strategies`'s "hard oracle" row uses a hand-rolled tie-break instead of the canonical `oracle_choice`, and neither `oracle_labels` nor `routing_evaluation` validate that `cost` is dense before calling it — the caller-side gap that lets EN-01 trigger silently.
12. **[RA-01](router.agentic.md)**: the migration from `agentic.llm_clients.LLMResponse` to `router.llm.client.LLMResponse` silently drops the `raw`/`meta` fields the old dataclass carried; nothing reads them today, but the loss is undocumented.
13. **[RM-03](router.models.md) / [TT-01](training.trainers.md)**: pattern across all three trainers — `NIRTTrainer`'s artifacts never populate `training_dataset_version`/`created_at` (round-trips through a `run.json` never designed to carry those keys), consistent with `MLPRouterTrainer`'s equivalent gap (TN-05) — likely an accepted provenance gap across the whole `TrainingHarness` path rather than one-off bugs, worth a single decision rather than three separate fixes.

---

## 3. One line per doc

- **[router.md](router.md):** top-level `Router` facade (`docs/architecture.md`). Latency is a documented-but-wired-nowhere knob; no-registry mode makes capability constraints unconditionally unsatisfiable.
- **[router.llm.md](router.llm.md):** provider adapters (new), cost, registry. The only network-reaching adapter drops generation kwargs; `guess_provider` mis-guesses real entries in `configs/model_registry.yaml`.
- **[router.models.md](router.models.md):** artifact store/registry (`docs/architecture.md`). Non-atomic checkpoint writes can leave silently-blank provenance; `NIRTTrainer` never populates dataset-version/created-at in production.
- **[router.routing.md](router.routing.md):** `RouterModel`/`NIRTRouter`/`MLPRouter` (`docs/routing_interface.md`). The fallback flag callers are told to check doesn't see cost/λ-driven fallbacks; `cost_model` is plumbed through but never consulted.
- **[router.agentic.md](router.agentic.md):** agentic orchestration on the new LLM adapters (`docs/agentic_router.md`). Clean migration — none of the four previously-known bugs recurred; only an undocumented field loss and stale docstrings.
- **[training.md](training.md):** `TrainingHarness`/`TrainingJob`/`TrainingRun` pipeline. Training-side metrics are always zero for every real trainer; `TrainingJob.run_id` is silently ignored.
- **[training.trainers.md](training.trainers.md):** `RouterModelTrainer` ABC + `MLPRouterTrainer`. The MLP artifact-driven load path is a dead end; seed precedence is inverted relative to the sibling trainers.
- **[training.nirt.md](training.nirt.md):** `NIRTTrainer`/`BaselineNIRTTrainer` (`docs/nirt_model.md`, `docs/baseline_control_arm.md`). Thin wrapper layer, mostly sound; one real crash on an explicit `null` config section, plus byte-for-byte duplicated helpers.
- **[decompose.md](decompose.md):** `default_prompt_decomposer` factory (`docs/prompt_decomposition.md`). Silently drops misspelled `artifact_paths` keys; duplicates encoder passes `RouterHead` already shares; isn't actually called from `Router` today.
- **[evaluation.nirt.md](evaluation.nirt.md):** `evaluation/nirt/routing.py`. The canonical oracle tie-break mis-selects under `NaN` cost at tied-max cells; the safe tie-break is opt-in, not default.
- **[evaluation.routing.md](evaluation.routing.md):** `evaluation/routing/oracle.py`. The 2026-09-14 column-order tie-break fix holds, but a second, hand-rolled tie-break in the same file reintroduces the same class of bug.

---

## 4. Cross-cutting observations

- **Stale `Router` rename fallout**: RC-02 and RR-03 are the same underlying issue (`router_tool.py` not updated for the `Router` → `RouterModel` rename), found independently by two reviewers — fix once.
- **Provenance fields empty across the board**: RM-03, TN-05, and TT-01's artifact-loadability gap all point at the same root cause — `TrainingHarness`'s three trainers were built against a `run.json`/artifact contract that doesn't carry `training_dataset_version`/`created_at`, and one router kind isn't wired into the loader factory at all. Worth one `TrainingHarness`-level fix rather than three.
- **Oracle tie-break bug has two entry points**: EN-01 (the canonical `oracle_choice`, when cost is `NaN`) and ER-01 (a second, non-canonical tie-break in `compare_routing_strategies`) are the same failure mode reachable two ways in `evaluation/`.

---

## 5. Triage checklist

- [ ] Fix `_extract_metrics`/trainer `history` key mismatch so `TrainingMetrics` reflects real training numbers (TP-01).
- [ ] Give `MLPRouter` a `from_artifact` classmethod so `MLPRouterTrainer`'s saved artifacts are actually loadable (TT-01), or document the hand-rebuild path as intentional.
- [ ] Forward `generation_kwargs` through `LangChainAdapter.complete`/`.stream` (RL-01).
- [ ] Make `oracle_choice`'s tie-break NaN-safe, and make the safe (`model_ids=`) path the default rather than opt-in (EN-01, EN-02); point `compare_routing_strategies`'s hard-oracle row at the same canonical function (ER-01).
- [ ] Decide on checkpoint-write atomicity (temp file + rename) for `LocalArtifactStore.save()` (RM-01), matching whatever gets decided for the sibling `router.embeddings` finding from 2026-09-14.
- [ ] Either wire a real latency signal or make `max_latency`/`latency_weight` explicitly a documented no-op (RC-01).
- [ ] Fix `RoutingResult.fallback` to reflect the cost-adjusted utility actually selected on, not a cost-blind copy (RR-01).
- [ ] Update `router_tool.py`'s stale `Router` import/docstring (RC-02/RR-03).
- [ ] Validate/error on unrecognized `artifact_paths` keys in `default_prompt_decomposer` instead of silently dropping them (DC-01).
- [ ] Guard the config-translation `setdefault` helpers against an explicit `None` sub-section, and de-duplicate the two byte-identical copies (TN-01, TN-02).
- [ ] Make one decision on trainer artifact provenance (`training_dataset_version`/`created_at`) and apply it across `NIRTTrainer`, `BaselineNIRTTrainer`, and `MLPRouterTrainer` together (RM-03, TN-05).

---

## Verification performed

Each package review was run by an independent agent against the current
working tree, citing `path:start-end` with quoted lines and tracing each
finding's triggering path before marking it confirmed or plausible; several
findings were probe-verified against real functions (EN-01's NaN reproduction,
`router.agentic`'s full 47-test pass against the migrated adapters). No source
file was changed. This index does not re-run a repo-wide test suite or import
linter pass beyond what individual reviewers report inline — see each doc's
own evidence for what was executed.
