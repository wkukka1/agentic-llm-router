# Next steps

Status as of 2026-08-24. Concise by design — see [README](README.md) for architecture.

## What's real vs. placeholder

| Component | State |
|---|---|
| Domain head | **Trained.** 13 experiments, best `11_finetune_bge_base` 0.894 acc @ 24ms; `07_finetune_minilm` 0.886 @ 8.9ms is the better trade. **Not a routing signal on its own — see priority 0.** |
| Calibration | **Done.** Temperature fit on val; ECE 0.20 → 0.03. |
| Policy / pool / skills / orchestrator / gate / session | **Built + tested**, runs offline. |
| Difficulty | **Heuristic.** Design formula, hand-set weights. |
| Task type | **Heuristic.** Keyword rules. |
| Pool `quality` | **Invented priors.** Not measured. |
| Answer generation | Written, tested with a fake backend; **never run against a real model.** |

## Priorities

**0. ~~Validate the premise~~ — DONE, and it failed. Read this first.**
Both routing paths were tested on the same held-out preference rows against the
same ground truth (did the strong model actually win?). Neither works:

| path | ROC-AUC |
|---|---|
| direct — TF-IDF on `routing_label` | 0.551 |
| direct — TF-IDF, wide-gap subset | 0.549 |
| direct — fine-tuned MiniLM | **0.567** (best tried) |
| indirect — domain head + heuristic difficulty | **0.485 (below chance)** |

The direct path beats the indirect one by 8 AUC points, so priority 0's
*direction* was right: supervising the decision directly is better than inferring
it from a topic. But 0.567 is not deployable either, and the indirect path is
slightly *anti*-correlated with the truth and below the majority baseline — so
the 55% "cost saving" the routing evaluation reports is achieved by guessing,
not routing. Filtering to wide-gap battles did not help (0.551 → 0.549), so the
ceiling is the task, not the label construction.

Caveats, fairly: difficulty is still a heuristic, and the domain head is far out
of distribution on open user prompts. This indicts the *implementation*, not
necessarily the concept.

**Consequence: reorder everything below.**

**1. Try a cascade before any more prediction.** Run the weak model, escalate on
*its own* failure signals — low logprob, self-reported uncertainty, a verifier.
This sidesteps predicting a quantity that appears close to unpredictable, and it
is cheaper to build than any of the heads below. This is now the top priority.

**2. If you keep a predictive router, fix the model pair.** RouteLLM's reported
gains come from one *fixed* strong/weak pair, not a 50-model mixture. Pick the
two models actually deployed and label against those. The loader already supports
this via `min_tier_gap`; restricting to a single pair is a small extension.

**3. Train the difficulty regressor.** Highest leverage. Weak/strong leans on it hardest, and thresholds are currently being fit to a placeholder's output distribution. Target: LMArena `criteria_v0.1` (7 hardness flags → score in [0,1]); loader exists at `router.data.sources.arena`. Reuse the whole model/metric/experiment stack — only the head changes (regression, so swap macro-F1 for MAE/Spearman).

**4. Train the task-type head.** Same data, same stack. Weak labels: `is_code`, `math_v0.1`, `creative_writing_v0.1`. Note the flags are not mutually exclusive — multi-label is the honest framing, current code collapses by precedence.

**5. Measure pool quality on LLMRouterBench.** Until this exists, every routing threshold is unfalsifiable. Replace the priors in `configs/pool.yaml`, then re-fit `PolicyThresholds` against a real cost/quality frontier instead of quantile heuristics.

## Known gaps

- **Domain labels are academic (Dewey), not routing-shaped.** `technology` is 71% medicine and health, `science` is 43% mathematics, `cs_general` is 27% library science. The head learns shelving conventions, not "which model handles this". This is what priority 0 exists to test.
- **~6 points of accuracy is format artifact.** `question_only` ablation costs TF-IDF 6.4 points (0.810 → 0.746) and the fine-tuned model 6.3 (0.886 → 0.823) — near-identical across two model families, so it is a property of the data. Benchmark option-blocks and context headers won't exist in real traffic. Validate on an open-prompt distribution before trusting the number.
- **Single-source training data.** RouterArena only, 5.9k train rows. No open-ended prompts, no multi-turn, no code-heavy real traffic.
- **9 classes, no Religion (Dewey 2).** Absent from RouterArena; the head cannot emit it.
- **Execute path unexercised.** `--mode execute` works against a fake backend in tests. Never run against a real provider.
- **No CI.** 90 tests, nothing enforcing them.
- **Latency measured on MPS.** p50 8.9ms is Apple-Silicon-local, single-prompt, unbatched. A CPU-only server will be slower (unmeasured). `02_tfidf_logreg_wordchar` at 1.06ms is the sub-ms fallback at −7.5 points.

## Things to consider

- **Is the orchestrator worth it?** It triples cost on 8% of traffic. Nothing yet proves decomposition beats one strong call. This is the first thing to A/B once quality is measurable.
- **Cascade instead of classify?** Run weak first, escalate on low self-reported confidence. Cheaper to build, often competitive, and sidesteps the domain taxonomy entirely. Worth a baseline before investing further in heads.
- **Calibration drift.** Temperature is fit once on val. Real traffic shifts; needs periodic refit or online calibration.
- **The gate is a UX risk.** A false "please clarify" is visible on every turn; a false "looks fine" costs one call. Current bias is deliberately towards letting prompts through — revisit with real users, not intuition.
- **Skills are declared, not implemented.** `configs/skills.yaml` names tools; nothing executes them.
