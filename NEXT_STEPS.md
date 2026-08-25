# Next steps

Status as of 2026-08-24. Concise by design — see [README](README.md) for architecture.

## What's real vs. placeholder

| Component | State |
|---|---|
| Domain head | **Trained.** 10 experiments, best `07_finetune_minilm` 0.886 acc / 0.885 macro-F1 @ 8.9ms. |
| Calibration | **Done.** Temperature fit on val; ECE 0.20 → 0.03. |
| Policy / pool / skills / orchestrator / gate / session | **Built + tested**, runs offline. |
| Difficulty | **Heuristic.** Design formula, hand-set weights. |
| Task type | **Heuristic.** Keyword rules. |
| Pool `quality` | **Invented priors.** Not measured. |
| Answer generation | Written, tested with a fake backend; **never run against a real model.** |

## Priorities

**1. Train the difficulty regressor.** Highest leverage. Weak/strong leans on it hardest, and thresholds are currently being fit to a placeholder's output distribution. Target: LMArena `criteria_v0.1` (7 hardness flags → score in [0,1]); loader exists at `router.data.sources.arena`. Reuse the whole model/metric/experiment stack — only the head changes (regression, so swap macro-F1 for MAE/Spearman).

**2. Train the task-type head.** Same data, same stack. Weak labels: `is_code`, `math_v0.1`, `creative_writing_v0.1`. Note the flags are not mutually exclusive — multi-label is the honest framing, current code collapses by precedence.

**3. Measure pool quality on LLMRouterBench.** Until this exists, every routing threshold is unfalsifiable. Replace the priors in `configs/pool.yaml`, then re-fit `PolicyThresholds` against a real cost/quality frontier instead of quantile heuristics.

## Known gaps

- **Domain labels are academic (Dewey), not routing-shaped.** "Which model handles this" correlates weakly with "which library shelf". The task-type axis matters more for routing; domain may end up a secondary signal.
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
