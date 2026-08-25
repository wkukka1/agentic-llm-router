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

**0. Validate the premise before building on it.** The domain head assumes
`prompt → domain → capability → model`. That is two inferential leaps, and the
first one is already shaky: `technology` is 71% medicine, `science` is 43% maths,
and the format ablation costs ~6 points. Meanwhile `lmarena-ai/arena-human-preference-140k`
labels the routing decision **directly** — `winner` is which model actually won
the head-to-head, over ~140k real user prompts rather than 8.4k academic MCQs.

Train a binary strong-vs-weak head on that (the RouteLLM framing; see also
`routellm/gpt4_judge_battles`) and compare it against the domain path on the same
routing objective. Needs no new infrastructure — a source loader and a label
column; the data/model/metric/experiment layers are already label-agnostic.
Map `winner` to a strong-vs-weak target by the tier of `model_a`/`model_b`,
dropping `tie` and `both_bad` or treating them as "weak suffices".

*If it wins, the domain head becomes an optional secondary signal rather than
the backbone, and priorities 1–2 shrink accordingly.* Cheap to answer, and it
governs everything below it — so answer it first.

**1. Train the difficulty regressor.** Highest leverage. Weak/strong leans on it hardest, and thresholds are currently being fit to a placeholder's output distribution. Target: LMArena `criteria_v0.1` (7 hardness flags → score in [0,1]); loader exists at `router.data.sources.arena`. Reuse the whole model/metric/experiment stack — only the head changes (regression, so swap macro-F1 for MAE/Spearman).

**2. Train the task-type head.** Same data, same stack. Weak labels: `is_code`, `math_v0.1`, `creative_writing_v0.1`. Note the flags are not mutually exclusive — multi-label is the honest framing, current code collapses by precedence.

**3. Measure pool quality on LLMRouterBench.** Until this exists, every routing threshold is unfalsifiable. Replace the priors in `configs/pool.yaml`, then re-fit `PolicyThresholds` against a real cost/quality frontier instead of quantile heuristics.

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
