# Next steps

Scope: the domain-classification layer only. Status as of 2026-08-26.

## State

| | |
|---|---|
| Domain head | **Trained.** Best `11_finetune_bge_base` 0.894 acc / 0.891 macro-F1. |
| Calibration | **Done.** Temperature on validation; ECE 0.20 → 0.03. |
| Feature diagnostics | **Done.** VIF / PCA / effective rank via `router diagnose`. |
| OOD validation | **Not done.** This is the gap that matters most. |

## Priorities

**1. Measure the out-of-distribution gap.** Hand-label ~200 real prompts
(LMArena, or your own traffic) with the 9 domains and score the current model on
them. Everything else is downstream of this number.

Why it is first: the `question_only` ablation costs ~6 points across two very
different model families, which means roughly six points of the 0.894 is
benchmark formatting rather than topic understanding. Val and test cannot detect
it — same 79 benchmarks, same conventions. If the model holds up on real
prompts, ship it. If it drops to ~0.65, the fine-tune bought an artifact, and a
zero-shot NLI model becomes competitive because it never learned one.
Cost: ~2 hours of labelling.

**2. Emit the distribution, not the argmax.** Top-2 is already 0.957 against a
top-1 of 0.894. Downstream consumers that take `p(domain)` as a 9-vector get
both the 95% target and strictly more information than a hard label. Requires no
retraining — `predict_proba` is already the interface.

**3. Decide whether the taxonomy is right.** Dewey is a shelving convention, not
a capability signal: `technology` is 71% medicine and health, `science` is 43%
mathematics. The largest confusion (`technology → science`) is the model giving
the *more* defensible answer and being marked wrong. Merging the confusable
classes, or switching to a capability-shaped taxonomy, would raise the ceiling
more than any modelling change.

## Known limits

- **~6 points is format artifact** (`question_only`: 0.886 → 0.823 fine-tuned)
- **Single source**, 5,878 train rows, all academic MCQ. No open-ended prompts,
  no multi-turn.
- **9 classes, no Religion** (Dewey 2 absent from RouterArena)
- **Accuracy ceiling is label noise, not capacity.** 22M → 109M gained 0.8
  points; 149M ModernBERT was worse than 22M MiniLM. More parameters will not
  help.
- **Latency measured on Apple Silicon (MPS)**, single-prompt, unbatched. A
  CPU-only server will be slower; unmeasured.
- **DeBERTa-v3 never completed** — it loads in fp16 and crashed the weighted
  loss. Fixed (loss is now fp32) but the run was not repeated.

## Things to consider

- **Don't standardise embeddings.** Costs ~7 points; see the README. The failure
  is `StandardScaler` destroying L2-normalised geometry, not PCA.
- **Don't chase top-1 accuracy.** The scaling curve is flat and the residual
  errors are ambiguous labels. Effort is better spent on priority 1.
- **Calibration drifts.** Temperature is fit once on validation. Real traffic
  shifts; it needs periodic refitting.
- **PCA at 95% variance is free** if a smaller feature vector is useful
  downstream: 384 → 286 dimensions at identical accuracy, provided you skip
  standardisation.
