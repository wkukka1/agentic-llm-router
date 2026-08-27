# Next steps

Status 2026-08-27. Current model: 0.674 top-1 / 0.829 top-2 on 350 held-out
real prompts.

## The one thing that matters

**Label more real prompts.** Everything else has been tried and is flat.

Evidence that data is the binding constraint:

| lever | effect |
|---|---|
| hand-labelled rows 597 → 872 | **+8 points** (0.589 → 0.674) |
| transfer learning vs single stage | +8 points |
| epochs / learning-rate sweep | ±1 point (66.3–67.4%) |
| model size 22M → 109M → 149M (v1) | +0.8, then negative |
| taxonomy 12 → 6 classes (v1) | +7, but destroys the label space |

Going from ~870 to ~3,000 labelled real prompts is the highest-value work
available. At roughly 90 labels/hour that is about 24 hours of reading, and on
the observed curve should land near 0.78–0.82.

Two things worth doing while labelling:

- **Target the weak classes.** `law_politics` (n=74) has perfect precision and
  0.33 recall — it simply has not seen enough. `humanities` (n=95) is the only
  genuinely confused class.
- **Double-label a sample.** ~200 prompts labelled twice would measure
  self-consistency. With 10 fuzzy classes the labeller's own noise may be a real
  part of the ceiling, and right now that is unmeasured.

## Also worth doing

**Ship top-2 or abstention, not bare top-1.** Top-2 is 0.829 and confidence ≥0.9
gives 0.811 over 57% of traffic. If the downstream stage accepts a ranked pair
or can defer, that is available today at no cost.

**Re-check `meta_other`.** It is 12% of real traffic and a genuine catch-all
(greetings, jailbreaks, questions about the assistant, context-free follow-ups).
If the router would treat those differently from each other, it needs splitting
— and that needs new labels.

## Known limits

- **Labels are single-annotator.** One person read each prompt once. Boundaries
  between `business_finance` / `personal_life` / `meta_other` are genuinely
  fuzzy and the labelling is not audited.
- **LMArena flags are model-assisted**, not gold human labels. They supply 3 of
  the 10 domains in stage 1.
- **`law_politics` recall is 0.33.** Do not rely on it to catch legal or
  political prompts yet.
- **Benchmark scores (0.88–0.99) mean nothing on their own.** v1 scored 0.91
  there and 0.47 in the wild. Always read the real-prompt column.
- **Latency is Apple Silicon, single-prompt, unbatched.** A CPU server will be
  slower; unmeasured.
- **BIG-bench contributes ~1,700 rows** and its prompts are synthetic. It was
  included to test whether task diversity helps; its effect has not been
  ablated separately.

## Things not to bother with

- **Bigger encoders.** v1 tested this thoroughly: the ceiling was labels.
- **Hyperparameter tuning.** Swept; worth ~1 point.
- **Standardising embeddings before PCA.** Costs ~7 points; see README.
