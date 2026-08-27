# Next steps

Scope: the capability-classification stage. Status 2026-08-26.

## State

| | |
|---|---|
| Capability head | **Trained.** 76.6% on real prompts, 93.1% on benchmarks, 8.9 ms. |
| Calibration | **Done.** ECE 0.027 → 0.020. |
| Real-prompt evaluation | **Done and wired in.** Every run scores per source. |
| Precision on small classes | **Weak.** math 0.49, creative_writing 0.56. |
| Translation on real prompts | **Untested.** No real training examples exist. |

## Priorities

**1. Fix precision on `math` and `creative_writing`.** Both have good recall
(0.85, 0.84) and poor precision (0.49, 0.56) — the model finds them but
over-claims them. This is a class-imbalance artifact: they are 1–2% of real
traffic and the loss is class-weighted, which trades precision for recall.
Cheapest fixes, in order: tune the class weights, then adjust the decision
threshold per class on validation, then resample. No new data required.

**2. Get real translation examples.** LMArena has no translation flag, so the
class is trained on 180 benchmark rows and has never been evaluated in the wild.
Either mine LMArena for prompts containing translation requests and hand-check a
sample, or accept that `translation` is benchmark-only and fold it into `other`
until there is data.

**3. Decide whether `other` should be split.** It is 58% of real traffic and the
lowest-recall class (0.724). Inside it are at least: factual lookup, personal
advice, analysis/explanation, and meta-questions about the assistant. If the
downstream router would treat those differently, they need separate labels — and
new annotation, since no existing source distinguishes them.

## Known limits

- **`macro-F1` on real prompts is 0.575**, well below accuracy (0.766), because
  the two small classes drag it down. Accuracy alone flatters this model.
- **LMArena flags are the ground truth**, and they are model-assisted
  annotations, not gold human labels. `is_code` firing on 34.7% of prompts is
  plausible but unaudited.
- **The 200-prompt manual evaluation used labels generated in-session**, under
  the old taxonomy. It established the v1 collapse; it has not been redone
  against the capability labels.
- **Latency is Apple-Silicon (MPS)**, single-prompt, unbatched.
- **v1 artifacts remain** under `experiments/domain/` and `artifacts/domain/`
  for the record. They are not part of the current pipeline.

## Things to consider

- **Do not standardise embeddings** (costs ~7 points; see README).
- **Top-2 is 0.961 on real prompts** against 0.766 top-1. If the next stage can
  accept a ranked pair, that gap is free accuracy.
- **More data helps, but only matching data.** A power-law fit on the v1
  benchmark curve suggested 2× data → +4 points, but that extrapolation held
  only within one distribution and did not survive contact with real prompts.
  Treat any such projection as valid only for the distribution it was fit on.
- **Scaling the model does not help.** v1 tested 22M → 109M → 149M: +0.8 points
  then negative. The ceiling was labels, not capacity.
