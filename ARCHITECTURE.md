# Architecture

Two classifiers over the raw prompt. No LLM at inference, no network calls, no
fine-tuned weights. Everything is a frozen encoder plus a linear head.

```
                                prompt
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
      ┌───────────────┐                        ┌────────────────┐
      │  DOMAIN HEAD  │                        │   TASK HEAD    │
      │               │                        │                │
      │ 6 frozen      │                        │ 3 frozen       │
      │ encoders      │                        │ encoders       │
      │ + tf-idf SVM  │                        │ + tf-idf       │
      │      │        │                        │      │         │
      │ logistic head │                        │ logistic head  │
      │ per member    │                        │ per member     │
      │      │        │                        │      │         │
      │ weighted avg  │                        │ equal average  │
      │      │        │                        │      │         │
      │ temperature   │                        │ temperature    │
      │      │        │                        │      │         │
      │ 10 → 8 merge  │                        │                │
      │      │        │                        │                │
      │ adaptive      │                        │                │
      │ shortlist     │                        │                │
      └───────┬───────┘                        └────────┬───────┘
              │                                         │
              └────────────────┬────────────────────────┘
                               ▼
                        RouterHead
                    ┌──────────────────┐
                    │ .key             │  "medicine_health/summarize"
                    │ .should_defer    │  true if EITHER axis is unsure
                    │ .vector()        │  24 floats, named
                    └──────────────────┘
                               │
                               ▼
                   difficulty / ability model
                          (not ours)
```

## The two label spaces

**Domain — what the prompt is about.** Ten classes, optionally merged to eight
at inference:

`software_tech` `science_math` `medicine_health` `business_finance`
`law_politics` `humanities` `arts_entertainment` `language` `personal_life`
`meta_other`

Merged: `business_finance`+`law_politics` → `business_law`,
`humanities`+`arts_entertainment` → `culture`.

**Task — what the prompt asks to be done.** Seven classes:

`answer` `create` `ideate` `media` `classify` `summarize` `extract`

The axes are independent by construction. "Summarise this contract" and
"summarise this paper" share a task and differ in domain; "explain contract law"
and "draft me a contract" share a domain and differ in task. Neither is
recoverable from the other.

## Why each piece is there

**Frozen encoders, not fine-tuning.** Five seeds of one fine-tune configuration
spanned 63.5–67.8% — a 4-point swing from randomness alone, wider than most
effects worth measuring. Frozen encoders are deterministic.

**Averaging, not stacking.** A stacked meta-learner over member probabilities is
strictly more expressive and lost at every regularisation strength on the domain
head (0.715–0.737 against 0.741). With ~1,950 rows per fold the extra capacity
buys variance. It *won* on the task head's larger Dolly set, which is how we
know the protocol is measuring something real.

**Temperature scaling.** Fitted on validation by the experiment runner. Raw ECE
is 0.272; calibrated, 0.058. Every threshold downstream depends on the
confidences meaning what they say.

**Post-hoc merging, never coarse training.** Predicting ten domains and summing
into eight reaches 0.763; training directly on the eight coarse labels reaches
0.738. Coarse training discards distinctions the model could otherwise learn and
add up afterwards.

**Adaptive shortlist.** Emit domains until their probabilities account for a
threshold of the mass — one label when the model is decisive, more when it is
genuinely torn. On the 402 externally-labelled prompts, mass ≥ 0.85 puts the
truth in the shortlist 0.985 of the time using 1.40 labels, against 0.980 using
2.00 for a fixed pair. Better and cheaper.

**Pessimistic deferral.** `RouterPrediction.should_defer` is true if *either*
axis is unsure. A confident domain paired with an unsure task is not a confident
route.

## The handoff

`RouterHead.vectorise(prompts)` returns an `(n, 24)` matrix and its column
names, together, so they cannot drift apart:

| block | columns | why |
|---|---|---|
| domain distribution | 8 (or 10 unmerged) | the shape carries what the label does not — 0.45/0.44 and 0.89 have the same argmax |
| task distribution | 7 | same |
| domain confidence, margin, entropy | 3 | the summaries a difficulty model is most likely to want, computed once so every consumer computes them alike |
| domain shortlist size | 1 | how many domains the head could not separate |
| task confidence, margin, entropy | 3 | |
| prompt log-chars, log-words | 2 | the one prompt property neither head exposes |

Length is stable for a given head configuration. Changing `merge_domains`
changes it, which is why the names travel alongside.

## Results

All numbers are cross-validated on hand-labelled **real** prompts. Benchmark
scores are not reported anywhere in this repository: v1 scored 0.91 on
benchmarks and 0.47 in the wild, and that lesson cost two rebuilds.

**Domain** — 2,441 prompts, nested 5×5 CV:

| | top-1 | top-2 | top-3 |
|---|---|---|---|
| 10 classes | 0.741 | 0.899 | 0.946 |
| 8 merged | **0.763** | **0.919** | 0.965 |
| 402 external prompts (labels not ours) | 0.923 | 0.980 | — |
| adaptive shortlist, external | **0.985 at 1.40 labels** | | |

**Task** — 1,000 prompts, 7 classes, 5-fold CV. Non-real rows are training-only;
the evaluation set is real throughout:

| training data | top-1 | top-2 | macro-F1 |
|---|---|---|---|
| majority-class baseline | 0.729 | — | — |
| real labels only | 0.802 | 0.933 | 0.524 |
| **+ hand-written and generated (shipped)** | 0.793 | **0.950** | **0.581** |

The shipped configuration trades 1 point of top-1 for 1.7 of top-2 and 5.7 of
macro-F1. Neither difference is significant on 1,000 rows, but `extract` goes
from never being predicted at all to F1 0.364, and top-2 is what the router
consumes.

**Overfitting audit** (`router overfit`), both heads clean.

**These are not the product numbers.** The audit deliberately runs a *single*
encoder with a plain logistic head — no ensemble, no calibration, no merge —
because it measures the train/test *gap* and the shape of the learning curve,
and a simpler model makes both easier to read. The shipped ensemble scores
0.763 / 0.919 where the audit's single encoder scores 0.695.

| | domain | task |
|---|---|---|
| real vs shuffled labels | 0.695 / 0.123 | 0.790 / 0.382 |
| permutation sd above null | 92× | 21× |
| train–test gap | +0.166 | +0.088 |
| learning curve, last step | +0.001 | +0.035 |
| dropping >0.95 near-twins | −0.008 | −0.009 |

## The ceiling

Two measurements, and they disagree in a way that matters.

**Second annotator, with the rubric written down.** 300 prompts relabelled by
three agents from the ten tie-break rules in this document, stored labels
hidden: **0.817 raw agreement, Cohen's kappa 0.793** (0.830 / 0.795 on the
merged eight). `data/handlabelled/domain_second_annotator.parquet`.

**Same annotator, from memory.** 200 prompts re-annotated by the original
annotator without a written rubric: **0.770**, with a defensible second domain
on 42.5% of prompts.
`data/handlabelled/reannotation_200.parquet`.

The gap between them is the rubric. Codifying the boundaries -- every ruling
made during this project, written down -- raised agreement by 4.7 points over
re-deriving them from memory.

    second annotator, rubric written down   0.817
    same annotator, from memory             0.770
    classifier top-1, nested CV             0.741
    classifier top-2                        0.899

**This corrects an earlier conclusion.** On the 0.770 figure alone this project
reported that the classifier was at its labelling ceiling and that further
modelling was wasted. The first half of that was wrong: there is roughly **7.6
points of headroom** to what a second annotator achieves.

The second half survives. Every model lever tried failed to reach it -- eight
encoders, stacking, self-training, cross-head features, two-stage specialists,
all inside the noise band. What the evidence now points at is **label
consistency**: the 2,441 training labels predate the rubric that raised
agreement, so they encode boundary decisions made before `software_tech` /
`meta_other` was settled, before hardware was folded in, and before the merge
rules existed.

Per class, where the second annotator agreed least:

| class | n | agreement |
|---|---|---|
| `medicine_health` | 13 | 0.538 |
| `meta_other` | 38 | 0.684 |
| `software_tech` | 28 | 0.750 |
| `humanities` | 31 | 0.806 |
| `law_politics` | 26 | 0.808 |
| `language` | 18 | 0.944 |

`meta_other` is the model's worst class (F1 0.659) and the annotators'
second-worst. That is one underspecified label showing up twice, not two
separate problems.

## Known limits

- **Single-annotator training labels.** All 2,441 domain labels were written by
  one annotator, before the taxonomy rubric existed. A second annotator working
  from that rubric agrees with them 0.817 of the time, which is both the ceiling
  estimate and a measure of how much the training labels drifted from the rules
  eventually written down.
- **The task eval set is too small for its rare classes.** 1,000 random real
  prompts contain 3 `extract` and 11 `summarize`, so macro-F1 cannot be measured
  tightly on them however good the model gets.
- **Synthetic training data is separable from real text** at AUC 0.94–0.97. It
  helps where a class had nothing, and it is a stopgap, not a fix.
- **Prevalence claims are invalid.** The labelled set was sampled to cover rare
  classes, so `software_tech` is 13% of labels and ~30% of real traffic.
  Accuracy measured on it is valid; frequency is not.
- **Latency ~130 ms** on Apple Silicon, single-prompt, unbatched, two heads
  running independently. They do not share encoder passes; that is the first
  thing to change if serving cost matters more than the last point of accuracy.
