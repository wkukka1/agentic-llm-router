# Next steps

Status 2026-09-02. Shipped: 6-member weighted ensemble, **0.758 top-1 /
0.895 top-2 / 0.953 top-3** on a frozen 400-prompt real-user eval; 0.923 top-1
on the 402 externally-labelled prompts.

## The ceiling is a labelling ceiling, and it has now been measured

200 random hand-labelled prompts were re-annotated with the stored labels
hidden. The re-annotation reproduced the stored label **77.0%** of the time,
and found a defensible *second* domain on **42.5%** of prompts.

That reframes every accuracy number in this repository. The classifier scores
0.741 top-1 under nested cross-validation, three points below the rate at which
the same annotator reproduces their own labels — and a second person would
agree less than one person agrees with themselves, so 0.770 is an upper bound
on the upper bound. Splitting the errors accordingly:

| | |
|---|---|
| model picks a label the re-annotation also accepts | 47% of errors |
| model picks a label no annotation accepts | 53% of errors |
| top-1 lands on *some* defensible label | 0.855 |
| top-2 contains a defensible label | 0.950 |
| top-1 on single-domain prompts | 0.826 |
| top-1 on dual-domain prompts | 0.588 |

The whole accuracy deficit lives in the dual-domain 42.5%. Nothing about a
larger encoder addresses a target that is genuinely two-valued.

`data/handlabelled/reannotation_200.parquet` holds the re-annotation.

**Corollary: stop optimising single-label top-1 on this taxonomy.** The
remaining work is in the output contract, not the model — see `shortlist_mass`
in `router/inference.py`, which sizes the candidate list by probability mass and
beats a fixed pair on both in-house and external data while emitting fewer
labels.

## Overfitting: audited, both heads clean

`router overfit` runs five checks over both label sets. Latest:

| | domain | task |
|---|---|---|
| real vs shuffled labels | 0.695 / 0.123 | 0.790 / 0.382 |
| permutation sd above null | 92× | 21× |
| train–test gap | +0.166 | +0.088 |
| learning curve, last step | **+0.001** | **+0.035** |
| dropping >0.95 near-twins | −0.008 | −0.009 |

Two things to read off that. The permutation check is the load-bearing one and
both heads pass it by a wide margin — destroying the labels destroys the skill,
so nothing is leaking. And the learning curves say opposite things: **domain has
plateaued and task has not.** More domain labels are close to worthless; more
task labels are the single highest-value thing left.

One trap worth naming: the yardstick for the permutation check is the shuffled
score, *not* the majority-class rate. A model fitted with
`class_weight="balanced"` cannot fall back on the majority class, so on shuffled
labels it lands below the majority rate. Scored against the majority rate that
looks suspicious; it is the opposite.

## Read this before running another experiment

**Seed variance is ±3.4 points (95% CI).** Five seeds of one config spanned
63.5–67.8%. Any single-run difference below ~3 points is noise. Run 3–5 seeds,
or use the deterministic frozen-encoder members, before believing an effect.

**The eval set is frozen and must stay frozen.**
`data/handlabelled/eval_frozen.parquet`, matched by prompt text. It was
previously resampled per build, which silently made runs incomparable and
produced a learning-curve extrapolation that was wrong by orders of magnitude.

**Select on validation, never on test.** Searching ensemble combinations against
test read 76.0% vs 73.75% honest — a 2.25-point illusion.

## The 93% question

93% top-1 across all traffic is **not reachable** on this taxonomy, and the
re-annotation above explains why: the labels themselves only reproduce at 77%.
Measured on the frozen eval, error decays as `n^-0.066` in labelled data, which
puts 93% at ~10^13 examples. Model scale, hyperparameters, kNN, synthetic data,
feature concatenation, a doubled ensemble and stacking were each worth ≤2 points
or nothing.

A 95% **top-2** target is met three ways, all measured:

| configuration | result |
|---|---|
| top-2 over the most-confident 50% of traffic | 0.980 |
| top-2 over the most-confident 70% | 0.961 |
| top-3 over all traffic | 0.953 |

What is *not* available is 95% top-1 across all traffic (0.758). If the consumer
can accept a shortlist or defer the least-confident tail, the target is met
today. Best of all: pass `p.distribution` and let the consumer choose its own
operating point.

## The task classifier: retrained on real prompts, now shipping-grade

Dolly-15k gets it to 0.822 top-1 under nested cross-validation and **0.700 on
real prompts, below the 0.729 of always predicting `answer`.** 1,000 real
prompts were hand-labelled with the six task types instead:

| trained on | real-prompt top-1 |
|---|---|
| always predict `answer` | 0.729 |
| 14,776 Dolly rows | 0.700 |
| 1,000 real rows, three encoders | 0.787 |
| **+ a word/char tf-idf member** | **0.828 top-1 / 0.952 top-2** |

+0.099 over the baseline, 95% CI [+0.067, +0.131]. Mixing Dolly *in* hurts at
every ratio — 0.460 at equal weight, still 0.739 when real rows are weighted
fifteen to one. `router/tasktype.py` carries the full table and the failed
prior-correction attempt.

Real traffic is `answer` 73%, `create` 17%, and a long tail; `extract` is 0.3%.
So macro-F1 0.522 is the honest number for the rare classes, and the split the
router can act on today is answer-shaped vs produce-shaped work.

## Priorities

**1. Get a *second person* on those 200 prompts.** The self-consistency check is
now done properly (0.770, labels hidden) and it says further modelling work is
close to wasted. What it cannot say is how much lower true inter-annotator
agreement is, and that number sets the honest reporting ceiling for the whole
project. `data/handlabelled/reannotation_200.parquet` has the prompts and both
existing label sets; a third column from someone else finishes the measurement.
~2 hours.

**2. Split `create`.** Done, but not the way this file predicted. The prediction
was that `create` conflated prose with code; a regex for code-generation intent
over all 1,240 labelled prompts matches **six**, one of them genuine. Code
generation is as absent from this traffic as `extract` is.

What `create` actually held was media generation (45 rows), text editing (52)
and new writing (102). Measured, `media` separates and `edit` does not, and
taking both costs 2.2 points of top-1:

| taxonomy | top-1 | macro-F1 | collapsed vs 6-class |
|---|---|---|---|
| six classes | 0.828 | 0.522 | — |
| eight (create/edit/media) | 0.774 | 0.492 | −0.022 [−0.036, −0.008] **significant** |
| **seven (media only)** | **0.802** | **0.524** | −0.013 [−0.026, +0.000] not significant |

`media` is now the seventh task type (F1 0.725, recall 0.841). It is the one
task that does not route to a language model at all. `edit` survives in the
`task_detail` column for a larger sample later.

**3. Rare task types: measured, and the answer is mostly "don't".** 240 prompts
were mined from the unlabelled pool by scoring for the rare classes and
hand-labelling the top candidates (`data/handlabelled/real_tasks_mined.parquet`).
Mining precision: `ideate` 47/60, `classify` 31/60, `summarize` 16/60,
**`extract` 1/60**. Hunting the entire unlabelled pool for `extract` found one
example — the class is not hard, it is absent, and it should probably leave the
label space. Adding the mined rows to training moves macro-F1 +0.025 and top-1
-0.013, neither significant under a paired bootstrap, so they are off by default
(`build_task_dataset(include_mined=True)` turns them on).

**4. Fix `law_politics` recall (0.381).** Precision is 0.889 — it is simply too
cautious, with only ~106 real examples. Synthetic data raised its F1 from 0.500
to 0.700 in isolation but cost accuracy elsewhere when applied to four classes
at once; applying it to this class alone was neutral overall. Real examples
(r/legaladvice, policy forums) are the better fix.

**5. Consider whether `meta_other` should be split.** 12% of traffic, precision
0.585 — it absorbs greetings, jailbreaks, questions about the assistant, and
context-free follow-ups. If the router treats those differently, they need
separate labels.

## Known limits

- **Single-annotator labels**, unaudited. This is the load-bearing caveat.
- **LMArena flags are model-assisted**, and supply 3 of 10 domains in stage 1.
- **Latency 58 ms** — three encoders, two of them large. A single frozen
  `bge-base` head gives 0.680 at ~10 ms if that trade is better.
- **Apple Silicon (MPS), single-prompt, unbatched.** A CPU server will differ.
- **Benchmark scores (0.83–0.99) are not evidence of anything.** v1 scored 0.91
  there and 0.47 in the wild. Read the real-prompt column only.

## Things measured and not worth repeating

| tried | result |
|---|---|
| bigger encoders for fine-tuning (v1: 22M→109M→149M) | +0.8 then negative |
| hyperparameter sweeps | ±1 point |
| kNN alone | 0.673 (below ensemble) |
| synthetic data, 4 classes | net −1.75 (within noise); helped `law_politics` only |
| multi-encoder feature concatenation | 0.733, 5× cost |
| standardising embeddings before PCA | −7 points |
| Kaggle query-domain dataset | wrong taxonomy; collapses to one class |
| Dolly-15k as task-type training data | 0.700 real vs 0.729 for a constant; harmful when mixed in |
| EM prior correction for the Dolly/real shift | 0.700 → 0.220; the shift is not label shift |
| doubling the ensemble, 4 encoders → 8 | 0.7378 → 0.7407 top-1 (fold sd 0.005–0.017) |
| adding a lexical (tf-idf) member to the 8 | 0.7407 → 0.7411 |
| stacked logistic regression over member probabilities | 0.715–0.737, *worse* than averaging at every C |
| longer `max_length` | no prompt in the set exceeds 256 tokens; nothing to gain |

The first three lines are one result, not three: four extra encoders from four
different pretraining families, a lexical member, and six combination rules
compared under nested cross-validation move top-1 by **+0.3 points**, less than
one fold's standard deviation. The one rule that reliably helped was per-member
temperature scaling before averaging, and it helped log loss (0.93 → 0.75)
rather than accuracy — which matters, because every threshold in
`inference.py` depends on the confidences meaning what they say.

Stacking deserves its own note: it is strictly more expressive than a weighted
average and it lost at every regularisation strength tried. With ~1,950 training
rows behind each fold's meta-learner, the extra capacity buys variance, not
signal. That is the overfitting this protocol exists to catch.
