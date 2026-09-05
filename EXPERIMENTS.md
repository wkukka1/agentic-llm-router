# Experiment log

Everything measured, including — especially — what failed. Kept because the most
expensive mistake on this project was re-running something already settled, and
because several of these look obviously right until measured.

Protocol notes that apply throughout:

- **Real prompts only.** Benchmark accuracy is not evidence. v1 scored 0.91 on
  benchmark-derived data and 0.47 on real traffic.
- **Select on validation, never on test.** An ensemble search against test read
  76.0% where the honest protocol gave 75.75%.
- **Seed variance is ±3.4 points** for anything fine-tuned. Single-run
  differences below ~3 points there are noise.
- **Nested cross-validation** for anything with fitted weights or a stacker.
- **Paired bootstrap** on every difference, reported as a 95% interval.

---

## Data

| tried | result |
|---|---|
| Benchmark sets (MMLU-Pro, BIG-bench, RouterArena) as training data | 0.91 benchmark, **0.47 real**. Exam formatting is learnable and useless. Dropped entirely. |
| Hand-labelling real LMArena prompts | The project's key asset: 2,441 domain labels, 1,000 task labels. |
| Kaggle query-domain dataset | Wrong taxonomy; collapses to one class. |
| PromptTensor prompt bank | Degrades monotonically with volume, to 0.615. |
| Synthetic **domain** prompts, 4 weak classes | Net −1.75, within noise. Helped `law_politics` alone; neutral overall. |
| Synthetic **task** prompts, 4 weak classes | macro-F1 0.524 → 0.581, `extract` 0.000 → 0.364. Kept. |
| Dolly-15k for task type | 0.822 on Dolly, **0.700 on real** — below the 0.729 of a constant predictor. Rejected as training data. |
| Mixing Dolly into real task data | 0.622 at 2,000 rows, 0.460 at full size, 0.739 weighted 15:1 against. Harmful at every ratio. |
| EM prior correction for the Dolly shift | 0.700 → **0.220**. Not label shift: P(x\|y) moved as far as P(y). The true prior reaches 0.740, so the method is sound and the assumption is wrong. |
| Targeted mining of unlabelled prompts for rare task classes | `ideate` 47/60 precision, `classify` 31/60, `summarize` 16/60, **`extract` 1/60**. Mining works except where the class is genuinely absent. |
| Adding mined rows to training | macro-F1 +0.025, top-1 −0.013, neither significant. Off by default. |
| Agent-generated task prompts, split by generator | macro-F1 0.573 → 0.581, top-2 0.944 → 0.950. Kept, with the caveat below. |
| Is synthetic text separable from real? | **AUC 0.936 generated, 0.974 hand-written.** Trivially separable. A model trained on it partly learns synthetic style. It helps only where a class had nothing. |
| Agent-labelling the remaining 1,201 real prompts | **Cohen's kappa 0.819** against hand labels on a 60-prompt overlap — the first inter-annotator measurement in this project. But adding them moves nothing: see below. |

### Agent-labelled real prompts: high agreement, no gain

The task learning curve was still climbing (+0.035 over its last quarter of
data) where the domain curve had gone flat (+0.001), so more real task labels
were the highest-value work left. Six agents labelled the 1,201 real prompts
that had never been task-labelled, from the written rubric in
`docs/TASK_LABELLING_PROMPT.md`.

The labels are good. On a 60-prompt overlap labelled by hand without seeing the
agent's answer: **0.967 raw agreement, Cohen's kappa 0.819**, two disagreements
in sixty and both on the `answer`/`create` boundary.

They still did not help. Scored on the hand-labelled 1,000, agent rows in
training only:

| training data | top-1 | top-2 | macro-F1 | `extract` F1 |
|---|---|---|---|---|
| 1,000 hand labels only | 0.802 | 0.933 | 0.524 | 0.000 |
| + synthetic | 0.793 | 0.950 | 0.581 | 0.364 |
| + agent labels alone | 0.771 | 0.938 | 0.512 | 0.000 |
| + agent + synthetic | 0.783 | 0.951 | 0.587 | 0.400 |
| + agent reweighted + synthetic | 0.785 | 0.949 | 0.579 | 0.364 |

Against "+ synthetic", the shipped combination is −0.010 top-1 [−0.026, +0.006]
and +0.005 macro-F1 [−0.015, +0.026]. Neither clears zero.

Two candidate explanations, and the evidence does not separate them. The agent
set is **80.1% `answer`** against 72.9% in the hand-labelled set — the rubric
tells a torn annotator to prefer `answer`, and they did, so per-item accuracy is
high while the marginal distribution shifted. And the 1,000-prompt eval set
holds 3 `extract` and 11 `summarize`, so it cannot resolve differences this size
whatever causes them. Reweighting the agent rows to match the hand-labelled
class shares recovers the top-1 loss but not the macro-F1, so the marginal shift
is part of the story and not all of it.

**Off by default, and the retrain is why.** Cross-validation made the
combination look marginally favourable, so it was briefly defaulted on. Then the
shipped artifact was rebuilt with it and scored on the same held-out 201-row
split:

| | without agent labels | with | delta |
|---|---|---|---|
| top-1 | 0.756 | 0.741 | −0.015 |
| top-2 | 0.950 | 0.945 | −0.005 |
| macro-F1 | 0.507 | 0.484 | −0.023 |
| balanced accuracy | 0.756 | 0.729 | −0.027 |
| ECE (calibrated) | 0.085 | 0.094 | +0.008 |

Worse on every metric. "Nominally best in cross-validation, significant nowhere"
was not a good enough reason to ship, and the held-out split said so plainly.
The flag stays (`build_task_dataset(include_agent=True)`) and the labels stay
committed, because kappa 0.819 says they are good data and a larger eval set may
yet find the gain.

**Two lessons.** Per-item label quality and dataset usefulness are different
things: kappa 0.819 is near-perfect agreement and bought nothing. And a
cross-validated margin inside the noise band is not evidence — when it was
checked against a held-out split it reversed sign on all five metrics.

What the task head needs is not more labels of traffic it already reads well; it
is more *eval* rows in the classes it sees rarely.

## Encoders and heads

| tried | result |
|---|---|
| Fine-tuning transformers (MiniLM → bge-base → ModernBERT → DeBERTa) | +0.8 then negative with scale. Five seeds span 63.5–67.8. Rejected for instability. |
| Frozen encoder + logistic head | The shipped design. Deterministic. |
| Doubling the encoder pool, 4 → 8 families | 0.7378 → 0.7407. Less than one fold's sd, at 2.3× serving cost. |
| Adding a lexical tf-idf member (domain) | 0.7407 → 0.7411. |
| Adding a lexical tf-idf member (task) | **Strongest single member at 0.814 solo**, above every encoder. Task type is announced by the opening verb. |
| Missing instruction prefix on `intfloat/*` | Cost 3.25 points. A real bug; now applied automatically. |
| Longer `max_length` | No prompt in the set exceeds 256 tokens. Nothing to gain. |
| kNN over embeddings | 0.673, below the ensemble. |
| Zero-shot label similarity | Well below a trained head. |
| Multi-encoder feature concatenation | 0.733 at 5× cost. |
| PCA whitening | VIF exactly 1.000, top-2 0.860 against 0.895. Rejected. |
| Standardising embeddings before PCA | −7 points. |

## Combination rules

Nested 5×5 CV, 9 members, 2,441 real prompts.

| rule | top-1 | top-2 |
|---|---|---|
| shipped fixed weights | 0.7378 | 0.8955 |
| equal average | 0.7329 | 0.8955 |
| fitted simplex weights | 0.7337 | 0.8869 |
| temperature + equal | 0.7333 | 0.8947 |
| **temperature + fitted weights** | **0.7411** | **0.8992** |
| stacked LR, C=0.03 | 0.7272 | 0.8898 |
| stacked LR, C=1 | 0.7046 | 0.8640 |

Stacking is strictly more expressive and loses at every C — with ~1,950 rows per
fold the extra capacity buys variance. Per-member temperature scaling was the
only rule that reliably helped, and it helped log loss (0.93 → 0.75) rather than
accuracy, which matters because every threshold depends on the confidences
meaning what they say.

## Taxonomy

| tried | result |
|---|---|
| Dewey-style topic taxonomy (v1) | `technology` was 71% medicine. Abandoned. |
| Merging `science_math` + `software_tech` | Scored better; **rejected on judgement** — maths and code route to different models. |
| Merging `business_finance`+`law_politics` and `humanities`+`arts_entertainment` | 0.741 → 0.763 top-1. Kept, applied post-hoc. |
| Training on merged labels directly | 0.738 against 0.763 post-hoc. Merge at inference, never in training. |
| Hardware as its own domain | Only 31 of 2,441 prompts. Folded into `software_tech`. |
| Settling the `software_tech`/`meta_other` boundary | 41% of external errors. One ruling took combined 0.891 → 0.923. |
| Splitting `create` into prose and **code** | Predicted, then measured: **6 of 1,240 prompts** carry code-generation intent, one genuine. Code generation is absent from this traffic. |
| Splitting `create` into create / edit / **media** | Costs −0.022 top-1 [−0.036, −0.008], significant. `edit` reaches only F1 0.465. |
| Splitting **`media`** out alone | −0.013 [−0.026, +0.000], not significant; macro-F1 0.522 → 0.524; `media` reaches F1 0.725. **Kept** — the one task that does not route to a language model. |
| `media` as a binary gate instead of a class | Average precision 0.783 against 0.044 random; F1 0.712 against 0.725 as a class. Class wins on simplicity; the threshold is still available from the distribution. |

## Output contract

| tried | result |
|---|---|
| Temperature scaling | ECE 0.272 → 0.058. |
| Fixed top-2 shortlist | 0.899 in-house / 0.980 external, at 2.00 labels. |
| **Adaptive shortlist by probability mass** | mass ≥ 0.85: **0.985 external at 1.40 labels**. Better and cheaper. |
| Adaptive shortlist on the task head | Does not transfer — `class_weight="balanced"` flattens the probabilities, so mass ≥ 0.75 asks for 3.15 of 6 labels. |
| Selective prediction (abstain below a threshold) | 0.90 top-1 at 51% coverage; 0.94 at 38%. |

## Final sweep

Four things not previously tried, run before shipping.

All four came back negative. Baseline 0.7378 top-1 / 0.8955 top-2.

| tried | top-1 | vs baseline |
|---|---|---|
| Task distribution as a domain feature, scale 1 | 0.7329 | −0.005 [−0.014, +0.003] |
| Task distribution as a domain feature, scale 4 | 0.7284 | −0.009 [−0.019, +0.000] |
| Self-training, 244 pseudo-labels at conf ≥ 0.7 | 0.7395 | +0.002 [−0.003, +0.006] |
| Self-training, 940 pseudo-labels at conf ≥ 0.5 | 0.7349 | −0.003 [−0.010, +0.004] |
| Label correction, scored on the 2,395 untouched rows | 0.7386 | −0.004 [−0.014, +0.006] |
| Two-stage specialist, hard override | 0.7349 | −0.003 [−0.014, +0.008] |
| Two-stage specialist, soft 50/50 blend | 0.7378 | −0.000 [−0.009, +0.009] |

Not one interval clears zero. Three of these deserve a note.

**Cross-head features are not just neutral, they are slightly negative.** The
task distribution is free at serving time, so it was worth testing, but domain
and task really are independent — knowing a prompt asks for a summary tells you
almost nothing about what it is a summary of.

**Label correction not helping is the ceiling result again, from another angle.**
The re-annotation disagreed with 46 stored labels. Correcting them and scoring
on the 2,395 rows the correction never touched gives −0.004. If those 46 had
been *mistakes*, fixing them should have cleaned the decision boundary and
helped everywhere. It did not, because they were not mistakes — they were the
other defensible choice on an ambiguous prompt.

**The two-stage specialist is a leakage demonstration.** Fitted on all rows of
its two classes it scored **+0.101, significant** — a big, clean-looking win.
Refitted inside the training fold, where nothing it sees is scored, it gives
−0.003 hard and −0.000 soft. The entire effect was the specialist having seen
its test rows. It was run in the leaky form first deliberately, as a cheap upper
bound: an idea that cannot win with leakage cannot win without it. This one won
with leakage and lost without, which is the case the honest protocol exists for.

## Does the router vector carry difficulty signal?

The handoff to the difficulty head is a 24-dimensional vector, and whether that
is worth passing at all was never checked. Tested against real LMArena pairwise
outcomes on 28,819 English prompts (3,000 sampled), with `both_bad` read as "the
prompt was hard" and `tie` as "the prompt did not discriminate".

Baselines are the point. A vector that cannot beat prompt length is not adding
anything a difficulty model could not compute for free.

| features | `both_bad` (HARD) | `tie` (no discrimination) |
|---|---|---|
| prompt length alone | 0.538 [0.510, 0.570] | 0.457 |
| domain argmax alone (one-hot) | 0.518 [0.483, 0.546] | 0.508 |
| domain distribution alone | 0.537 [0.508, 0.571] | 0.524 |
| **full 24-dim vector** | **0.577 [0.541, 0.607]** | 0.516 [0.490, 0.543] |

Three conclusions, in order of how much they should change what gets built.

**Send the vector, not the label.** The domain argmax on its own is
indistinguishable from chance (0.518, interval spanning 0.5). The full vector
reaches 0.577 and its interval clears both baselines. Reducing the handoff to a
label would throw away all of the signal there is.

**The signal is in the uncertainty, not the classes.** Ranked by coefficient
magnitude, the columns carrying it are `task.confidence`, `domain.entropy`,
`task.entropy`, `domain.margin`, `domain.confidence` and
`domain.shortlist_size` -- every one a measure of how torn the classifiers were,
not of what they decided. Which is intuitive after the fact: a prompt that two
classifiers find ambiguous is a prompt models find hard. (The individual
coefficients are not worth reading closely -- confidence and entropy are
strongly collinear by construction, so the fit distributes weight between them
arbitrarily. Only the aggregate AUC is stable.)

**It is weak because the problem is, not because the vector is.** AUC 0.577 is
small, so the obvious next question is what the ceiling is. Measured on the same
3,000 prompts and the same target:

| features | AUC |
|---|---|
| prompt length alone (1 feature) | 0.527 [0.499, 0.558] |
| **our 24-dim handoff vector** | **0.577 [0.541, 0.607]** |
| full e5-large embedding, 1024-d | 0.567 [0.537, 0.600] |
| tf-idf bag of words | 0.508 [0.478, 0.538] |
| 24-dim + full embedding | 0.541 [0.513, 0.572] |

**The 24-dimensional summary beats the full 1024-dimensional embedding it was
derived from.** Forty times more information about the prompt does not help, and
combining the two is worse than either -- 1,048 features over 326 positives is
simply overfitting.

So ~0.577 is close to everything there is. **Prompt text does not predict
difficulty.** That is a fact about the problem, not a shortcoming of the
handoff, and it is the single most consequential thing here for anyone building
the difficulty model: it will need signal that is not in the prompt --
model responses, per-model historical performance, output length, or agreement
between actual generations. The 24 dims are worth passing because they are the
best prompt-side features available and beat the label alone (0.518, chance),
but they belong in the model as a minor input, not as its backbone.

There is nothing at all for discrimination -- the `tie` column never separates
from chance.

One caveat bounds all of it: each prompt carries a single human judgement of a
single model pair. That target is extremely noisy, which caps how high any AUC
measured against it can go. 0.577 against a noisy target is not the same as
0.577 against a clean one, and the true signal is likely somewhat higher than
this measures.

## Overfitting audit

`router overfit`. Five checks; both heads clean.

**Not the product numbers** — the audit runs one encoder with a plain logistic
head so the gap and the curve are readable. The shipped ensemble reaches
0.763 / 0.919 against the audit's 0.695.

| | domain | task |
|---|---|---|
| real vs shuffled labels | 0.695 / 0.123 | 0.790 / 0.382 |
| permutation sd above null | 92× | 21× |
| train–test gap | +0.166 | +0.088 |
| learning curve, last step | +0.001 | +0.035 |
| dropping >0.95 near-twins | −0.008 | −0.009 |

Two bugs in the audit itself, both found by writing tests for it:

- The first verdict scored the permutation null against the **majority-class
  rate**, which flags a `class_weight="balanced"` model as suspect for the very
  property that makes it sound — it cannot fall back on the majority class, so
  on shuffled labels it scores *below* that rate. The yardstick is the shuffled
  score.
- The near-duplicate check **failed open**: when dropping every twin left too
  little to refit, it silently returned "clean". It now reports NOT CHECKED.

The learning curves say opposite things, and that is the useful part. **Domain
has plateaued (+0.001) and task has not (+0.035).** More domain labels are close
to worthless; more task labels are the highest-value work remaining.

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

## Relabelling the training set against the rubric: significantly worse

The second-annotator measurement showed 0.817 agreement with the stored labels
and this was read as ~7.6 points of headroom, on the theory that the 2,441
training labels predate the rubric and drifted from it. So the whole corpus was
relabelled by twelve agents following the written rubric, changing **20.1% of
rows**. The moves were the rubric's explicit rules firing:

    medicine_health -> personal_life   43 rows   (rule 6: wellbeing is not clinical)
    meta_other       shrinks by 63              (rule 10: not a shortcut)
    personal_life    grows by 79
    software_tech -> science_math      24 rows

Scored on the 402 externally-labelled prompts -- the only target neither
labelling pass touched, written by someone outside this project:

| training labels | top-1 | top-2 | macro-F1 |
|---|---|---|---|
| **original** | **0.930** | **0.985** | **0.922** |
| rubric-relabelled | 0.896 | 0.980 | 0.878 |

**−0.035 top-1, 95% CI [−0.062, −0.008]. Significant, and every class got
worse** -- `meta_other` most (F1 0.848 → 0.722), then `science_math` (−0.058)
and `medicine_health` (−0.039), which are exactly the classes the rubric's rules
moved most.

So the hypothesis was wrong, and in an instructive way. The original labels are
not drifted or inconsistent versions of the rubric; they are **better** than the
rubric, judged against ground truth neither of them saw. The written rules are a
lossy summary of a labelling policy that was partly tacit, and enforcing the
summary destroyed the part that did not get written down.

**This also re-frames the 0.817 figure.** It is agreement between an annotator
and a *written rubric*, not a measurement of label noise, and it does not imply
7.6 points of headroom -- the direction of the disagreement matters and here it
points the wrong way. An earlier revision of this document claimed that headroom
on the strength of 0.817 alone. That claim was premature; this experiment is
what a test of it looks like.

The rubric labels stay committed as `real_prompts_rubric.parquet` with the
originals alongside in `domain_original`, because the disagreement between them
is a useful map of where the taxonomy is genuinely ambiguous. They are not used
for training.

## What actually moved the number

In order of size, across the whole project:

1. **Switching from benchmark data to hand-labelled real prompts.** 0.47 → 0.70.
2. **Fixing the missing instruction prefix.** +3.25 on one member.
3. **Settling one taxonomy boundary** (`software_tech`/`meta_other`). +3.2.
4. **Merging domains post-hoc.** +2.2.
5. **The ensemble over a single encoder.** +2.
6. Everything else combined: under a point.

Which is the summary of this log. The wins were in the data and the label space,
not in the model.
