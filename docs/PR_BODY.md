## Two prompt classifiers for the router's first stage

Reads a user prompt and returns **what it's about** and **what it asks to be done**, as calibrated distributions the routing stage can threshold on. No LLM at inference, no network calls, nothing fine-tuned — frozen encoders plus linear heads, so it's deterministic and ~130 ms.

| head | classes | top-1 | top-2 | measured on |
|---|---|---|---|---|
| **domain** | 8 (merged from 10) | **0.923** | **0.980** | 402 prompts labelled outside this project |
| **task** | 6 | 0.844 | 0.967 | 1,000 real prompts, 5-fold CV |

Every number here is on **hand-labelled real chat traffic**. Benchmark accuracy is not reported anywhere in this PR: an earlier version scored 0.91 on benchmark data and 0.47 in the wild, and that mistake cost two rebuilds.

---

## What's in it

```
prompt ──┬──► DOMAIN HEAD  6 frozen encoders + tf-idf SVM → logistic head each
         │                 → weighted average → temperature → 10→8 merge
         │                 → adaptive shortlist
         │
         └──► TASK HEAD    3 frozen encoders + tf-idf → equal average → temperature
                        │
                 RouterHead
                   .key           "medicine_health/summarize"
                   .should_defer  true if EITHER axis is unsure
                   .vectorise()   (n, 24) + column names → downstream difficulty model
```

- `src/router/` — 15 modules, 3,057 lines. Label spaces, loaders, splitting, the model registry, the experiment runner, serving, and the overfitting audit.
- `data/handlabelled/` — the labels. 2,441 domain, 1,000 task, plus the annotator-agreement sets. **This is the asset**; everything else is reproducible from it.
- `ARCHITECTURE.md` — design, rationale for each component, full results, known limits.
- `EXPERIMENTS.md` — everything tried, including the failures. Read before running a new experiment.

## Design decisions worth reviewing

**Frozen encoders, not fine-tuning.** Five seeds of one fine-tune config spanned 63.5–67.8% — a 4-point swing from randomness alone, wider than most effects worth measuring. Frozen encoders give the same answer every time.

**Weighted averaging, not stacking.** A stacked meta-learner is strictly more expressive and lost at every regularisation strength on the domain head (0.715–0.737 vs 0.741). With ~1,950 rows per fold the extra capacity buys variance. It *won* on the task head's larger source, which is how we know the protocol distinguishes the two cases.

**Post-hoc merging, never coarse training.** Predicting ten domains and summing into eight reaches 0.763; training directly on eight reaches 0.738.

**Adaptive shortlist.** Emit domains until they cover a probability threshold — one label when decisive, more when torn. On the external set, mass ≥ 0.85 holds the true domain **0.985 of the time using 1.40 labels**, against 0.980 using 2.00 for a fixed pair. Better and cheaper.

## Overfitting: audited, both heads clean

`router overfit` runs five checks. The load-bearing one is label permutation — shuffle the labels, refit, and the skill must vanish:

| | domain | task |
|---|---|---|
| real vs shuffled labels | 0.695 / 0.123 | 0.790 / 0.382 |
| **permutation σ above null** | **92×** | **21×** |
| train–test gap | +0.166 | +0.088 |
| learning curve, last step | +0.001 | +0.035 |
| dropping >0.95 near-twins | −0.008 | −0.009 |

*(These deliberately use a single encoder with no ensemble or calibration — the audit measures the gap and the curve, not accuracy. The shipped ensemble is 0.763/0.919.)*

Writing tests for the audit found two bugs in the audit itself: it scored the permutation null against the majority-class rate (which flags a `class_weight="balanced"` model as suspect for the property that makes it sound), and the near-duplicate check failed *open* when there was too little left to refit.

## What was tried and rejected

The full log is in `EXPERIMENTS.md`. The headline is that **nine improvement levers were tested on the domain head and every one came back neutral or negative**:

| lever | result |
|---|---|
| 8 encoders instead of 4 | +0.003, noise |
| stacking | worse at every C |
| self-training on unlabelled prompts | +0.002, noise |
| cross-head features (task → domain) | −0.005, negative |
| two-stage specialists | −0.003 (a leaky version read +0.101) |
| relabelling against a written rubric | **−0.035, significant** |
| flat-prior correction | +0.005, not significant |
| per-class coefficients | exactly 0 — optimiser found nothing |

Three of those are worth a reviewer's attention:

**The two-stage specialist is a leakage demonstration.** Fitted on all rows of its two classes it scored **+0.101, significant** — a big, clean-looking win. Refitted inside the training fold it gives −0.003. The entire effect was the specialist having seen its test rows. It was run leaky first on purpose, as a cheap upper bound.

**Relabelling made it significantly worse.** A second annotator following a written rubric agrees with the stored labels 0.817 of the time, which looked like headroom. All 2,441 labels were relabelled against the rubric (20.1% changed) and the classifier retrained: **0.930 → 0.896 on the external set**, every class down. The original labels are *better* than the rubric, judged against ground truth neither saw — the written rules are a lossy summary of a policy that was partly tacit.

**Dolly-15k for task type is worse than a constant.** It scores 0.822 on Dolly and **0.700 on real prompts**, below the 0.729 of always predicting `answer`. Mixing it into real data hurts at every ratio. Prior correction doesn't rescue it (0.700 → 0.220): the shift isn't label shift, P(x|y) moved as far as P(y).

## `ideate` merged into `answer`

Adopted, as a routing decision by the owner of the model pool rather than a statistical one. On cross-validation the head goes 0.793 → **0.844** top-1, 0.950 → **0.967** top-2, macro-F1 0.581 → **0.613**, and `answer` F1 0.88 → 0.91 — that last one showing the boundary had been *costing* genuine `answer` prompts to a distinction the head could not make. On held-out data `ideate` had precision 0.087; it was not a working class.

**Read the baseline alongside it.** Merging also moves the majority-class baseline from 0.729 to 0.798, so the head's *margin over a constant predictor* falls from +6.4 to +4.6 points. The macro-F1 and `answer`-F1 gains are real; part of the top-1 rise is simply `answer` being a bigger class now.

The trade accepted: *"give me 10 startup ideas"* and *"what's the capital of Peru"* now share a label. Every label file keeps a `task_detail` column with the pre-merge value, so re-splitting needs no relabelling.

## The handoff to the difficulty model

`RouterHead.vectorise(prompts)` returns an `(n, 24)` matrix **with its column names**, so the two can't drift apart: both full distributions, plus confidence, margin, entropy, shortlist size and prompt length.

Validated against real LMArena pairwise outcomes on 3,000 prompts:

| features | AUC predicting "both models failed" |
|---|---|
| prompt length alone | 0.527 |
| **our 24-dim vector** | **0.577** |
| full 1024-d embedding | 0.567 |
| domain label alone | 0.518 — chance |

**Send the vector, not the label** — the label alone is chance. But budget it as weak: the 24-dim summary beats the full embedding it's derived from, which means ~0.577 is close to everything available. **Prompt text does not predict difficulty.** Any difficulty model will need signal from outside the prompt — model responses, per-model history, output length.

## Known limits

- **Single-annotator training labels.** All 2,441 domain labels come from one annotator. The 402 external prompts are the only outside check, and they caught a mistake all the internal cross-validation endorsed.
- **The task head's held-out set is unrepresentative.** The 402 prompts it was validated on are 97.5% `answer`, against ~80% in random traffic — they were curated for *domain* evaluation. It generalises to unseen prompts (top-1 0.910, top-2 1.000) but that set can't measure task-type accuracy properly.
- **`create` precision is poor on held-out data — 0.033.** It labels `answer` prompts `create` far more often than it is right. Trust the head's `answer` prediction (precision 0.992); treat other predictions as suggestions and threshold `distribution` rather than taking the argmax.
- **The task eval set is too small for its rare classes.** 1,000 random prompts contain 3 `extract` and 11 `summarize`, so macro-F1 on those can't be measured tightly however good the model gets.
- **Synthetic training data is separable from real text** at AUC 0.94–0.97. It helps where a class had nothing; it's a stopgap.
- **Prevalence claims are invalid.** The labelled set was sampled to cover rare classes. Accuracy measured on it is valid; frequency is not.

## Checks

- 90 tests passing, `ruff` clean
- CI runs on Ubuntu and Windows
- Clean clone verified: 46 tracked files, 14 MB
- `artifacts/` and `data/processed/` are gitignored and regenerable
