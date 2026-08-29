# agentic-llm-router — domain classifier

Classifies a prompt into one of **10 domains**, as a calibrated distribution the
next routing stage can threshold on.

Every number below is reported on **350 held-out real user prompts**, hand
labelled and never trained on. Benchmark scores are reported separately, because
this project has twice been misled by them.

## Result

Every number is on a **frozen 400-prompt real-user eval set**, hand-labelled and
never trained on. The set is committed (`data/handlabelled/eval_frozen.parquet`)
and matched by prompt text, so results from different runs are comparable.

| | |
|---|---|
| **top-1** | **0.758** |
| top-2 | 0.895 |
| top-3 | **0.953** |
| macro-F1 | 0.744 |
| ECE (raw → calibrated) | 0.272 → **0.058** |
| latency | ~150 ms (6 encoders; accuracy prioritised over speed) |

Calibrated confidence is informative, so **abstention works**:

| traffic covered | top-1 |
|---|---|
| coverage | top-1 | top-2 |
|---|---|---|
| 40% | **0.931** | **1.000** |
| 50% | 0.915 | **0.980** |
| 60% | 0.896 | **0.975** |
| 70% | 0.864 | **0.961** |
| 100% | 0.758 | 0.895 |

**A 95% top-2 target is met on 70% of traffic** (0.961), and on 40% it is
perfect (1.000). Across *all* traffic top-3 reaches 0.953; top-2 is 0.895.

So 95% is achievable two ways -- accept three candidates, or accept two and
defer the least-confident 30%. It is not achievable at top-2 across 100% of
traffic with this data.

Per class:

| domain | precision | recall | F1 | n |
|---|---|---|---|---|
| personal_life | 0.811 | 0.811 | 0.811 | 37 |
| business_finance | 0.754 | 0.867 | 0.806 | 60 |
| arts_entertainment | 0.789 | 0.763 | 0.776 | 59 |
| software_tech | 0.776 | 0.776 | 0.776 | 58 |
| medicine_health | 0.818 | 0.720 | 0.766 | 25 |
| language | 0.846 | 0.688 | 0.759 | 32 |
| science_math | 0.694 | 0.758 | 0.725 | 33 |
| meta_other | 0.679 | 0.776 | 0.724 | 49 |
| humanities | 0.762 | 0.615 | 0.681 | 26 |
| law_politics | 0.667 | 0.571 | 0.615 | 21 |

`law_politics` is the weakest class; it confuses with `business_finance` on
prompts where both are genuinely defensible (contract law, tax, regulation).

## Optional: merged domains

Two pairs can be collapsed, because the distinction is neither reliably
labellable nor useful to a router:

    business_finance + law_politics -> business_law
    humanities + arts_entertainment -> culture

| | classes | top-1 | top-2 | top-3 |
|---|---|---|---|---|
| baseline | 10 | 0.758 | 0.895 | 0.953 |
| **merged at inference** | 8 | **0.773** | **0.910** | **0.963** |
| merged by retraining | 8 | 0.738 | — | — |

**Merge at inference, not at training** — that is the finding worth keeping.
Summing a 10-class model's probabilities into 8 groups scores 0.773; retraining
the same ensemble on 8 merged labels scores 0.738. Training on coarse labels
discards distinctions the model can otherwise learn and sum over. So the stored
labels stay fine-grained and `DomainHead(..., merge_domains=True)` applies the
grouping at serve time, reversibly.

`science_math + software_tech -> technical` scores better still (0.785 / 0.925)
and is deliberately excluded: maths and code route to different models, so
collapsing them buys a metric by destroying a distinction the router needs.

## Serving

```python
from router.inference import DomainHead

head = DomainHead("artifacts/v7/PROD_ensemble", defer_below=0.55, shortlist_size=2)
p = head.predict("whats the statute of limitations on unpaid invoices")
# p.domain        -> "business_finance"
# p.confidence    -> 0.618   (calibrated; temperature comes from the run dir)
# p.shortlist     -> ["business_finance", "law_politics"]
# p.distribution  -> all 10 domains
# p.should_defer  -> False
```

For a consumer that works with probabilities -- NMIRT, for instance --
`p.distribution` is strictly better than any hard label or shortlist: it carries
everything the model knows and lets the consumer pick its own operating point.

## Two measurement errors worth knowing about

This project produced two numbers that were wrong for methodological reasons,
and both are worth recording so they are not repeated.

**1. The eval set was being resampled every build.** Scores from different runs
were computed on different test sets, so none of them were comparable. A
learning curve fitted across those points suggested "~5,200 labels reaches 93%".
Re-measured on a frozen eval, the curve is nearly flat (error ~ n^-0.066), which
puts 93% out of reach by data volume alone. The eval set is now frozen on disk.

**2. Seed variance is larger than most effects being measured.** Five seeds of
one fine-tuning config spanned 63.5–67.8% — **sd 1.75, 95% CI ±3.4 points**.
Every single-run comparison below ~3 points in this project was noise, including
the apparent effect of synthetic data. Run multiple seeds before believing a
difference.

A third trap was caught before it shipped: searching ensemble member
combinations against the *test* set gave 76.0%; the same search done honestly on
validation gives 73.75%. That 2.25-point gap was pure selection bias.

## Why an ensemble of frozen encoders, not a fine-tune

| approach | real-prompt top-1 |
|---|---|
| zero-shot (no training) | 0.388 |
| fine-tuned MiniLM, mean of 5 seeds | 0.648 |
| fine-tuned MiniLM, best single seed | 0.678 |
| best single frozen encoder + linear head | 0.685 |
| **6-member weighted ensemble (shipped)** | **0.758** |

Fine-tuning on ~1.3k examples overfits and is high-variance. Frozen encoders
with a light head are deterministic, individually stronger, and averaging three
*different* encoders adds diversity that averaging seeds cannot. Six members: `multilingual-e5-large-instruct`, `bge-base`, `bge-large`,
`bge-m3`, `mxbai-embed-large` (linear heads) and `bge-large` (kNN, k=25).

**One member was silently broken.** `intfloat/*` encoders are trained with a
`"query: "` prefix on every input; we were embedding raw text. Adding it moved
e5-large from 0.6675 to 0.7000 — a 3.25-point bug, not a tuning choice.
`EmbeddingEncoder` now applies the prefix automatically per model family.

Things that did **not** help, measured on the frozen eval:

| tried | result |
|---|---|
| multi-encoder feature concatenation | 0.733, at 5× inference cost |
| kNN alone | 0.673 |
| per-class decision weights fitted on val | val 0.694→0.736, **test 0.745→0.735** |
| synthetic prompts for 4 weak classes | net −1.75 (inside noise) |

The per-class weights are the instructive failure: they improved validation
macro-F1 by 4 points and *hurt* test. With a ~350-row validation set, threshold
fitting is curve-fitting to noise.

## How it got here

Two earlier versions failed, and both failures shaped this one.

**v1 — Dewey decimal topics (9 classes).** 91% on benchmark questions,
**47% on real prompts**. It had learned exam formatting. Its best class
(`language`, 0.99 F1) was the WMT translation subset, identifiable because the
text is not English; on real traffic it became a dumping ground, predicted 4×
more often than it occurred. Scaling the model did nothing: 22M → 109M gained
0.8 points and 149M scored *worse* than 22M.

**v2 — capability labels (5 classes).** Fixed the distribution problem but
collapsed 58% of traffic into a single `other` class. Fine for routing, useless
as a domain signal.

**v3 — this.** Ten domains, chosen so that every class is assignable from real
traffic and nothing has to be forced. `personal_life` and `meta_other` exist
because roughly a third of real prompts are advice, chat, or questions about the
assistant — v1 had nowhere to put those.

## Why two training stages

Neither data source works alone:

| | prompts | labels |
|---|---|---|
| benchmarks (MMLU-Pro, RouterArena, BIG-bench) | exam-formatted | reliable, abundant |
| LMArena flags | real traffic | 3 domains only |
| **hand-labelled** | **real traffic** | **all 10 domains, 1,441 rows** |

Measured, on the same held-out real prompts:

| training | real-prompt top-1 |
|---|---|
| benchmarks + real, mixed | 0.589 |
| hand-labelled real only | 0.548 |
| **benchmark pretrain → real fine-tune** | **0.674** |

Stage 1 learns domain vocabulary from 20k benchmark rows. Stage 2 corrects the
distribution on 872 real prompts. The combination beats either alone by 8–13
points. Epochs and learning rate were swept across the transfer stage and moved
the result by ~1 point (66.3–67.4%), so **the remaining headroom is labelled
real data, not tuning**.

Two labelling decisions carry a lot of weight:

- **RouterArena labels come from its `Category` (Dewey subclass), not `Domain`.**
  "61 Medicine and health" is unambiguous; its parent "6 Technology" lumps
  medicine with engineering, and that ambiguity is much of why v1 failed.
- **An unflagged LMArena row gets no label at all.** It could be any of the ten
  domains, and guessing would poison the set. Those rows are exactly what the
  hand labelling covers.

## The hand-labelled set

`data/handlabelled/real_prompts.parquet` — 1,441 real LMArena prompts, read and
labelled individually. It is committed rather than gitignored: it is the one
asset here that cannot be regenerated by rerunning a script.

Its class distribution is the real distribution, not a balanced one:
business_finance 229, software_tech 198, arts_entertainment 192, meta_other 183,
science_math 135, personal_life 129, language 119, humanities 95,
medicine_health 87, law_politics 74.

## Usage

```bash
pip install -e .

# 1. build the split the shipped model uses (hand-labelled real prompts,
#    frozen eval held out by prompt text)
python -m router.cli build-real

# 2. train the production ensemble
python -m router.cli train experiments/v4/PROD_ensemble.yaml \
    --out-dir artifacts/prod --save-model --no-latency

# 3. inspect
python -m router.cli analyze --out-dir artifacts/prod
python -m router.cli diagnose        # feature correlation: VIF, PCA, effective rank
pytest
```

`--no-latency` matters for ensembles: latency timing runs 200 prompts one at a
time through every member, which for six encoders is ~1200 sequential passes
and dominates the run.

The two-stage fine-tune path (`experiments/v3/`) is kept for the record. It is
*not* the shipped model -- it scored 0.674 against the ensemble's 0.758, and its
seed variance was ±3.4 points.

Only `data/handlabelled/` and `data/synthetic/` are committed; `data/processed/`
and `artifacts/` are rebuilt by the commands above. Benchmark sources download
from Hugging Face on first use.

Adding an experiment is a YAML file; adding a model is a `@register("name")`
decorator. `init_from` on a fine-tune config chains the two stages.

## Feature correlation

`diagnose` reports the embedding space's redundancy. On bge-small over 2,000
prompts: **384/384 dimensions have VIF > 10, max pairwise correlation 0.462,
effective rank 186 of 384.**

Low pairwise correlation with enormous VIF means the redundancy is
*multivariate* — no two dimensions duplicate each other, but each is nearly
predicted by a linear combination of the rest. Correlation cannot see this.

- **`PCA → VIF` drops nothing**: components are orthogonal, so every VIF is 1.0.
- **Do not standardise embeddings.** Reduction cost ~7 points, and the cause was
  `StandardScaler`, not PCA — the encoder emits L2-normalised vectors.

## Layout

```
src/router/
  taxonomy.py     10 domains + mappings from 4 sources
  sources.py      loaders: hand-labelled, LMArena, MMLU-Pro, RouterArena, BIG-bench
  dataset.py      canonical row, dedupe, stratified split, leakage guard
  embeddings.py   frozen encoder + on-disk cache
  reduction.py    PCA / VIF feature diagnostics
  models.py       classifier interface, registry, tfidf / frozen / zero-shot heads
  finetune.py     fine-tuned transformer, with `init_from` for stage-2 transfer
  metrics.py      accuracy, per-class P/R/F1, ECE, top-k, selective risk, calibration
  experiment.py   run -> run directory; sweep -> leaderboard; per-source scoring
  analysis.py     confusion, per-class breakdown, risk/coverage
  cli.py          build-data / describe-data / train / analyze / diagnose
data/handlabelled/real_prompts.parquet   1,441 hand-labelled real prompts
experiments/v3/                          the current pipeline
```

Per-source scoring is built into the experiment runner, so no future model can
report a flattering aggregate that hides a collapse on real traffic.

## Next

See [NEXT_STEPS.md](NEXT_STEPS.md).
