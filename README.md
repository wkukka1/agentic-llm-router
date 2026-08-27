# agentic-llm-router — domain classifier

The topic-classification layer of the router. Given a prompt, predict a
distribution over 9 domains, calibrated well enough that a downstream stage can
threshold on the confidence.

This repo is scoped to that one layer. The routing policy it feeds — an NMIRT
model that learns query difficulty, discrimination and per-model ability — lives
elsewhere. The earlier hand-tuned agentic layer is on the `parked/` branch,
superseded by that design.

## Results

13 experiments run, 9 kept. RouterArena, 5,878 train rows.

| experiment | acc | macro-F1 | ECE raw → cal | p50 ms |
|---|---|---|---|---|
| **11_finetune_bge_base** (109M) | **0.894** | **0.891** | 0.055 → 0.029 | 24.2 |
| 07_finetune_minilm (22M) | 0.886 | 0.885 | 0.061 → 0.063 | 8.9 |
| 10_finetune_minilm_qonly | 0.823 | 0.823 | 0.052 → 0.054 | 9.4 |
| 02_tfidf_logreg_wordchar | 0.810 | 0.809 | 0.160 → 0.028 | 1.1 |
| 12_embed_bge_small_pca | 0.763 | 0.758 | 0.026 → 0.026 | 14.2 |
| 05_embed_bge_small_logreg | 0.762 | 0.756 | 0.026 → 0.023 | 9.9 |
| 01_tfidf_logreg_word | 0.759 | 0.757 | 0.200 → 0.032 | 0.4 |
| 09_tfidf_logreg_wordchar_qonly | 0.746 | 0.739 | 0.129 → 0.025 | 0.9 |
| 04_embed_minilm_logreg | 0.729 | 0.724 | 0.033 → 0.040 | 9.7 |

**Fine-tuning is the lever, not head capacity or model size.** Freezing the
encoder costs ~15 points (0.729 vs 0.886 on the same MiniLM). But a non-linear
head on frozen features buys nothing (0.761 MLP vs 0.762 linear) — the frozen
space is already linearly separable. And scaling the fine-tune 22M → 109M gains
only 0.8 points, while a 149M ModernBERT scored *worse* than MiniLM at 22M.

That flat scaling curve is a **label-noise ceiling**, and the error analysis
says the same thing independently:

- Difficulty barely moves accuracy: easy .876 / medium .902 / hard .885 — flat
- The largest confusion is `technology → science`, and 11 of those 16 cases are
  `MMLUPro_health`. Medical questions sit under `technology` because Dewey class
  6 covers Medicine, so the model is penalised for the *more* defensible answer.
- Worst classes are the arbitrary ones: `70 Arts` 44% error, `15 Psychology` 36%
- `language` at 0.976 F1 is mostly WMT translation pairs, trivially separable
  because the prompt is not in English

### Hitting 95%

Top-1 at 95% is probably out of reach on this label space. It is already
reachable other ways:

| | |
|---|---|
| top-1 | 0.894 |
| **top-2** | **0.957** |
| top-3 | 0.981 |
| top-1 @ 90% coverage | 0.949 |

Passing the 9-dim distribution downstream instead of an argmax clears the target
and gives the next stage strictly more to work with.

### The number to distrust

The `question_only` ablation costs TF-IDF 6.4 points and the fine-tuned model
6.3 — near-identical across two very different model families, so roughly six
points is **benchmark formatting**, not topic understanding. Val and test cannot
detect this because they are drawn from the same 79 benchmarks with the same
conventions. Treat 0.894 as in-distribution only until it is checked against
real prompts.

## Data

[`RouteWorks/RouterArena`](https://huggingface.co/datasets/RouteWorks/RouterArena)
— 8,400 rows aggregating 79 benchmarks (MMLU-Pro, PubMedQA, LiveCodeBench,
NarrativeQA, MATH, WMT19, …). Labels are Dewey top-level classes; class 2
(Religion) is absent, leaving 9 at roughly 2:1 imbalance.

Rows carry `Context`, `Question` and `Options`. How they are reassembled is an
experimental axis, since option blocks are a format artifact a classifier will
latch onto instead of the topic: `full_prompt`, `no_options`, `question_only`.

## Usage

```bash
pip install -e .

python -m router.cli build-data --variant all
python -m router.cli describe-data
python -m router.cli train experiments/domain --save-model
python -m router.cli analyze                    # per-class P/R/F1, confusion, slices
python -m router.cli diagnose                   # feature correlation: VIF, PCA, effective rank
pytest
```

Adding an experiment is a YAML file; adding a model is a `@register("name")`
decorator. Neither touches the runner.

## Feature correlation

`diagnose` reports the redundancy structure of the embedding space. On
bge-small over 2,000 prompts:

| measure | value |
|---|---|
| dimensions with VIF > 10 | 384 / 384 |
| median VIF | 34,503 |
| max abs pairwise correlation | 0.462 |
| **effective rank** | **186** of 384 |

**VIF and pairwise correlation disagree, and that is the finding.** Correlation
is low, which alone suggests no redundancy — yet nearly every dimension has an
enormous VIF. The redundancy is *multivariate*: no two dimensions duplicate each
other, but each is almost perfectly predicted by a linear combination of the
rest. Pairwise correlation cannot see this; VIF can.

Two practical consequences:

- **`PCA → VIF` drops nothing.** Components are orthogonal, so every VIF is
  exactly 1.0. Confirmed on real data and pinned in `tests/test_reduction.py`.
- **Do not standardise embeddings.** Reduction cost ~7 points of accuracy
  (0.762 → 0.692), and the cause was `StandardScaler`, not PCA or truncation:
  the encoder emits L2-normalised vectors and rescaling each dimension to unit
  variance destroys that geometry. PCA *without* standardisation is
  accuracy-neutral at 286 of 384 dimensions.

Use the report to understand the feature space, not as a reason to prune it.

## Layout

```
src/router/
  taxonomy.py     Domain / TaskType / Difficulty label spaces + mappings
  dataset.py      canonical Example row, RouterArena source, dedupe/split/leakage
  embeddings.py   frozen encoder + on-disk cache
  reduction.py    PCA / VIF feature-correlation diagnostics
  models.py       DomainClassifier interface, registry, tfidf + frozen-embedding heads
  finetune.py     end-to-end fine-tuned transformer
  metrics.py      accuracy, per-class P/R/F1, ECE, top-k, selective risk, calibration
  experiment.py   run one experiment -> run directory; sweep -> leaderboard
  analysis.py     confusion, per-class breakdown, risk/coverage, confidently-wrong
  config.py       experiment config (YAML -> dataclass)
  cli.py          build-data / describe-data / train / analyze / diagnose
experiments/domain/    9 experiment configs
```

Each run directory holds `config.yaml`, `metrics.json`,
`test_predictions.parquet` and optionally `model/`, so error analysis and
re-scoring never require a retrain.

Probabilities, not labels, are the contract: `predict_proba` is the interface,
every run is temperature-scaled on validation, and ECE is reported alongside
accuracy. A threshold on a miscalibrated score is meaningless.

## Next

See [NEXT_STEPS.md](NEXT_STEPS.md). The short version: measure the
out-of-distribution gap on real prompts before trusting 0.894, and emit the
distribution rather than the argmax.
