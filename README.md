# agentic-llm-router — capability classifier

The first stage of the router. Given a prompt, predict what the answering model
needs to be good at — `code`, `math`, `creative_writing`, `translation`, or
`other` — as a calibrated distribution the next stage can threshold on.

Scoped to that one stage. The routing policy it feeds (NMIRT: query difficulty,
discrimination, per-model ability) lives elsewhere. An earlier hand-tuned
agentic layer is on the `parked/` branch.

## The result that matters

Accuracy is reported **separately on real prompts and benchmark questions**,
because the aggregate hides the only question worth asking.

| | benchmark questions | **real prompts** | gap |
|---|---|---|---|
| v1 — Dewey topic labels | 91.2% | **47.5%** | 44 pts |
| **v2 — capability labels** | 93.1% | **76.6%** | **16 pts** |

**v1 was measuring formatting recognition.** It scored 91% on exam questions and
collapsed to 47% on real user prompts — worse than it looks, because it dumped
everything into `language`, its best benchmark class (0.99 F1), which turned out
to be the WMT translation subset being trivially identifiable by non-English
text. Faced with a short conversational prompt, that class became a default.

v2 fixes the labels *and* the training distribution. Real-prompt accuracy
improved 29 points.

Current model (`C2_finetune_minilm`, 22M params, 8.9 ms):

| | overall | real | benchmark |
|---|---|---|---|
| accuracy | 0.803 | 0.766 | 0.931 |
| top-2 | 0.968 | 0.961 | 0.989 |
| macro-F1 | 0.749 | 0.575 | 0.700 |

Per-class on **real** prompts:

| class | precision | recall | n |
|---|---|---|---|
| code | 0.795 | 0.800 | 1,538 |
| other | 0.858 | 0.724 | 2,301 |
| creative_writing | 0.556 | 0.840 | 324 |
| math | 0.491 | 0.855 | 234 |

Honest reading: **good recall, poor precision on the small classes.** The model
finds math and creative-writing prompts but over-claims them. Translation has
*zero* real training examples and is untested outside benchmarks.

## Why two data sources

Training needs (prompt, label) pairs. Each source supplies only one half.

| | prompts | labels |
|---|---|---|
| RouterArena | exam-formatted | reliable |
| LMArena | real user traffic | limited to 3 flags |

RouterArena alone produced the 44-point collapse. LMArena alone cannot label
translation. So both are used, mapped into one shared label set:

```
LiveCodeBench, MMLUPro_computer science  ──► code             ◄── LMArena is_code
MATH, AIME, GSM8K, MMLUPro_math          ──► math             ◄── LMArena math_v0.1
                                             creative_writing ◄── LMArena creative_writing_v0.1
WMT19                                    ──► translation
everything else                          ──► other            ◄── LMArena: no flag set
```

Two decisions worth spelling out:

- **Labels come from RouterArena's `Dataset name`, not its `Domain`.** A row from
  LiveCodeBench is a coding task *by construction*; its Dewey class is a
  cataloguing opinion. That swap is most of the fix.
- **An absent LMArena flag is a real negative**, so unflagged rows become
  `other` rather than being dropped. Roughly half of real traffic is `other`,
  and a model that never sees it will not predict it.

Training set: **26,391 rows, 78% real prompts, 22% benchmark.** Splits are
stratified by capability *and* source, so every split holds both distributions.

## Usage

```bash
pip install -e .

python -m router.cli build-capability          # real prompts + benchmarks -> splits
python -m router.cli describe-data --variant capability
python -m router.cli train experiments/capability --save-model
python -m router.cli analyze --out-dir artifacts/capability
python -m router.cli diagnose                  # feature correlation: VIF, PCA, effective rank
pytest
```

Adding an experiment is a YAML file; adding a model is a `@register("name")`
decorator. Neither touches the runner.

## Feature correlation

`diagnose` describes the embedding space's redundancy. On bge-small over 2,000
prompts: **384/384 dimensions have VIF > 10, max pairwise correlation 0.462,
effective rank 186 of 384.**

Low pairwise correlation alongside enormous VIF means the redundancy is
*multivariate* — no two dimensions duplicate each other, but each is nearly
predicted by a linear combination of the rest. Correlation cannot see this.

Two consequences:

- **`PCA → VIF` drops nothing.** Components are orthogonal, so every VIF is
  exactly 1.0. Pinned in `tests/test_reduction.py`.
- **Do not standardise embeddings.** Reduction cost ~7 points (0.762 → 0.692),
  and the cause was `StandardScaler`, not PCA: the encoder emits L2-normalised
  vectors and rescaling each dimension destroys that geometry. PCA *without*
  standardisation is accuracy-neutral at 286 of 384 dimensions.

## Layout

```
src/router/
  taxonomy.py     Capability label space + mappings from both sources
  dataset.py      canonical row, both loaders, dedupe/split/leakage
  embeddings.py   frozen encoder + on-disk cache
  reduction.py    PCA / VIF feature-correlation diagnostics
  models.py       classifier interface, registry, tfidf / frozen / zero-shot heads
  finetune.py     end-to-end fine-tuned transformer
  metrics.py      accuracy, per-class P/R/F1, ECE, top-k, selective risk, calibration
  experiment.py   one run -> run directory; sweep -> leaderboard; per-source scoring
  analysis.py     confusion, per-class breakdown, risk/coverage, confidently-wrong
  config.py       experiment config (YAML -> dataclass)
  cli.py          build-capability / build-data / describe-data / train / analyze / diagnose
experiments/capability/   capability experiments
experiments/domain/       v1 topic experiments, kept for the record
```

Probabilities, not labels, are the contract: `predict_proba` is the interface,
every run is temperature-scaled on validation, and ECE is reported alongside
accuracy.

## Next

See [NEXT_STEPS.md](NEXT_STEPS.md).
