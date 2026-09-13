# agentic-llm-router — prompt signals

Three heads over a raw user prompt, feeding the routing stage that picks a
model. No LLM at inference, no network calls, no fine-tuned weights.

| head | question | output | score |
|---|---|---|---|
| **domain** | what is it about | 8 classes (merged from 10) | 0.763 top-1 / 0.919 top-2 |
| **task** | what does it ask to be done | 6 classes | 0.844 top-1 / 0.967 top-2 |
| **length** | how much work is it | expected tokens + P(tokens > T) | rho 0.573, against a ≈0.742 ceiling |
| **attributes** | what does it need | 11 calibrated probabilities | code 0.905 precision @ 50% recall |

The first two are trained on labels written by hand. The length head is trained
on labels nobody wrote: every arena row carries both models' responses and an
exact token count, so its corpus is 109,335 prompts at zero annotation cost.

Both cross-validated on hand-labelled real prompts, in the configuration that
ships. On 402 prompts labelled by someone outside this project the domain head
reaches 0.923 top-1, and its adaptive shortlist holds the true domain 0.985 of
the time using 1.40 labels.

Every number in this repository is measured on **hand-labelled real user
prompts**, cross-validated. Benchmark accuracy is not reported anywhere: v1
scored 0.91 on benchmark data and 0.47 in the wild, and that cost two rebuilds.

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — how it works, why each piece is
  there, the full results, and the known limits.
- **[EXPERIMENTS.md](EXPERIMENTS.md)** — everything tried, including what
  failed and why. Read before running a new experiment.

## Serving

```python
from prompt_decomposition import RouterHead

head = RouterHead("artifacts/v7/PROD_ensemble", "artifacts/v11/PROD_task",
                  merge_domains=True, shortlist_mass=0.85, defer_below=0.35)

p = head.predict("summarise this cardiology paper")
p.key                  # "medicine_health/summarize"
p.domain.shortlist     # ["medicine_health"]  -- length adapts to how torn it is
p.domain.distribution  # calibrated, all 8 domains
p.task.distribution    # calibrated, all 7 tasks
p.should_defer         # True if EITHER axis is unsure
```

### One vector, every signal

The handoff to whatever routes or scores downstream. Pass the run directories
you have; heads you have not trained are simply absent, and the column names
say which.

```python
head = RouterHead("artifacts/v7/PROD_ensemble", "artifacts/v11/PROD_task",
                  length_run="artifacts/length/bge-small-en-v1.5__surface_embedding",
                  attribute_run="artifacts/attributes/bge-small-en-v1.5__surface_embedding",
                  merge_domains=True, shortlist_mass=0.85)

X, names = head.vectorise(prompts)   # (n, 57) plus its column names

p = head.predict("write me a python function that reverses a linked list")
p.length.expected_tokens        # ~710
p.attributes.gates              # {"code": 0.96, "math": 0.01, ...} calibrated
p.attributes.criteria           # the seven judged dimensions
```

Blocks are appended, never interleaved: 14 distributions + 9 scalars is the
core, then +1 length, +11 attributes, +21 surface. Loading another head extends
the vector on the right and leaves every existing column index alone. Heads
that share an encoder share the pass.

**Difficulty is deliberately not a column.** Four feature families failed to
predict it from prompt text and the arena's own hardness rubric predicts the
routing outcome at chance; it belongs downstream of this vector, computed from
these signals together with model behaviour.

Read length as a ranking, not a token count: ranked by prediction, the top 25%
of traffic holds 37.9% of all generated tokens (random 25%, perfect 53.6%), and
any single estimate is typically within a factor of 2.

## Reproducing

```bash
pip install -r requirements.txt

# domain: build splits from the hand-labelled real prompts, then train
python -m prompt_decomposition.cli build-real --variant real_only
python -m prompt_decomposition.cli train experiments/v4/PROD_ensemble.yaml --save-model

# task: same, with the synthetic supplements folded into train only
python -m prompt_decomposition.cli build-task
python -m prompt_decomposition.cli train experiments/v5/PROD_task.yaml --save-model

# checks
python -m prompt_decomposition.cli overfit                          # five-check audit
python -m prompt_decomposition.cli analyze PROD_ensemble            # confusion, per-class F1
python -m prompt_decomposition.cli external artifacts/v7/PROD_ensemble

# length: builds its own corpus from the arena dump, then sweeps feature sets
python -m prompt_decomposition.cli length --build
```

## Layout

```
src/prompt_decomposition/
  composite.py          RouterHead: both heads over one prompt, plus the vector
  cli.py                build-task / build-real / train / analyze / external /
                        overfit / length
  core/                 shared machinery, no single head owns it
    sources.py          every loader, one per data source
    dataset.py          canonical rows, splitting, leakage assertions
    embeddings.py       frozen encoders with an on-disk cache
    models.py           four model types behind one registry
    experiment.py       the runner: fit, score, calibrate, write a run dir
    metrics.py          accuracy, top-k, ECE, coverage curves
    analysis.py         confusion matrices, per-class F1, error slices
    overfit.py          the five-check audit
    head.py             CalibratedHead: load a run dir, serve calibrated scores
    config.py           an experiment is a YAML file
    settings.py         seed and paths, one place
  domain_classifier/    taxonomy.py (label space, merges) + head.py
  task_classifier/      taxonomy.py (label space, and why) + head.py
  length_estimator/     expected output length -- the head whose labels are free
    data.py             corpus from the arena dump, split by hashing the prompt
    model.py            ridge on log tokens, with a calibrated residual spread
    audit.py            gap, learning curve, alpha sweep, label permutation
    head.py             LengthHead: expected tokens and P(tokens > T)
    experiment.py       the feature-set sweep
  signals/              surface.py: 23 free deterministic features

data/handlabelled/   the labels. The asset. Committed deliberately.
data/synthetic/      written and generated supplements, training only
data/external/       402 prompts labelled by someone else
experiments/         the two shipped configs
docs/                the prompts used to source more labels
```

## The one thing to know

The domain classifier is limited by its **labels**, not its architecture. A
second annotator working from the written rubric agrees with the stored labels
**81.7%** of the time (kappa 0.793); the same annotator re-deriving the rules
from memory manages 77.0%; the model scores 74.1%.

That gap looked like headroom, so it was tested: all 2,441 labels were
relabelled against the rubric and the classifier retrained. On the external
prompts it got **significantly worse** (0.930 → 0.896). The original labels beat
the rubric-consistent ones against ground truth neither saw — the written rules
are a lossy summary of a partly tacit policy.

Every lever tried is now neutral or negative: eight encoders, stacking,
self-training, cross-head features, two-stage specialists, and relabelling.

Meanwhile the shortlist is the product, not the argmax. Pass `distribution`
downstream and let the consumer pick its operating point.
