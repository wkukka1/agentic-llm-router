# agentic-llm-router — prompt classifiers

Two classifiers over a raw user prompt, feeding the routing stage that picks a
model. No LLM at inference, no network calls, no fine-tuned weights.

| head | question | classes | top-1 | top-2 |
|---|---|---|---|---|
| **domain** | what is it about | 8 (merged from 10) | 0.763 | 0.919 |
| **task** | what does it ask to be done | 7 | 0.793 | 0.950 |

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
from router.inference import RouterHead

head = RouterHead("artifacts/v7/PROD_ensemble", "artifacts/v11/PROD_task",
                  merge_domains=True, shortlist_mass=0.85, defer_below=0.35)

p = head.predict("summarise this cardiology paper")
p.key                  # "medicine_health/summarize"
p.domain.shortlist     # ["medicine_health"]  -- length adapts to how torn it is
p.domain.distribution  # calibrated, all 8 domains
p.task.distribution    # calibrated, all 7 tasks
p.should_defer         # True if EITHER axis is unsure
```

For a downstream difficulty or ability model, take the vector rather than the
labels — the shape of the distribution carries what the argmax does not:

```python
X, names = head.vectorise(prompts)   # (n, 24) plus its column names
```

## Reproducing

```bash
pip install -r requirements.txt

# domain: build splits from the hand-labelled real prompts, then train
python -m router.cli build-real --variant real_only
python -m router.cli train experiments/v4/PROD_ensemble.yaml --save-model

# task: same, with the synthetic supplements folded into train only
python -m router.cli build-task
python -m router.cli train experiments/v5/PROD_task.yaml --save-model

# checks
python -m router.cli overfit                          # five-check audit
python -m router.cli analyze PROD_ensemble            # confusion, per-class F1
python -m router.cli external artifacts/v7/PROD_ensemble
```

## Layout

```
src/router/
  taxonomy.py    domain label space, source mappings, merge rules
  tasktype.py    task label space, and why it looks like this
  sources.py     every loader, one per data source
  dataset.py     canonical rows, splitting, leakage assertions
  embeddings.py  frozen encoders with an on-disk cache
  models.py      five model types behind one registry
  experiment.py  the runner: fit, score, calibrate, write a run dir
  metrics.py     accuracy, top-k, ECE, coverage curves
  analysis.py    confusion matrices, per-class F1, error slices
  overfit.py     the five-check audit
  inference.py   serving: DomainHead, TaskHead, RouterHead, the vector
  cli.py         build-task / build-real / train / analyze / external / overfit

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

So there is real headroom — but no model lever reaches it. Eight encoders,
stacking, self-training, cross-head features and two-stage specialists all land
inside the noise band. The 2,441 training labels predate the rubric, and making
them consistent with it is the open lever.

Meanwhile the shortlist is the product, not the argmax. Pass `distribution`
downstream and let the consumer pick its operating point.
