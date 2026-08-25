# agentic-llm-router

Routing a prompt to the right model in a pool, instead of sending everything to
the most expensive one.

This is not a one-shot classifier that picks a model. The flow is:

1. **Gate** the prompt. If it is underspecified, stop and go back to the user for
   the missing requirements — routing a bad prompt wastes the expensive path and
   produces an answer that gets rejected anyway.
2. **Signal** it: domain, difficulty, task type.
3. **Decide weak or strong.**
   - *weak* → one cheap model answers directly.
   - *strong* → an **orchestrator** decomposes the prompt into sub-queries,
     routes each one back through the same signal + policy stack (with skills
     attached), executes them in dependency order feeding answers forward, and
     synthesises.
4. **Follow up.** The answer and any follow-up turn feed back in, carrying
   session context — so "now do it in Rust" is routable rather than a
   domain-less fragment.

Current status: the **domain head is trained** (10-experiment sweep, best 88.6%
test accuracy). The agentic control plane is built and tested end to end. The
difficulty and task-type signals are **heuristic placeholders** behind the same
interfaces the trained heads will use — see [Status](#status).

---

## Architecture

```
  prompt
    │
    ▼
 ┌────────┐  underspecified   ┌──────────────────────┐
 │  gate  │──────────────────►│ back to user:        │
 └────────┘                   │ missing requirements │
    │ sufficient              └──────────────────────┘
    ▼
 ┌──────────────────────────────────────┐
 │ signals                              │
 │  domain head   p(domain)   [trained] │
 │  difficulty    d ∈ [0,1]   [heur.]   │
 │  task type     capability  [heur.]   │
 └──────────────────────────────────────┘
    │
    ▼
 ┌──────────────────────────────────────────────────┐
 │ policy: weak / strong / orchestrate              │
 │   escalate on: difficulty · ambiguity · multi-ask│
 └──────────────────────────────────────────────────┘
    │                                  │
    │ weak                             │ strong
    ▼                                  ▼
 ┌──────────┐                   ┌──────────────┐
 │ 1 cheap  │                   │ orchestrator │
 │  model   │                   └──────────────┘
 └──────────┘                          │ decompose
    │                                  ▼
    │                    ┌─────────────────────────────┐
    │                    │ sub-query ──► route ──► skill│  ×N
    │                    │      ▲            │          │  (dependency order,
    │                    │      └── interleave ─────────┤   answers fed forward)
    │                    └─────────────────────────────┘
    │                                  │ synthesise
    ▼                                  ▼
 ┌──────────────────────────────────────────────────┐
 │ answer  ──►  session  ──►  next turn (follow-up) │
 └──────────────────────────────────────────────────┘
```

Escalation has three *distinct* triggers, kept separate so each can be tuned:

| Trigger | Meaning | Guard |
|---|---|---|
| **hard** | difficulty ≥ `strong_difficulty` | — |
| **ambiguous** | domain head is unsure, so the cheap model's fitness is unknown | only fires above `ambiguity_difficulty_floor`, so a trivial prompt never pays for ambiguity |
| **decomposable** | prompt contains several asks needing interleaving | fires at a lower difficulty bar (`decompose_difficulty`) |

Layers, and why they are separate:

| Layer | Module | Responsibility |
|---|---|---|
| Taxonomy | `router.data.taxonomy` | The label spaces. Domain, task type and difficulty are **orthogonal axes**, not one flat class list. |
| Schema | `router.data.schema` | One canonical `Example` row. Every source normalises into it, so nothing downstream knows where data came from. |
| Sources | `router.data.sources.*` | One module per dataset. Each returns `list[Example]`. |
| Build | `router.data.build` | Dedup → stratified split → leakage assertion → parquet. Runs once; every experiment scores identical rows. |
| Features | `router.features.embeddings` | Frozen encoders with an on-disk cache keyed by (encoder, pooling, length, content). |
| Models | `router.models.*` | Everything implements `DomainClassifier` (`fit`/`predict_proba`/`save`/`load`) and self-registers by name. |
| Training | `router.training.*` | Experiment execution, metrics, calibration, error analysis, leaderboard. |
| Inference | `router.inference` | `DomainHead` — loads a run directory, serves calibrated predictions. The seam between training and serving. |
| Agent | `router.agent.*` | The control plane: gate, signals, policy, pool, skills, orchestrator, session. |

Adding an experiment is a YAML file. Adding a model is a class with a
`@register("name")` decorator. Neither requires touching the runner.

### Why probabilities, not labels

Every decision in the router design is a *threshold on confidence* — escalate to
a stronger model, split into sub-queries, ask a clarifying follow-up. So the
contract of `DomainClassifier` is `predict_proba`, every run is temperature-scaled
on validation, and the reported metrics include ECE and a full risk/coverage
curve rather than accuracy alone.

---

## Data

| Source | Rows | What it actually provides |
|---|---|---|
| [`RouteWorks/RouterArena`](https://huggingface.co/datasets/RouteWorks/RouterArena) | 8,400 | **Clean supervision.** `Domain` (9 Dewey classes), `Category` (28), `Difficulty` (easy/medium/hard). Primary training set. |
| [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | ~140k | **No domain label.** `category_tag` is *binary flags* (`math`, `is_code`, `creative_writing`, `if`) plus 7 `criteria_v0.1` hardness flags. Use for the task-type and difficulty heads and for realistic prompt distribution — never as domain ground truth. |

Domain labels come from RouterArena's Dewey prefix; class 2 (Religion) is absent,
leaving 9 classes at roughly 2:1 imbalance (1400 / 700 per class).

### Prompt variants

RouterArena rows carry `Context`, `Question` and `Options`. How you reassemble
them is a real experimental axis, because option blocks and context headers are
*format artifacts* a classifier will happily latch onto instead of the topic:

- `full_prompt` — context + question + rendered options (what a user sends)
- `no_options` — context + question
- `question_only` — question alone (hardest, closest to open user traffic)

Experiments 09 and 10 ablate against `question_only`; the gap is the share of
accuracy that will **not** transfer to real traffic.

---

## Usage

```bash
pip install -e .

# 1. data
python -m router.cli build-data --variant all     # download + dedupe + stratified split
python -m router.cli describe-data                # split statistics

# 2. train the domain head
python -m router.cli train experiments/domain --save-model
python -m router.cli analyze                      # error analysis for the leader

# 3. route prompts through the full agentic pipeline
python -m router.cli route \
  --head artifacts/domain/07_finetune_minilm \
  "What year did the Berlin Wall fall?" \
  "1. Refactor the parser for testability
   2. Then also profile the tokenizer and explain the bottleneck"

pytest
```

`route` runs in **plan** mode by default: it produces the full routing plan and
costs it, without generating answers. That runs offline with no API keys and is
what the evaluation harness scores. `--mode execute --backend-model <model>`
additionally calls the assigned models through LiteLLM.

Every decision explains itself:

```
orchestrate via orchestrator claude-opus-5
  - domain=cs_general (p=0.88), task=code, difficulty=0.60
  - prompt contains multiple asks -> decomposable
  - decomposable and difficulty >= 0.45
  - orchestrator claude-opus-5 will decompose and delegate
  - decomposed into 3 sub-queries (est. $0.0381 to execute)
    [0] -> claude-haiku-4-5  skills=[]                  :: Refactor this Python class for testability
    [1] -> claude-sonnet-5   skills=['code_interpreter'] :: Profile the hot loop and explain why it was slow
    [2] -> claude-sonnet-5   skills=['code_interpreter'] :: Write pytest cases for the result
```

### Evaluating the policy

```bash
python -m router.cli route-eval \
  --head artifacts/domain/07_finetune_minilm \
  --calibrate-thresholds
```

This routes a labelled split in plan mode and reports the **economics**: where
traffic lands, the model mix, and cost against the two trivial baselines
(everything to the cheapest model, everything to the best one).

Quality is deliberately *not* estimated here — doing so from the pool's own
prior `quality` field would be circular. Measuring it is LLMRouterBench's job.
What this gives you is the cost side of the frontier, which is what threshold
tuning needs.

#### Why thresholds are calibrated, not hand-set

The first run of `route-eval` found a real flaw. On RouterArena's benchmark-style
prompts the heuristic difficulty estimator is heavily compressed — 99th
percentile 0.50, max 0.59 — so the hand-set `strong_difficulty = 0.55` fired on
**0.35%** of prompts and the strong tier was effectively dead code. All traffic
went weak, or orchestrated on the multi-ask rule.

A difficulty estimator's absolute scale is arbitrary, so the thresholds are now
expressible as *quantiles of observed traffic*, fit on the train split:

```
--strong-quantile 0.70    # "send the hardest 30% to a strong model"
```

The quantiles are the routing budget, and the policy becomes invariant to which
difficulty estimator is plugged in — so swapping the heuristic for the trained
regressor will not require re-tuning by hand.

| | uncalibrated | calibrated (q=0.70/0.92) |
|---|---|---|
| weak | 93.3% | 71.3% |
| strong | **0.0%** | 16.3% |
| orchestrator | 2.7% | 8.3% |
| clarify | 4.0% | 4.0% |
| saving vs always-best | 69.9% | 54.8% |

### Configuring the pool

`configs/pool.yaml` describes the models (tier, cost per 1M tokens, domain
strengths, permitted skills) and `configs/skills.yaml` the tools and their
trigger conditions. Selection within a tier is a utility argmax over
quality − cost + domain fit, so changing the pool never means changing the
policy. `--cost-weight` moves the router between cost- and quality-optimising;
`--max-cost` downgrades any turn whose plan exceeds a per-turn cap rather than
failing it.

Skills are gated twice: by the signals that trigger them *and* by what the
selected model is permitted to call — so a weak model never gets handed a code
interpreter.

---

## Layout

```
src/router/
  config.py            experiment config (YAML -> dataclass)
  cli.py               build-data / describe-data / train / analyze
  inference.py         DomainHead — serving seam for the router
  data/
    taxonomy.py        Domain / TaskType / Difficulty label spaces + mappings
    schema.py          canonical Example row
    build.py           dedup, stratified split, leakage check
    sources/           routerarena.py, arena.py
  features/
    embeddings.py      frozen encoder + on-disk cache
  models/
    base.py            DomainClassifier interface
    registry.py        name -> class factory
    linear.py          tfidf_logreg, tfidf_linear_svm
    embedding_head.py  embed_logreg, embed_mlp
    finetune.py        finetune_transformer
  training/
    experiment.py      run one experiment -> run directory
    runner.py          sweep + leaderboard
    metrics.py         accuracy, macro-F1, ECE, top-k, selective risk, latency
    calibration.py     temperature scaling
    analysis.py        confusion, slices, risk/coverage, confidently-wrong rows
  agent/
    contracts.py       Action / Signals / RouteDecision / SubQuery / Session
    gate.py            stage 1 — prompt sufficiency, back to the user
    difficulty.py      difficulty estimator (heuristic; trained head to come)
    tasktype.py        task-type inference (heuristic; trained head to come)
    policy.py          weak / strong / orchestrate + thresholds
    pool.py            ModelPool, ModelSpec, tier- and cost-aware selection
    skills.py          skill registry and trigger rules
    orchestrator.py    decompose -> route each -> interleave -> synthesise
    backends.py        LLMBackend protocol, LiteLLM + heuristic implementations
    pipeline.py        AgenticRouter — the entry point
configs/               pool.yaml, skills.yaml
experiments/domain/    10 experiment configs
artifacts/domain/      leaderboard.{csv,md} + one directory per run
evaluations/           LLMRouterBench (submodule) — downstream router evaluation
```

Each run directory contains `config.yaml`, `metrics.json`,
`test_predictions.parquet` and optionally `model/` — so error analysis and
serving never require a retrain.

---

## Results

All 10 experiments, sorted by macro-F1. Reproduce with
`python -m router.cli train experiments/domain`.

| experiment | acc | macro-F1 | ECE raw → cal | p50 ms | MB |
|---|---|---|---|---|---|
| 07_finetune_minilm | 0.886 | 0.885 | 0.061 → 0.063 | 8.87 | 91 |
| 08_finetune_bge_small | 0.877 | 0.874 | 0.140 → 0.059 | 12.04 | 133 |
| 10_finetune_minilm_qonly | 0.823 | 0.823 | 0.052 → 0.054 | 9.40 | 91 |
| 03_tfidf_svm_wordchar | 0.812 | 0.809 | 0.099 → 0.038 | 2.69 | 60 |
| 02_tfidf_logreg_wordchar | 0.810 | 0.809 | 0.160 → 0.028 | 1.06 | 26 |
| 05_embed_bge_small_logreg | 0.762 | 0.756 | 0.026 → 0.023 | 9.86 | 0 |
| 06_embed_bge_small_mlp | 0.761 | 0.756 | 0.029 → 0.026 | 9.92 | 2 |
| 01_tfidf_logreg_word | 0.759 | 0.757 | 0.200 → 0.032 | 0.37 | 11 |
| 09_tfidf_logreg_wordchar_qonly | 0.746 | 0.739 | 0.129 → 0.025 | 0.86 | 13 |
| 04_embed_minilm_logreg | 0.729 | 0.724 | 0.033 → 0.040 | 9.70 |

Scaling up (experiments 11–13, `max_length` 512):

| experiment | params | acc | macro-F1 | p50 ms | train |
|---|---|---|---|---|---|
| **11_finetune_bge_base** | 109M | **0.894** | **0.891** | 24.2 | 61 min |
| 13_finetune_modernbert_base | 149M | 0.878 | 0.875 | 70.5 | 75 min |
| 07_finetune_minilm *(22M, for reference)* | 22M | 0.886 | 0.885 | 8.9 | 6 min |
| 12_finetune_deberta_v3_base | 184M | *failed, then fixed* | | | |

**Capacity buys almost nothing here — as the error analysis predicted.** Going
22M → 109M gains **0.8 points** (0.886 → 0.894) for 2.7× the latency and 10× the
training time. ModernBERT at 149M is *worse* than MiniLM at 22M. That is the
signature of a label-noise ceiling rather than a capacity ceiling, and it is
consistent with the confusion analysis below: the model is being asked to
reproduce Dewey shelving decisions that are genuinely ambiguous.

Experiment 12 exposed a real bug — DeBERTa-v3 loads in fp16 and crashed the
class-weighted loss (`expected scalar type Half but found Float`). The loss is
now computed in fp32 unconditionally, with a regression test. The run has not
been repeated; on this evidence it would not change the conclusion. 0 |

**Winner: `07_finetune_minilm`** — and it dominates rather than trades off. It beats
the larger `bge-small` on accuracy *and* latency *and* size *and* training time
(6 encoder layers vs 12). No weights are committed; `artifacts/` is gitignored.

Three findings worth carrying forward:

1. **Fine-tuning is where the gains are, not head capacity.** Frozen embeddings +
   MLP (0.761) ≈ frozen + linear (0.762). The frozen space is already linearly
   separable; adding curvature buys nothing. Fine-tuning the encoder buys 12 points.
2. **~6 points of accuracy is benchmark formatting, not topic.** The
   `question_only` ablation costs TF-IDF 6.4 points (0.810 → 0.746) and the
   fine-tuned model 6.3 (0.886 → 0.823). That the effect is near-identical across
   two very different model families makes it a property of the *data*, not the
   model — and it will not transfer to open user traffic.
3. **Calibration is close to free and sometimes essential.** `bge-small` is badly
   overconfident raw (ECE 0.140) and lands at 0.059 after temperature scaling;
   MiniLM was already calibrated (T ≈ 1.01). Since every routing decision is a
   confidence threshold, this is not cosmetic.

### Where the domain head underperforms

Full analysis: `python -m router.cli analyze 07_finetune_minilm`, which writes
per-class F1, the confusion matrix, slices and confidently-wrong rows to
`artifacts/domain/07_finetune_minilm/analysis.md`.

Per-class F1 spans 0.82 to 0.98:

| class | F1 | | class | F1 |
|---|---|---|---|---|
| history | 0.822 | | technology | 0.877 |
| philosophy_psychology | 0.851 | | social_science | 0.887 |
| arts_recreation | 0.862 | | literature | 0.897 |
| science | 0.862 | | cs_general | 0.927 |
| | | | **language** | **0.976** |

Row-normalised confusion (rows = true label, diagonal bolded):

| true \ pred | arts | cs | hist | lang | lit | phil | sci | soc | tech |
|---|---|---|---|---|---|---|---|---|---|
| arts_recreation | **.80** | .03 | .08 | — | .06 | — | .02 | .01 | .01 |
| cs_general | .01 | **.93** | .02 | — | — | .01 | .02 | — | .01 |
| history | .02 | .05 | **.87** | — | .03 | — | .04 | — | — |
| language | — | — | .01 | **.96** | — | — | .02 | .01 | — |
| literature | .02 | — | .01 | — | **.95** | .01 | — | .01 | — |
| philosophy_psych | — | .01 | .02 | — | .05 | **.82** | .02 | .04 | .05 |
| science | .01 | .02 | .04 | — | .01 | .01 | **.88** | .01 | .03 |
| social_science | — | .01 | — | — | .03 | .01 | .03 | **.90** | .02 |
| technology | — | .01 | — | .01 | — | .03 | **.08** | .02 | **.85** |

**The errors are not where the task is hard — they are where the taxonomy is
arbitrary.** Three things say so:

1. **Difficulty barely moves accuracy.** easy 0.876 / medium 0.902 / hard 0.885.
   Essentially flat, and *medium is the best*. If the model were failing on
   genuinely hard prompts this would slope downward.
2. **The single largest confusion is `technology → science` (16 cases), and 11
   of those 16 are `MMLUPro_health`.** Medical questions are filed under
   `technology` because Dewey class 6 covers Medicine — so the model calls a
   medical question "science", which is the more defensible answer. It is being
   penalised for disagreeing with a shelving convention.
3. **The worst categories are the arbitrary ones**: `70 Arts` (44% error),
   `15 Psychology` (36%). Both sit on boundaries the taxonomy draws and a human
   would not.

Conversely `language` at 0.976 F1 is not a triumph — that class is largely the
WMT translation subset, which is trivially separable because the prompt contains
non-English text. It is the same format-artifact effect the `question_only`
ablation measures, showing up as a suspiciously easy class.

Net: the remaining ~11% of errors are mostly *label noise rather than model
capacity*, so more capacity will buy less than the headline number suggests.
This is the concrete evidence behind priority 0.

**Latency in context:** the 8.87 ms head gates an LLM call that takes 2–6 s, so it
is ~0.2% of turn latency while cutting ~55% of cost. Measured single-prompt on
Apple Silicon (MPS), unbatched — a CPU-only server will be slower and is
unmeasured. `02_tfidf_logreg_wordchar` at 1.06 ms is the sub-millisecond fallback
at −7.5 points.

---

## Status

| Component | State |
|---|---|
| Domain head | **Trained.** 10-experiment sweep; best 0.886 test accuracy / 0.885 macro-F1. |
| Calibration | **Done.** Temperature fit on validation; ECE 0.20 → 0.03. |
| Policy, pool, skills, orchestrator, gate, session | **Built and tested** (90 tests), runs fully offline. |
| Difficulty, task type | **Heuristic placeholders** behind the interfaces the trained heads will use. |
| Pool `quality` scores | **Priors, not measurements.** |

See **[NEXT_STEPS.md](NEXT_STEPS.md)** for priorities, known gaps and open questions.
