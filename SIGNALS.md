# Signals

What can be read off a prompt, what it costs, and what it is worth. This is the
input side of the router: every routing decision downstream is a function of
these, so the honest question is not "what would we like" but "what survives
measurement".

Two classifiers exist today. This document is the map of what else is possible,
and — more usefully — what has already been measured **impossible**.

---

## The ideal set

If every signal were free and perfectly accurate, a router would want these.
Grouped by the decision each one actually serves, because a signal that does
not change a routing decision is not worth building however interesting it is.

### Gating — does this need a capability the cheap path lacks?

| signal | what it decides |
|---|---|
| **tool need** (search / code execution / image gen) | whether any language model alone can finish this |
| **recency need** | whether parametric knowledge is enough, or it must hit a search index |
| **has attached context** | whether the job is "read this" or "know this" |
| **modality** | text out vs image/video out — a different model family entirely |
| **safety / refusal risk** | whether it needs the careful model or a refusal path |
| **language** | whether the cheap model is competent in it |

These are the highest-value signals because they are *binary preconditions*.
Getting one wrong is not a quality regression, it is a failed request.

### Sizing — how much work is this?

| signal | what it decides |
|---|---|
| **expected output length** | the dominant term in generation cost |
| **decomposability / step count** | whether this goes straight to a model or to the orchestrator |
| **constraint count** | how many things must simultaneously hold in the answer |
| **reasoning depth** | chain-of-thought budget |

### Matching — which model suits this?

| signal | what it decides |
|---|---|
| **domain** ✅ built | which model's strengths apply |
| **task type** ✅ built | answer-shaped vs produce-shaped work |
| **difficulty** | weak vs strong model |
| **subjectivity** | whether "correct" even exists, and so whether accuracy matters |
| **ambiguity / underspecification** | whether to ask a clarifying question first |

---

## What is actually possible

Measured, not assumed. Sources are hand-labelled real chat traffic and 38,434
LMArena prompts carrying both models' responses and a human preference.

### Built and shipping

| signal | accuracy | cost |
|---|---|---|
| **domain**, 8 classes | 0.923 top-1 / 0.980 top-2 | 6 encoder passes |
| **task type**, 6 classes | 0.844 top-1 / 0.967 top-2 | 3 encoder passes + tf-idf |
| **surface features**, 23 | deterministic | free |

`prompt_decomposition.signals.surface` covers the deterministic layer: code fences, URLs,
maths notation, enumeration, requested output format, stated length limits,
persona framing, recency words, imperative-vs-interrogative opening. No model,
no labels, microseconds. These are the signals most likely to survive
distribution shift — traffic drifts, but "contains a code fence" does not.

### Measured impossible — do not spend time here

**Difficulty cannot be read from the prompt.** This is the most important
negative result in the project and it is worth stating precisely, because the
obvious next instinct is to throw a bigger model at it.

| features | AUC predicting "both models failed" |
|---|---|
| prompt length alone | 0.527 |
| our 24-dim router vector | **0.577** |
| full 1024-d sentence embedding | 0.567 |
| tf-idf bag of words | 0.508 |

A 1024-dimensional embedding does **worse** than a 24-dimensional summary of it,
and combining them is worse than either (0.541 — 1,048 features over 326
positives is simply overfitting). There is no headroom being left on the table;
~0.577 is close to everything prompt text contains.

The consequence for the architecture: **the difficulty head must read model
behaviour, not prompt text.** Response length, response disagreement, per-model
historical performance on similar prompts, self-reported uncertainty. The 24-dim
vector is worth passing as a minor input — it beats the domain label alone,
which is chance at 0.518 — but it cannot be the backbone.

**`extract` is not a viable task class in this traffic.** Targeted mining across
1,441 unlabelled prompts, specifically hunting for it, found one example. It is
0.3% of traffic. A class that rare cannot be learned or measured.

### Worth building next, in order

**1. Tool need — search / code / image.** The highest-value unbuilt signal,
because it is a gating decision rather than a quality one: routing a
"what happened today" prompt to a parametric model produces a confidently wrong
answer, not a slightly worse one. Partly reachable from surface signals already
(`needs_recency`, `has_code_shape`), which means a weak version exists for free
and a labelled version would be cheap to validate against it.

**2. Expected output length.** The cost driver, and unlike difficulty it has an
abundant free target: every arena row carries the responses both models actually
produced. No annotation needed — 38,434 labelled examples already in hand. See
the measurement section below.

**3. Decomposability.** The signal your orchestrator recursion actually turns
on: does this prompt contain multiple separable subtasks? Surface signals get
part of it (`n_enumerated_items`, `n_questions`) but the semantic version needs
labels. Worth ~200 hand-labelled prompts to find out whether it is learnable
before committing.

**4. Ambiguity / underspecification.** Also has a free proxy target:
`len_ratio`, how differently two models sized the same job. If two strong models
write 80 words and 800 words for the same prompt, the prompt did not determine
its own answer.

### Deliberately not pursued

**Subjectivity, reasoning depth, constraint count.** Each is plausible and each
needs hand labels with no free target available. Given that difficulty — the
signal with the strongest motivation of any of them — turned out to be
unreadable from prompt text, the prior on these is poor. Build them only after a
cheaper signal has been shown to be insufficient for a specific routing
decision.

---

## Measurement: what the free signals actually buy

38,434 LMArena prompts, each with both models' responses and a human preference.
5-fold cross-validated, scored against the cheapest possible baseline — prompt
length alone — so "better than nothing" is visible rather than assumed.

### Response length: learnable, and worth an encoder pass

| features | Spearman rho | AUC (top-quartile length) |
|---|---|---|
| prompt length alone | +0.227 | 0.618 |
| 23 surface features (free) | +0.335 | 0.673 [0.662, 0.686] |
| **1024-d sentence embedding** | **+0.542** | **0.744 [0.733, 0.754]** |
| surface + embedding | +0.552 | 0.745 |

**This is the signal to build next.** It is the cost driver, it already has
38,434 labels nobody has to annotate — every arena row carries the responses
both models actually produced — and it is genuinely predictable.

The encoder earns its keep here: rho +0.542 against +0.335 for regex. Surface
features add essentially nothing on top (+0.552, AUC 0.745), so this is an
encoder job, not a feature-engineering job. The free features remain a
reasonable fallback if an encoder pass is unaffordable, but they leave a lot on
the table.

Among the surface features, the interpretable one is `asks_length_limit` at
−0.143: asking for brevity does produce shorter answers, and the regex catches
it. `log_chars` and `log_words` dominate but are collinear and split weight.

### Difficulty: not learnable at any price

| features | AUC predicting "both models failed" |
|---|---|
| prompt length alone | 0.516 |
| 23 surface features | 0.562 [0.553, 0.570] |
| 1024-d sentence embedding | 0.565 [0.545, 0.587] |
| the full 24-dim classifier vector | 0.577 |
| surface + embedding | 0.564 |

Nothing separates from anything. Regex, a 1024-dimensional embedding, and two
trained classifiers all land between 0.56 and 0.58, and combining them does not
help. This is the same wall the earlier experiment hit from a different
direction.

### The asymmetry is the finding

Put the two tables side by side and the same feature sets behave oppositely:

| | surface | embedding | gain from semantics |
|---|---|---|---|
| response length | 0.673 | 0.744 | **+0.071** |
| both models failed | 0.530 | 0.565 | +0.035, from a far lower base |

**Semantics predict how much work a prompt needs. They do not predict whether
models will succeed at it.** Understanding what is being asked tells you the
size of the job and nothing about whether it is within reach — which is
intuitive in hindsight and is exactly why the difficulty head has to read model
behaviour rather than prompt text.

### Underspecification: not predictable, idea dropped

`len_ratio` — how differently two models sized the same job — was the proposed
free proxy for ambiguity. Surface features reach rho +0.023, embeddings +0.044,
against −0.031 for length alone. Nothing. Either the proxy does not measure
ambiguity or ambiguity is not readable from the prompt; the data cannot separate
those and neither reading justifies building it.

### What this changes

1. **Build the length estimator with an encoder**, not with the surface
   features. rho +0.542 vs +0.335 is the difference between useful and marginal,
   and the target is free.
2. **Stop trying to predict difficulty from the prompt.** Three independent
   feature families have now failed at it. The difficulty head must read model
   behaviour — response length, disagreement between generations, per-model
   history.
3. **Keep the surface features for gating, not for prediction.** They are
   preconditions (`has_code_fence`, `needs_recency`, `asks_format`), and they
   are free, deterministic and drift-proof. That is what they are good at.
4. **Drop the ambiguity signal** until a better target than `len_ratio` exists.
