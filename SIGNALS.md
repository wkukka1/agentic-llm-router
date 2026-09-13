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
| **expected output length** | rho 0.573 with the large encoder, 0.514 with a small one, ceiling 0.607 | free if the domain head already ran the encoder |

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

**2. Expected output length.** ✅ **Built** — `prompt_decomposition.length_estimator`,
Spearman 0.573 with the large encoder and 0.514 with a small one, against a
ceiling of 0.607. The target was free: every arena row carries both models'
responses with an exact token count. See the measurement section below.

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

### Response length: built, and it is the strongest prompt signal there is

**Superseded measurement.** The first pass at this scored *character* counts on
38,434 rows and reported rho +0.542. The shipped head is measured against exact
*token* counts -- `sum_assistant_a_tokens` / `sum_assistant_b_tokens`, which the
arena dump carries per row -- over 109,335 single-turn prompts, deduplicated on
normalised text and split by hashing the prompt. Different target, bigger
corpus, tighter protocol; the numbers below replace the ones above.

The target is the mean of `log1p(tokens)` across both models. Raw counts span 1
to 88,300, and the routing question is multiplicative ("200 tokens or 2,000"),
so log space is the honest scale.

**There is a ceiling, and it is not 1.0.** Given the *same* prompt, the two
arena models agree with each other on length at Spearman **0.607**. How long the
answer runs is partly a property of who answers. No prompt-only feature can
beat the target's own reproducibility, so every score below is read against
0.607 rather than against a perfect predictor.

76,697 train / 16,538 val / 16,100 test. Alpha on validation, test read once:

| features | Spearman | 95% CI | R² | typical error | AUC (top quartile) |
|---|---|---|---|---|---|
| constant (predict the mean) | — | — | 0.000 | ×2.23 | — |
| prompt length alone | 0.252 | [0.236, 0.266] | 0.060 | ×2.18 | 0.605 |
| 23 free surface features | 0.353 | [0.339, 0.367] | 0.132 | ×2.12 | 0.664 |
| 384-d encoder | 0.500 | [0.488, 0.512] | 0.290 | ×1.97 | 0.735 |
| **384-d + surface** | 0.514 | [0.502, 0.526] | 0.309 | ×1.96 | 0.739 |
| **1024-d + surface** | **0.573** | [0.562, 0.585] | 0.399 | ×1.88 | 0.764 |

The encoder is where the signal is: **0.353 free, 0.514 small encoder, 0.573
large one, against a 0.607 ceiling.** The large encoder reaches 94% of what two
models manage against each other. Surface features add +0.014 on top of an
encoder and are the whole story without one.

**The large encoder is already paid for.** `intfloat/e5-large-v2` is the
top-weighted member of the shipped domain ensemble (0.397). A router that has
already classified the domain has that embedding in hand, so the strongest
length head costs one matrix multiply rather than a second encoder pass. The
384-d model exists for the case where it does not.

**What the head is for, in numbers.** Ranked by predicted length, the top 25% of
traffic holds **37.9%** of all tokens generated -- against 25% for a random
quarter and 53.6% for a perfect ranking. It captures 45% of the achievable gain
over chance. As a gate for "this will be a long answer", the top 5% is 0.596
precision against a 0.250 base rate.

Grouped by *predicted* quartile, the levels are honest -- predicted 281 tokens
arrives at 295, predicted 1,038 arrives at 1,071 -- which is what lets the
estimate feed a cost model rather than only a ranking. The spread is wide and
stays wide: one residual sigma of 0.889 means the true length is typically
within a factor of 2 of the estimate, and the 80% interval holds 83.3% of rows,
so the head is slightly conservative rather than over-confident.

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

### Distance from training data: adds nothing over confidence

Proposed as an abstain signal — a prompt unlike anything in the training set is
one the classifier should not be trusted on. Built as four features (nearest
neighbour cosine, mean top-10 cosine, centroid similarity, centroid margin),
cross-fitted so a prompt is never in its own reference set, and scored on
whether it predicts a domain-classifier **error**:

| features | AUC predicting classifier error |
|---|---|
| **the classifier's own confidence** | **0.783 [0.765, 0.801]** |
| distance from training (4 features) | 0.663 [0.640, 0.689] |
| confidence + distance | 0.785 [0.766, 0.802] |

Combined gains **+0.002** over confidence alone. Per feature, three of the four
are at or below chance: `nn_similarity` 0.529, `knn_similarity` 0.505,
`centroid_similarity` 0.486. The one that carries signal — `centroid_margin`
at 0.656 — is a crude restatement of confidence, which is why adding it changes
nothing.

The 402 external prompts also sit as close to training as training prompts sit
to each other (knn 0.818 vs 0.829), so there is no distribution gap in the data
to detect even in principle. Module deleted; the measurement is the deliverable.

### Perplexity: worse than nothing

Also proposed. Scored with `distilgpt2` over both targets that matter:

| | classifier error | response length |
|---|---|---|
| perplexity alone | AUC 0.483 | rho +0.047 |
| baseline | 0.783 (confidence) | +0.317 (surface features) |
| baseline + perplexity | 0.783 | +0.320 |

Below chance on error, near-zero on length, and adds nothing to either baseline
— while costing a forward pass through a second model at serving time. Rejected.

### Underspecification: not predictable, idea dropped

`len_ratio` — how differently two models sized the same job — was the proposed
free proxy for ambiguity. Surface features reach rho +0.023, embeddings +0.044,
against −0.031 for length alone. Nothing. Either the proxy does not measure
ambiguity or ambiguity is not readable from the prompt; the data cannot separate
those and neither reading justifies building it.

### What this changes

1. **Build the length estimator with an encoder**, not with the surface
   features. Done: 0.573 against 0.353 for regex, and the large encoder is
   already running for the domain head so it costs nothing extra. The free
   features remain the fallback when no encoder pass is affordable.
2. **Stop trying to predict difficulty from the prompt.** Three independent
   feature families have now failed at it. The difficulty head must read model
   behaviour — response length, disagreement between generations, per-model
   history.
3. **Keep the surface features for gating, not for prediction.** They are
   preconditions (`has_code_fence`, `needs_recency`, `asks_format`), and they
   are free, deterministic and drift-proof. That is what they are good at.
4. **Drop the ambiguity signal** until a better target than `len_ratio` exists.
