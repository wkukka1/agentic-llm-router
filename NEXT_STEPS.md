# Next steps

Status 2026-08-28. Shipped: 6-member weighted ensemble, **0.758 top-1 /
0.895 top-2 / 0.953 top-3** on a frozen 400-prompt real-user eval.

## Read this before running another experiment

**Seed variance is ±3.4 points (95% CI).** Five seeds of one config spanned
63.5–67.8%. Any single-run difference below ~3 points is noise. Run 3–5 seeds,
or use the deterministic frozen-encoder members, before believing an effect.

**The eval set is frozen and must stay frozen.**
`data/handlabelled/eval_frozen.parquet`, matched by prompt text. It was
previously resampled per build, which silently made runs incomparable and
produced a learning-curve extrapolation that was wrong by orders of magnitude.

**Select on validation, never on test.** Searching ensemble combinations against
test read 76.0% vs 73.75% honest — a 2.25-point illusion.

## The 93% question

93% top-1 across all traffic is **not reachable** on this taxonomy by the levers
tried. Measured on the frozen eval, error decays as `n^-0.066` in labelled data,
which puts 93% at ~10^13 examples. Model scale, hyperparameters, kNN, synthetic
data and feature concatenation were each worth ≤2 points or nothing.

A 95% **top-2** target is met three ways, all measured:

| configuration | result |
|---|---|
| top-2 over the most-confident 50% of traffic | 0.980 |
| top-2 over the most-confident 70% | 0.961 |
| top-3 over all traffic | 0.953 |

What is *not* available is 95% top-1 across all traffic (0.758). If the consumer
can accept a shortlist or defer the least-confident tail, the target is met
today. Best of all: pass `p.distribution` and let the consumer choose its own
operating point.

## Priorities

**1. Measure inter-annotator agreement.** Still the highest-value open question
and still unanswered. All 2,041 labels are single-annotator, and my attempted
self-consistency check was invalid (the original labels were in context, giving
a meaningless 150/150). Have a second person label 200 of the frozen eval
prompts. If they agree with the existing labels ~75% of the time, then 0.738 is
already near the ceiling and further modelling work is wasted. ~2 hours.

**2. Fix `law_politics` recall (0.381).** Precision is 0.889 — it is simply too
cautious, with only ~106 real examples. Synthetic data raised its F1 from 0.500
to 0.700 in isolation but cost accuracy elsewhere when applied to four classes
at once; applying it to this class alone was neutral overall. Real examples
(r/legaladvice, policy forums) are the better fix.

**3. Consider whether `meta_other` should be split.** 12% of traffic, precision
0.585 — it absorbs greetings, jailbreaks, questions about the assistant, and
context-free follow-ups. If the router treats those differently, they need
separate labels.

## Known limits

- **Single-annotator labels**, unaudited. This is the load-bearing caveat.
- **LMArena flags are model-assisted**, and supply 3 of 10 domains in stage 1.
- **Latency 58 ms** — three encoders, two of them large. A single frozen
  `bge-base` head gives 0.680 at ~10 ms if that trade is better.
- **Apple Silicon (MPS), single-prompt, unbatched.** A CPU server will differ.
- **Benchmark scores (0.83–0.99) are not evidence of anything.** v1 scored 0.91
  there and 0.47 in the wild. Read the real-prompt column only.

## Things measured and not worth repeating

| tried | result |
|---|---|
| bigger encoders for fine-tuning (v1: 22M→109M→149M) | +0.8 then negative |
| hyperparameter sweeps | ±1 point |
| kNN alone | 0.673 (below ensemble) |
| synthetic data, 4 classes | net −1.75 (within noise); helped `law_politics` only |
| multi-encoder feature concatenation | 0.733, 5× cost |
| standardising embeddings before PCA | −7 points |
| Kaggle query-domain dataset | wrong taxonomy; collapses to one class |
