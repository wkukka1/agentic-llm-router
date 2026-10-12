# LENGTHHEAD-001: `expected_tokens` can be negative

- **Status:** reproduced (`known_bugs/_repros/repro_decompose.py`, section 2)
- **Severity:** low-med
- **Class:** `decompose.classifiers.prompt_decomposition.heads.length.LengthHead` ([length.py:93-101](../../src/decompose/classifiers/prompt_decomposition/heads/length.py#L93-L101)); surfaces through `LengthClassifier` ([prompt_heads.py:146-155](../../src/decompose/classifiers/prompt_heads.py#L146-L155))

  (Not to be confused with `router.nirt.components.LengthHead`, an unrelated NIRT module.)
- **Found:** 2026-09-24

## What

`predict_batch` turns the ridge output into a token count with
`expected_tokens=float(np.expm1(value))`. The ridge regresses `log1p(tokens)`
linearly over standardised features and nothing bounds it below, so any prompt
whose features sit far below the training mean can predict `value < 0`, and
`expm1` of a negative number is a negative token count.

## Failure scenario

Reproduced on a small surface-feature head fitted to `tokens ~ prompt size`:

| prompt | `expected_tokens` | `log_tokens` | bucket |
|---|---|---|---|
| `""` | -0.63 | -0.984 | short |
| `"hi"` | -0.12 | -0.133 | short |

`LengthClassifier.classify` passes the value straight through as the
`expected_tokens` signal, so a negative "expected output size" reaches any
consumer that multiplies it by a per-token price (a negative cost estimate) or
takes its log. The magnitude is small, but empty and one-word prompts are
exactly the ones a router sees as trivial and routes on cost.

`LengthPrediction.probability_over` is unaffected (it works in log space).

## Suggested fix

Floor the log estimate at 0 in `LengthHead.predict_batch`
(`value = max(value, 0.0)`) or clamp `expected_tokens` with `max(0.0, ...)`.
Flooring the log value also keeps `log_tokens` (used in `RouterPrediction`'s
vector) consistent with `expected_tokens`.
