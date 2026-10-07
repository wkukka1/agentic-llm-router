# MOD-001: `has_code_shape` and `has_math` fire on ordinary prose

- **Status:** reproduced (`known_bugs/_repros/repro_decompose.py`, section 1)
- **Severity:** low-med
- **Where:** `_CODE_SHAPE` ([surface.py:43-46](../../../src/decompose/classifiers/prompt_decomposition/signals/surface.py#L43-L46)) and `_MATH` ([surface.py:49](../../../src/decompose/classifiers/prompt_decomposition/signals/surface.py#L49)), used by `extract` and so by `SurfaceClassifier`
- **Found:** 2026-09-24

## What

**`has_code_shape`** includes `def|class|import|from|function|const|let|var|public|private`
in an unanchored keyword alternation. Those are ordinary English words, and the
match is per-word, so any prompt containing one sets the flag.

**`has_math`** includes a bare `\$\$?` (any dollar sign), and `\b\d+\s*[+\-*/=]\s*\d+`,
which matches hyphenated numbers as well as arithmetic.

## Failure scenario

| prompt | flag |
|---|---|
| `Translate this sentence from French to English` | `has_code_shape=1` |
| `What class should I take next semester?` | `has_code_shape=1` |
| `How much is $5 in euros?` | `has_math=1` |
| `What happened on 2024-01-15?` | `has_math=1` |
| `Convert 10-20 into a range` | `has_math=1` |

(The keywords are case-sensitive, so a sentence-initial `Let` / `From` is missed
while lowercase `let` / `from` mid-sentence hits. The flag is inconsistent as
well as noisy.) `SurfaceClassifier` documents these as "does this contain a code
fence" style preconditions and emits them unmodified; the head features built
from `extract` inherit the noise.

## Suggested fix

Drop the prose-collision keywords from `_CODE_SHAPE` (or require code context,
e.g. `def \w+\(`, `import \w+`, `from \w+ import`), and narrow `_MATH` (`$` only
when paired with a closing `$` or a LaTeX command; digits-around-a-hyphen need
whitespace or a math context).

**Retraining caveat:** the length and attribute heads fitted on the `surface` /
`surface+embedding` feature sets learned on these exact values, and
`SURFACE_FEATURES` order/meaning is shared with training. Changing the regexes
changes the features, so those heads must be retrained (and the composite
vector's `surface.*` columns re-baselined) in the same change.
