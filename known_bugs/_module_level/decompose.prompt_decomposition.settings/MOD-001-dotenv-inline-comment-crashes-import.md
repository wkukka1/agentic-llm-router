# MOD-001: A `.env` value with an inline comment crashes `settings()`, and so importing the encoder

- **Status:** reproduced (`known_bugs/_repros/repro_decompose.py`, section 5)
- **Severity:** low
- **Where:** `_load_dotenv` / `settings` ([settings.py:29-93](../../../src/decompose/classifiers/prompt_decomposition/settings.py#L29-L93)); consumer [encoder.py:38](../../../src/decompose/classifiers/prompt_decomposition/encoder.py#L38)
- **Found:** 2026-09-24

## What

`_load_dotenv` splits on the first `=` and strips whitespace and quotes; it does
not handle trailing `# comment`s, an `export ` prefix, or empty values.
`settings()` then does `int(get("ROUTER_SEED", ...))`.

## Failure scenario

```
ROUTER_SEED=123  # my seed
```

gives `values["ROUTER_SEED"] == "123  # my seed"` and
`ValueError: invalid literal for int() with base 10: '123  # my seed'`.

`encoder.py` evaluates `CACHE_DIR = settings().embedding_cache_dir` at module
scope, so the error surfaces as a failure to *import*
`decompose.classifiers.prompt_decomposition.encoder`, and with it every head
that lazily imports the encoder (`LengthHead`, `AttributeHead`, the
frozen-encoder domain/task models) at first prediction, far from the `.env` that
caused it. Likewise `export ROUTER_SEED=1` yields the key `export ROUTER_SEED`
(silently ignored), and an env var that is set but empty gives `int("")` or
`Path("")`.

The `.env` file is also located relative to the current working directory, so
the same code behaves differently depending on where it is launched from.

## Suggested fix

Strip an unquoted ` #...` tail, accept an `export ` prefix, and treat an empty
value as unset. Wrap the `int()` conversion so the error names the variable and
its source. Consider resolving `.env` from the project root rather than the
cwd, and evaluating `CACHE_DIR` lazily.
