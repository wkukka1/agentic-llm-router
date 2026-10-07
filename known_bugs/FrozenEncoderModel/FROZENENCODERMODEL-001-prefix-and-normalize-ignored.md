# FROZENENCODERMODEL-001: `prefix` and `normalize` params never reach the encoder

- **Status:** reproduced (`known_bugs/_repros/repro_decompose.py`, section 3)
- **Severity:** low-med
- **Class:** `_FrozenEncoderModel` and its registered subclasses `embed_logreg` / `embed_mlp` ([decompose/classifiers/prompt_decomposition/models.py:228-247](../../src/decompose/classifiers/prompt_decomposition/models.py#L228-L247))
- **Found:** 2026-09-24

## What

`_FrozenEncoderModel.__init__` collects extra keyword arguments into
`self.params` (`**kw`), so `prefix=` and `normalize=` are accepted without
complaint. The lazy `encoder` property then builds `EmbeddingEncoder` from only
`encoder_model`, `pooling`, `max_length`, `batch_size` and `device`; `prefix`
and `normalize` are never forwarded.

`_ENCODER_PARAMS` (line 215) lists `prefix` as one of the parameters that
"determine what the encoder produces", and `load` refuses a pickle whose saved
`prefix` differs from the instance's. That guard therefore compares a value that
has no effect on the encoder.

## Failure scenario

```python
m = build("embed_logreg", encoder_model="BAAI/bge-small-en-v1.5",
          prefix="Represent this sentence: ", normalize=False)
m.params["prefix"]     # 'Represent this sentence: '
m.encoder.prefix       # ''            <- ignored
m.params["normalize"]  # False
m.encoder.normalize    # True          <- ignored
```

A YAML config that sets a non-default prefix (say `passage: ` for an e5 model,
or a BGE query instruction) trains and serves with the encoder's own default
instead, with no warning. Train and serve agree with each other (both use the
same broken path), so nothing fails; the config just does not mean what it says,
and an experiment comparing prefixes would compare identical runs.

## Suggested fix

Forward `prefix=p.get("prefix")` and `normalize=p.get("normalize", True)` when
building `EmbeddingEncoder`, and give both explicit `__init__` parameters so a
typo is not swallowed by `**kw`. Existing trained artifacts that recorded a
`prefix` in `params` would then start serving a different encoder, so check the
saved `config.yaml` files before changing the behaviour.
