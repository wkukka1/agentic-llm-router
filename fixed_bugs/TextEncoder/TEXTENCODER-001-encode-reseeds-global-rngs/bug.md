# TEXTENCODER-001: every `encode()` call reseeds the process-global `random` / numpy / torch RNGs

- **Status:** fixed 2026-10-02 (see [fix.md](fix.md)); reproduced earlier with `known_bugs/_repros/repro_encode_reseeds_globals.py`
- **Severity:** med-low
- **Class:** `router.embeddings.encoder.TextEncoder` (`iter_encode`, [encoder.py:150](../../../src/router/embeddings/encoder.py#L150); also the lazy `model` property, [encoder.py:121](../../../src/router/embeddings/encoder.py#L121))
- **Found:** 2026-09-24

## What

`iter_encode` opens with `_seed_everything(self.cfg.seed, deterministic_algorithms=False)`,
and `encode()` goes through `iter_encode`. `seed_everything` calls
`random.seed`, `np.random.seed`, `torch.manual_seed` and
`torch.cuda.manual_seed_all` ([determinism.py:27-34](../../../src/router/determinism.py#L27-L34)),
all of which are process-global. The comment on the `model` property shows the
authors already thought about the serving path ("this also runs on the serving
path next to training / GPU jobs" and so skips the deterministic-algorithms
switch), but the RNG reseed itself was left in.

The live routers call it per request: `NIRTRouter` and `KNNRouter` go through
`_encode_texts` -> `encoder.encode` ([routers.py:122-150](../../../src/router/routing/routers.py#L122-L150)).

## Failure scenario

An application embedding the router seeds and uses its own RNG (A/B
assignment, request jitter, sampling of fallbacks):

```
random.seed(1234); np.random.seed(1234)
for each request: draw = (random.random(), np.random.rand()); router.route(...)
draws: (0.9665, 0.1915)  (0.6394, 0.3745)  (0.6394, 0.3745)
```

After the first routed request the stream is pinned to the seed-42 state, so
every later draw repeats. It also resets torch's global generator mid-training
if anything encodes inside a training loop (a `DataLoader` shuffle or dropout
mask then restarts from the same state).

`RandomRouter` is unaffected (it uses its own `default_rng`); this bites
*other* code sharing the process.

## Why the seed is not needed here

Encoding runs in eval mode under `torch.no_grad()` with no dropout; the seed
only mattered at model load (`SentenceTransformer` / `AutoModel` init), which
the `model` property already covers once.

## Suggested fix

Drop the reseed from `iter_encode` (keep the one-time seed around model load,
ideally scoped: save/restore the RNG states with `random.getstate()` /
`np.random.get_state()` / `torch.random.fork_rng()`).
