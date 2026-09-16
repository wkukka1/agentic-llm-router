"""Phase 0 data layer: loaders, canonical schemas, normalization,
response matrix, chance correction, splits, quality checks, and the
training-data facade (:mod:`training.data.facade`).

The serving-side batch container, ``NIRTDataset``, lives in
:mod:`router.data.nirt` instead -- live routers construct it directly and
must not import this package."""
