"""Shared, composable test fixtures/builders.

Kept deliberately small and orthogonal:

* :mod:`helpers.configs`  -- ``Config`` stand-ins (chance-correction, family maps)
* :mod:`helpers.stores`   -- ``EmbeddingStore`` builders
* :mod:`helpers.faiss_bank` -- FAISS ``QueryBank`` builder
* :mod:`helpers.nirt`     -- the ``NIRTModel`` synthetic IRT world + datasets/cfg
* :mod:`helpers.baseline` -- ``training.nirt.baseline`` training-config builder

Import what you need directly, e.g. ``from helpers import make_store, chance_cfg``.
"""

from __future__ import annotations

from helpers.baseline import baseline_cfg
from helpers.configs import ChanceCfg, FamilyCfg, chance_cfg
from helpers.faiss_bank import make_faiss_bank
from helpers.nirt import (
    FakeTrainingData,
    make_split_obs,
    make_synthetic_irt,
    nirt_cfg,
    nirt_datasets,
    toy_pairwise_arrays,
)
from helpers.stores import make_store

__all__ = [
    "ChanceCfg",
    "FamilyCfg",
    "FakeTrainingData",
    "baseline_cfg",
    "chance_cfg",
    "make_faiss_bank",
    "make_split_obs",
    "make_store",
    "make_synthetic_irt",
    "nirt_cfg",
    "nirt_datasets",
    "toy_pairwise_arrays",
]
