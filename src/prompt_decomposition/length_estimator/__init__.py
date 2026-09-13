"""Expected output length: the one router signal whose labels are free.

Every arena row carries both models' responses and an exact token count for
each, so this head is trained on 30,001 prompts nobody annotated. Length is the
dominant term in what a generation costs, which makes this the signal a
cost-aware router reads first.
"""

from prompt_decomposition.core.arena_corpus import (
    build_arena_corpus,
    load_arena_splits,
    save_arena_corpus,
)
from prompt_decomposition.core.features import FEATURE_SETS, build_features
from prompt_decomposition.length_estimator.head import LengthHead, LengthPrediction
from prompt_decomposition.length_estimator.model import LengthModel

__all__ = [
    "FEATURE_SETS",
    "LengthHead",
    "LengthModel",
    "LengthPrediction",
    "build_features",
    "build_arena_corpus",
    "load_arena_splits",
    "save_arena_corpus",
]
