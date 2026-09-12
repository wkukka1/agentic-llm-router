"""Expected output length: the one router signal whose labels are free.

Every arena row carries both models' responses and an exact token count for
each, so this head is trained on 30,001 prompts nobody annotated. Length is the
dominant term in what a generation costs, which makes this the signal a
cost-aware router reads first.
"""

from prompt_decomposition.length_estimator.data import (
    build_length_dataset,
    load_length_splits,
    save_length_dataset,
)
from prompt_decomposition.length_estimator.head import LengthHead, LengthPrediction
from prompt_decomposition.length_estimator.model import FEATURE_SETS, LengthModel, build_features

__all__ = [
    "FEATURE_SETS",
    "LengthHead",
    "LengthModel",
    "LengthPrediction",
    "build_features",
    "build_length_dataset",
    "load_length_splits",
    "save_length_dataset",
]
