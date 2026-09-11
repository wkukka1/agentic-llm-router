"""Serving heads: one module per classifier, composed by :mod:`~router.heads.composite`.

    base       loading a run directory, temperature calibration, entropy
    domain     what the prompt is about          0.923 top-1 / 0.980 top-2
    task       what it asks to be done           0.844 top-1 / 0.967 top-2
    composite  both over one prompt, plus the feature vector for the next stage

Adding a classifier is adding a module here and exporting it below -- nothing
in the existing heads changes. :mod:`router.inference` re-exports everything for
callers written before the split.
"""

from router.heads.base import CalibratedHead
from router.heads.composite import RouterHead, RouterPrediction
from router.heads.domain import DomainHead, DomainPrediction
from router.heads.task import TaskHead, TaskPrediction

__all__ = [
    "CalibratedHead",
    "DomainHead",
    "DomainPrediction",
    "RouterHead",
    "RouterPrediction",
    "TaskHead",
    "TaskPrediction",
]
