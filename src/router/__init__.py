"""Serving. Everything here runs at inference time, and imports nothing above it.

The dependency rule is one-way and load-bearing: `router` may not import
`training` or `evaluation`. It is what keeps a serving deployment from dragging
in the training stack, and what stops an offline convenience from quietly
becoming a serving dependency. `tests/test_layering.py` enforces it.
"""

from router.composite import RouterHead, RouterPrediction
from router.heads.attributes import AttributeHead, AttributePrediction
from router.heads.base import CalibratedHead
from router.heads.domain import DomainHead, DomainPrediction
from router.heads.length import LengthHead, LengthPrediction
from router.heads.task import TaskHead, TaskPrediction

__all__ = [
    "AttributeHead", "AttributePrediction", "CalibratedHead", "DomainHead",
    "DomainPrediction", "LengthHead", "LengthPrediction", "RouterHead",
    "RouterPrediction", "TaskHead", "TaskPrediction",
]
