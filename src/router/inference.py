"""Backwards-compatible re-export of :mod:`router.heads`.

The serving code used to live here as one file. It is now a package with one
module per classifier, so adding a head is adding a file rather than growing
this one:

    router.heads.base       loading, temperature calibration, entropy
    router.heads.domain     DomainHead, DomainPrediction
    router.heads.task       TaskHead, TaskPrediction
    router.heads.composite  RouterHead, RouterPrediction, the feature vector

Import from :mod:`router.heads` in new code. This module stays so that
``from router.inference import DomainHead`` keeps working, and because the
serving entry point is a public surface other repositories may already be
pinned to.
"""

from __future__ import annotations

from router.heads import (
    CalibratedHead,
    DomainHead,
    DomainPrediction,
    RouterHead,
    RouterPrediction,
    TaskHead,
    TaskPrediction,
)

#: Kept under its old private name: the split renamed it, and anything that
#: subclassed it outside this package would otherwise break silently.
_CalibratedHead = CalibratedHead

__all__ = [
    "CalibratedHead",
    "DomainHead",
    "DomainPrediction",
    "RouterHead",
    "RouterPrediction",
    "TaskHead",
    "TaskPrediction",
]
