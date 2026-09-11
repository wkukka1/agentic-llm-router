"""Signals extracted from a prompt, cheapest first.

    surface   regex and arithmetic over the string. Free, no labels, no drift.

Learned signals live in :mod:`router.heads` -- they need a trained run directory
and are a different kind of object. The split is by cost, because a router's
job is to spend less: surface signals can gate the expensive ones.

See SIGNALS.md for what is worth building next and what has been measured
impossible.
"""

from router.signals.surface import (
    SURFACE_FEATURES,
    SurfaceSignals,
    extract,
    extract_many,
)

__all__ = [
    "SURFACE_FEATURES",
    "SurfaceSignals",
    "extract",
    "extract_many",
]
