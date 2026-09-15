"""Free deterministic features, read straight off the prompt string."""

from decompose.classifiers.prompt_decomposition.signals.surface import (
    SURFACE_FEATURES,
    SurfaceSignals,
    extract,
    extract_many,
)

__all__ = ["SURFACE_FEATURES", "SurfaceSignals", "extract", "extract_many"]
