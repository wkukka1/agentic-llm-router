"""Back-compat shim -- the continuous synthetic gates now live in
:mod:`router.nirt.synthetic` alongside the Bernoulli one."""

from __future__ import annotations

from .synthetic import (  # noqa: F401
    Synthetic,
    SyntheticContinuous,
    make_synthetic_continuous,
    recovery_report,
    to_arrays,
)
