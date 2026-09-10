"""Back-compat shim -- the continuous synthetic gates live in
:mod:`router.nirt.baseline.synthetic` alongside the Bernoulli one. Import from
there directly; this module only exists for callers that predate the merge and
will be removed once they migrate."""

from __future__ import annotations

from .synthetic import (  # noqa: F401
    Synthetic,
    make_synthetic_continuous,
    recovery_report,
    to_arrays,
)
