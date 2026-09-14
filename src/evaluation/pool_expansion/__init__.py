"""Candidate-pool-expansion workstream.

Re-runs the *identical* Phase 1 / Phase 2 evaluation battery each time the
candidate pool changes, and records how the pool's routing ceiling and the ZOIB
router's achievable savings move as models are added. See
``docs/pool_expansion_results.md`` and ``artifacts/pool_expansion/ledger.json``.

The predictor (frozen ZOIB head + Phase 1 Bernoulli control) is never retuned --
the pool is the only variable.
"""

from .battery import (
    LEDGER_COLUMNS,
    ablation_pools,
    append_ledger,
    derived_headline,
    ledger_row,
    render_ledger_md,
    run_battery,
)

__all__ = [
    "run_battery",
    "append_ledger",
    "render_ledger_md",
    "ledger_row",
    "derived_headline",
    "ablation_pools",
    "LEDGER_COLUMNS",
]
