"""Generic ``[query_id x model_id]`` frame pivoting.

A plain pandas pivot with no training/eval-specific logic. Lives in
``router`` (not ``training.data.response_matrix``, its original home)
because serving-side prediction (:mod:`router.nirt.predict`) needs it and
must not import ``training``.
"""

from __future__ import annotations

import pandas as pd


def pivot_qm(frame: pd.DataFrame, value: str, *, aggfunc: str = "mean") -> pd.DataFrame:
    """Dense ``[query_id x model_id]`` matrix of ``frame[value]``.

    The one place the ``pivot_table(index="query_id", columns="model_id", ...)``
    idiom lives -- shared by the response matrix, the Phase 1 facade, the
    routing layer, and the classical-IRT baselines.
    """
    return frame.pivot_table(index="query_id", columns="model_id", values=value, aggfunc=aggfunc)
