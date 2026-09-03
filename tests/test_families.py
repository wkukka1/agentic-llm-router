from __future__ import annotations

import pandas as pd

from router._normalize import unit_rows
from router.config import coerce_auto_bool, coerce_hidden
from router.data.families import family_labels, family_of, task_family_map


class _Cfg:
    """Minimal stand-in for router.config.Config for the family map."""

    def __init__(self, families):
        self._f = families

    def get(self, dotted, default=None):
        return self._f if dotted == "profiles" else default


def test_task_family_map_lowercases_and_flattens():
    m = task_family_map(_Cfg({"task_families": {"math": ["GSM8K", "MATH"], "code": ["MBPP"]}}))
    assert m == {"gsm8k": "math", "math": "math", "mbpp": "code"}


def test_family_of_exact_then_substring():
    m = {"gsm8k": "math", "mbpp": "code"}
    assert family_of("gsm8k", m) == "math"                 # exact
    assert family_of("routerbench:mbpp:0001", m) == "code" # substring
    assert family_of("hellaswag", m) is None


def test_family_labels_fallback():
    cfg = _Cfg({"task_families": {"math": ["gsm8k"]}})
    qdf = pd.DataFrame({"query_id": ["a", "b"], "dataset": ["gsm8k", "arc"]})
    assert family_labels(["a", "b", "c"], cfg, queries_df=qdf) == ["math", "other", "other"]


def test_coerce_helpers():
    assert coerce_hidden(64) == 64
    assert coerce_hidden("128") == 128
    assert coerce_hidden("none") is None
    assert coerce_hidden(0) is None
    assert coerce_hidden(None, default=32) == 32
    assert coerce_auto_bool("auto", True) is True
    assert coerce_auto_bool("auto", False) is False
    assert coerce_auto_bool(None, True) is True
    assert coerce_auto_bool(False, True) is False
    assert coerce_auto_bool(1, False) is True


def test_unit_rows():
    import numpy as np

    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    u = unit_rows(x)
    assert np.allclose(np.linalg.norm(u[0]), 1.0)
    assert np.allclose(u[1], 0.0)          # zero row stays zero (eps floor)
