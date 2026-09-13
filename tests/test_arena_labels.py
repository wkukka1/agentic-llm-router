"""The free labels: eleven columns nobody had to annotate.

The tests are about alignment rather than accuracy. A label frame joined on the
wrong key returns all-NaN or, worse, silently shifted rows -- and a head trained
on shifted labels reports a plausible number that means nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prompt_decomposition.core.arena_labels import FREE_LABELS, label_for


@pytest.fixture
def labels():
    return pd.DataFrame(
        {"is_code": [True, False, True], "math_v0.1.math": [False, True, False]},
        index=pd.Index(["a", "b", "c"], name="arena_id"),
    )


def test_a_label_comes_back_in_the_order_asked_for(labels):
    """Not the order it is stored in -- that is the shifted-rows failure."""
    np.testing.assert_array_equal(label_for(labels, ["c", "a", "b"], "code"), [1.0, 1.0, 0.0])


def test_a_missing_row_is_nan_rather_than_dropped(labels):
    """Dropping would silently misalign the label against the feature matrix."""
    out = label_for(labels, ["a", "absent"], "code")
    assert out[0] == 1.0 and np.isnan(out[1])


def test_an_unknown_label_is_rejected_by_name(labels):
    with pytest.raises(KeyError, match="unknown label"):
        label_for(labels, ["a"], "difficulty")


def test_every_advertised_label_has_a_column_name():
    assert all(isinstance(v, str) and v for v in FREE_LABELS.values())
    assert len(set(FREE_LABELS.values())) == len(FREE_LABELS)
