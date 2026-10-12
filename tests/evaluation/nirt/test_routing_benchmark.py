"""Routing benchmark policies + summaries on tiny hand-checked pools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.nirt.routing_benchmark import combine, per_model_mean_cost, source_picks, summarize

MODELS = ["cheap", "mid", "gpt4"]
# 3 queries x 3 models
Y = np.array([[1.0, 1.0, 1.0],
              [0.0, 1.0, 1.0],
              [0.0, 0.0, 1.0]])
C = np.array([[0.1, 0.5, 2.0],
              [0.1, 0.5, 2.0],
              [0.1, 0.5, 2.0]])


def _picks(routers=None, lams=(0.0,)):
    return source_picks(Y, C, MODELS, "gpt4", routers or {}, lams)


def test_reference_always_picks_its_column():
    y, c, is_ref = _picks()["reference"]
    assert y.tolist() == [1.0, 1.0, 1.0] and c.tolist() == [2.0, 2.0, 2.0] and is_ref.all()


def test_oracle_best_quality_breaks_ties_to_cheapest():
    y, c, _ = _picks()["oracle_best_quality"]
    assert y.tolist() == [1.0, 1.0, 1.0]
    assert c.tolist() == [0.1, 0.5, 2.0]


def test_oracle_match_reference_is_cheapest_at_least_as_good():
    y, c, is_ref = _picks()["oracle_match_reference"]
    assert c.tolist() == [0.1, 0.5, 2.0]
    assert is_ref.tolist() == [False, False, True]


def test_router_uses_predictions_not_truth():
    pred = np.array([[0.2, 0.3, 0.9]] * 3)          # always prefers gpt4 at lam 0
    p = _picks({"r": (pred, C)}, lams=(0.0, 1.0))
    assert p["r@0"][2].all()
    # lam 1: utilities 0.1, -0.2, -1.1 -> cheap everywhere
    assert p["r@1"][1].tolist() == [0.1, 0.1, 0.1]


def test_per_model_cost_vector_is_accepted():
    pred = np.array([[0.2, 0.8, 0.9]] * 3)
    p = _picks({"r": (pred, np.array([0.1, 0.5, 2.0]))}, lams=(1.0,))
    assert p["r@1"][1].tolist() == [0.5, 0.5, 0.5]   # 0.8 - 0.5 = 0.3 beats 0.1 and -1.1


def test_summarize_relative_to_reference_and_combine():
    s = summarize(_picks())
    assert s["reference"]["q_vs_reference"] == 0.0 and s["reference"]["reference_share"] == 1.0
    m = s["oracle_match_reference"]
    assert m["cost_per_1k"] == pytest.approx(1e3 * (0.1 + 0.5 + 2.0) / 3)
    assert m["cost_vs_reference"] == pytest.approx((2.6 / 3) / 2.0 - 1)
    both = combine({"a": _picks(), "b": _picks()})
    assert len(both["reference"][0]) == 6
    assert summarize(both)["oracle_match_reference"]["quality"] == pytest.approx(m["quality"])


def test_per_model_mean_cost_uses_finite_train_cells_only():
    obs = pd.DataFrame({"model_id": ["a", "a", "a", "b", "b"],
                        "split": ["train", "train", "test", "train", "train"],
                        "cost": [1.0, 3.0, 100.0, np.nan, 4.0]})
    s = per_model_mean_cost(obs)
    assert s["a"] == 2.0 and s["b"] == 4.0


# --------------------------------------------------------------------------- #
# audit 2026-10-10: hand-checked selections, invariances, failure modes        #
# --------------------------------------------------------------------------- #
from evaluation.nirt.routing import align                     # noqa: E402
from router.nirt.frames import pivot_qm                       # noqa: E402

# expensive model listed FIRST so a column-order tie-break overpays
TIE_MODELS = ["a_big", "b_small", "ref"]
TIE_Y = np.array([[1.0, 1.0, 1.0],      # all tie -> cheapest is b_small
                  [1.0, 1.0, 0.0],      # a/b tie above ref -> b_small
                  [0.0, 0.0, 1.0]])     # only ref is right
TIE_C = np.array([[2.0, 0.1, 1.0],
                  [2.0, 0.1, 1.0],
                  [2.0, 0.1, 1.0]])


def test_hand_computed_router_picks_and_realized_metrics():
    # utilities at lam 10: q0 [.9-20, .5-1, .4-10] -> b; q1 [.2-20, .3-1, .9-10] -> b
    # q2 [.1-20, .05-1, .95-10] -> b.  lam 0: argmax pred -> a, ref, ref
    pq = np.array([[0.9, 0.5, 0.4], [0.2, 0.3, 0.9], [0.1, 0.05, 0.95]])
    pc = np.array([[2.0, 0.1, 1.0]] * 3)
    p = source_picks(TIE_Y, TIE_C, TIE_MODELS, "ref", {"r": (pq, pc)}, lams=(0.0, 10.0))
    y, c, is_ref = p["r@0"]
    assert y.tolist() == [1.0, 0.0, 1.0] and c.tolist() == [2.0, 1.0, 1.0]
    assert is_ref.tolist() == [False, True, True]
    y, c, is_ref = p["r@10"]
    assert y.tolist() == [1.0, 1.0, 0.0] and c.tolist() == [0.1, 0.1, 0.1] and not is_ref.any()
    s = summarize(p)
    assert s["r@0"]["quality"] == pytest.approx(2 / 3)
    assert s["r@0"]["cost_per_1k"] == pytest.approx(1e3 * 4 / 3)
    assert s["r@10"]["q_vs_reference"] == pytest.approx(2 / 3 - 2 / 3)
    assert s["r@10"]["cost_vs_reference"] == pytest.approx(0.1 / 1.0 - 1)
    assert s["r@0"]["reference_share"] == pytest.approx(2 / 3)


def test_oracle_definitions_on_tied_pool():
    p = source_picks(TIE_Y, TIE_C, TIE_MODELS, "ref", {}, lams=())
    y, c, _ = p["oracle_best_quality"]
    assert y.tolist() == [1.0, 1.0, 1.0] and c.tolist() == [0.1, 0.1, 1.0]
    y, c, _ = p["oracle_match_reference"]           # cheapest with Y >= Y[ref]
    assert y.tolist() == [1.0, 1.0, 1.0] and c.tolist() == [0.1, 0.1, 1.0]


def test_oracle_at_lam0_breaks_quality_ties_to_cheapest():
    """oracle@0 is the best-quality oracle; a column-order tie-break would pay
    for a_big (listed first) on every tied query."""
    p = source_picks(TIE_Y, TIE_C, TIE_MODELS, "ref", {}, lams=(0.0,))
    assert p["oracle@0"][1].tolist() == p["oracle_best_quality"][1].tolist() == [0.1, 0.1, 1.0]


def test_lam0_ignores_cost_predictions():
    pq = np.array([[0.9, 0.5, 0.4], [0.2, 0.3, 0.9], [0.1, 0.05, 0.95]])
    rng = np.random.default_rng(0)
    a = source_picks(TIE_Y, TIE_C, TIE_MODELS, "ref",
                     {"r": (pq, rng.uniform(0, 1e6, (3, 3)))}, lams=(0.0,))["r@0"]
    b = source_picks(TIE_Y, TIE_C, TIE_MODELS, "ref",
                     {"r": (pq, np.zeros(3))}, lams=(0.0,))["r@0"]
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    np.testing.assert_array_equal(a[1], TIE_C[np.arange(3), pq.argmax(1)])


def test_reported_metrics_are_realized_not_predicted():
    # predictions far from the truth on both axes: reported numbers must be Y / C at the pick
    pq = np.array([[0.0, 0.99, 0.0]] * 3)           # always picks b_small
    pc = np.array([[50.0, 7.0, 50.0]] * 3)          # predicted cost of b_small: 7
    y, c, _ = source_picks(TIE_Y, TIE_C, TIE_MODELS, "ref", {"r": (pq, pc)}, lams=(0.0, 0.01))["r@0.01"]
    assert y.tolist() == TIE_Y[:, 1].tolist()       # [1, 1, 0], not 0.99
    assert c.tolist() == [0.1, 0.1, 0.1]            # realized, not 7


def _shuffled(perm):
    pq = np.array([[0.9, 0.5, 0.4], [0.2, 0.3, 0.9], [0.1, 0.05, 0.95]])
    pc = np.array([[2.0, 0.1, 1.0]] * 3) * np.array([[1.0], [1.5], [0.7]])
    cm = np.array([2.0, 0.1, 1.0])
    models = [TIE_MODELS[j] for j in perm]
    return summarize(source_picks(TIE_Y[:, perm], TIE_C[:, perm], models, "ref",
                                  {"cell": (pq[:, perm], pc[:, perm]), "static": (pq[:, perm], cm[perm])},
                                  lams=(0.0, 0.1, 10.0, 1000.0)))


@pytest.mark.parametrize("perm", [[1, 0, 2], [2, 1, 0], [2, 0, 1]])
def test_model_column_order_does_not_change_results(perm):
    assert _shuffled(perm) == _shuffled([0, 1, 2])


def test_source_picks_is_reproducible():
    rng = np.random.default_rng(7)
    Y, C = rng.random((20, 4)), rng.random((20, 4))
    pq, pc = rng.random((20, 4)), rng.random((20, 4))
    args = (Y, C, list("wxyz"), "z", {"r": (pq, pc)}, (0.0, 1.0, 5.0))
    a, b = source_picks(*args), source_picks(*args)
    assert a.keys() == b.keys()
    for k in a:
        for u, v in zip(a[k], b[k]):
            np.testing.assert_array_equal(u, v)
    assert summarize(a) == summarize(b)


def test_combine_weights_by_query_count():
    one = source_picks(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]), ["m", "r"], "r", {}, ())
    three = source_picks(np.zeros((3, 2)), np.full((3, 2), 3.0), ["m", "r"], "r", {}, ())
    s = summarize(combine({"a": one, "b": three}))
    assert s["reference"]["quality"] == pytest.approx(0.25)        # 1/4, not mean-of-means 0.5
    assert s["reference"]["cost_per_1k"] == pytest.approx(1e3 * (1 + 9) / 4)


def test_align_maps_shuffled_long_predictions_onto_truth_order():
    long = pd.DataFrame({"query_id": ["q2", "q1", "q2", "q1"], "model_id": ["b", "a", "a", "b"],
                         "quality_pred": [0.4, 0.1, 0.3, 0.2]})
    truth = pd.DataFrame([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                         index=["q1", "q2", "q3"], columns=["b", "a"])
    got = align(pivot_qm(long, "quality_pred"), truth)
    np.testing.assert_array_equal(got[:2], [[0.2, 0.1], [0.4, 0.3]])
    assert np.isnan(got[2]).all()                   # q3 has no prediction -> dropped by caller


def test_source_picks_rejects_non_finite_truth():
    Y = TIE_Y.copy()
    Y[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        source_picks(Y, TIE_C, TIE_MODELS, "ref", {}, (0.0,))


def test_source_picks_rejects_empty_pool():
    with pytest.raises(ValueError, match="no queries"):
        source_picks(np.zeros((0, 3)), np.zeros((0, 3)), TIE_MODELS, "ref", {}, (0.0,))


def test_missing_per_model_cost_raises_instead_of_dropping_the_model():
    """A NaN C(m) (model with no train cost) used to silently remove that model
    from the candidate set for every lam > 0 while keeping it at lam = 0."""
    pq = np.array([[0.1, 0.9, 0.2]] * 3)
    with pytest.raises(ValueError, match="non-finite"):
        source_picks(TIE_Y, TIE_C, TIE_MODELS, "ref",
                     {"r": (pq, np.array([2.0, np.nan, 1.0]))}, (0.0, 1.0))
