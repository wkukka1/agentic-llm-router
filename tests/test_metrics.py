import numpy as np
import pytest

from router.metrics import (
    apply_temperature,
    evaluate,
    expected_calibration_error,
    fit_temperature,
    selective_accuracy,
    top_k_accuracy,
)

LABELS = ["a", "b", "c"]


def test_evaluate_on_perfect_predictions():
    proba = np.eye(3)
    m = evaluate(["a", "b", "c"], proba, LABELS)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["top2_accuracy"] == 1.0


def test_evaluate_confusion_matrix_orientation():
    # Two true "a" rows, both predicted "b".
    proba = np.array([[0.1, 0.9, 0.0], [0.2, 0.8, 0.0]])
    m = evaluate(["a", "a"], proba, LABELS)
    assert m["confusion_matrix"][0][1] == 2
    assert m["accuracy"] == 0.0


def test_top_k_accuracy_counts_runner_up():
    proba = np.array([[0.4, 0.5, 0.1]])
    y = np.array([0])
    assert top_k_accuracy(proba, y, 1) == 0.0
    assert top_k_accuracy(proba, y, 2) == 1.0


def test_selective_accuracy_prefers_confident_rows():
    confidence = np.array([0.9, 0.8, 0.4, 0.3])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert selective_accuracy(confidence, correct, 0.5) == 1.0
    assert selective_accuracy(confidence, correct, 1.0) == 0.5


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    confidence = np.full(100, 0.7)
    correct = np.concatenate([np.ones(70), np.zeros(30)])
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_flags_overconfidence():
    confidence = np.full(100, 0.99)
    correct = np.concatenate([np.ones(50), np.zeros(50)])
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.49, abs=0.01)


def test_apply_temperature_preserves_ranking():
    """Temperature scaling must never change which class wins."""
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(5), size=50)
    for temperature in (0.2, 1.0, 5.0):
        assert (apply_temperature(proba, temperature).argmax(1) == proba.argmax(1)).all()


def test_apply_temperature_rows_sum_to_one():
    proba = np.array([[0.7, 0.2, 0.1]])
    assert apply_temperature(proba, 3.0).sum() == pytest.approx(1.0)


def test_fit_temperature_softens_an_overconfident_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 400)
    proba = np.full((400, 3), 0.005)
    proba[np.arange(400), y] = 0.99
    # Corrupt 40% of the labels: the model stays confident but is often wrong.
    wrong = rng.random(400) < 0.4
    y[wrong] = (y[wrong] + 1) % 3
    proba = proba / proba.sum(1, keepdims=True)

    temperature = fit_temperature(proba, y)
    assert temperature > 1.0, "an overconfident model must be softened"
    assert apply_temperature(proba, temperature).max(1).mean() < proba.max(1).mean()


def test_fit_temperature_on_empty_input_is_identity():
    assert fit_temperature(np.zeros((0, 3)), np.array([], dtype=int)) == 1.0
