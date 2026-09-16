"""The attribute heads: eleven binary signals over one encoder pass.

The risks here differ from the classifiers'. These heads are trained with
`class_weight="balanced"` on attributes as rare as 7.5%, which makes their raw
probabilities systematically wrong -- so calibration is not a nicety, it is what
makes a threshold mean anything. And a head can post a respectable AUC on an
attribute that is true 82% of the time while being useless at any threshold,
which is why prevalence travels with every score.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from decompose.classifiers.prompt_decomposition.heads.attributes import AttributeHead, AttributePrediction
from decompose.classifiers.prompt_decomposition.heads.attributes_model import AttributeModel
from decompose.classifiers.prompt_decomposition.heads.attributes_taxonomy import (
    ATTRIBUTE_NAMES,
    CRITERIA,
    GATES,
    describe,
)
from training.prompt_decomposition.heads.attributes import evaluate_head


@pytest.fixture
def separable():
    """Two attributes a linear head can learn, one rare and one common."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (600, 6))
    return X, {
        "code": (X[:, 0] > 1.4).astype(float),          # ~8% positive
        "complexity": (X[:, 1] > 0.0).astype(float),    # ~50% positive
    }


class TestTaxonomy:
    def test_gates_and_criteria_partition_the_attributes(self):
        assert set(GATES) | set(CRITERIA) == set(ATTRIBUTE_NAMES)
        assert not set(GATES) & set(CRITERIA)

    def test_language_is_deliberately_not_modelled(self):
        """It probes at 0.990 and that is the argument against training it: the
        label comes from a detector, so imitating it is strictly worse than
        calling one."""
        from decompose.classifiers.prompt_decomposition.heads.attributes_taxonomy import NOT_MODELLED

        assert "non_english" in NOT_MODELLED
        assert "non_english" not in ATTRIBUTE_NAMES

    def test_an_unknown_attribute_is_rejected(self):
        with pytest.raises(KeyError, match="unknown attribute"):
            describe("difficulty")

    def test_difficulty_is_not_among_them(self):
        """Three experiments failed to predict it from prompt text; it is a
        product of other signals, not a head."""
        assert "difficulty" not in ATTRIBUTE_NAMES


class TestModel:
    def test_it_fits_one_head_per_labelled_attribute(self, separable):
        X, y = separable
        model = AttributeModel(names=("code", "complexity")).fit(X, y)
        assert set(model.heads) == {"code", "complexity"}
        assert set(model.base_rates) == {"code", "complexity"}

    def test_an_attribute_with_one_class_is_skipped_not_fitted(self, separable):
        """A head fitted on a constant label predicts that constant forever and
        reports a perfect-looking score."""
        X, _ = separable
        model = AttributeModel(names=("code",)).fit(X, {"code": np.zeros(len(X))})
        assert "code" not in model.heads

    def test_unlabelled_rows_are_skipped_rather_than_treated_as_negative(self, separable):
        """NaN means the judge did not label the row. Counting it as False
        would teach the head that unlabelled means absent."""
        X, y = separable
        partial = y["code"].copy()
        partial[:300] = np.nan
        model = AttributeModel(names=("code",)).fit(X, {"code": partial})
        assert model.base_rates["code"] == pytest.approx(np.nanmean(partial), abs=1e-9)

    def test_probabilities_are_per_attribute_and_in_range(self, separable):
        X, y = separable
        out = AttributeModel(names=("code", "complexity")).fit(X, y).predict_proba(X)
        assert set(out) == {"code", "complexity"}
        for values in out.values():
            assert values.shape == (len(X),)
            assert ((values >= 0) & (values <= 1)).all()

    def test_it_learns_the_signal_it_was_given(self, separable):
        from sklearn.metrics import roc_auc_score

        X, y = separable
        out = AttributeModel(names=("code",)).fit(X[:400], {"code": y["code"][:400]}).predict_proba(X[400:])
        assert roc_auc_score(y["code"][400:], out["code"]) > 0.9

    def test_calibration_changes_the_probabilities_it_was_fitted_on(self, separable):
        """`class_weight="balanced"` leaves a rare attribute's probabilities
        inflated; if calibration is a no-op the temperature never fitted."""
        X, y = separable
        model = AttributeModel(names=("code",)).fit(X[:400], {"code": y["code"][:400]})
        before = model.predict_proba(X[400:])["code"].mean()
        model.calibrate(X[400:], {"code": y["code"][400:]})
        assert "code" in model.calibrators
        assert model.predict_proba(X[400:])["code"].mean() != pytest.approx(before)

    def test_the_solver_converges_on_mixed_scale_features(self, separable):
        """The matrix mixes an L2-normalised embedding with raw surface counts.
        Unscaled, the solver hits its iteration cap and the head is quietly
        under-fitted -- a warning, not an error, so it ships."""
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        rng = np.random.default_rng(1)
        X = np.hstack([rng.normal(0, 0.05, (400, 4)), rng.normal(0, 500, (400, 2))])
        y = {"code": (X[:, 5] > 0).astype(float)}
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            AttributeModel(names=("code",)).fit(X, y)

    def test_save_and_load_round_trip(self, separable, tmp_path):
        X, y = separable
        model = AttributeModel(names=("code", "complexity")).fit(X, y).calibrate(X, y)
        model.save(tmp_path)
        loaded = AttributeModel.load(tmp_path)
        for name, values in model.predict_proba(X).items():
            np.testing.assert_allclose(loaded.predict_proba(X)[name], values)
        assert sorted(loaded.calibrators) == sorted(model.calibrators)


class TestCalibrationCorrectsAnOffset:
    """The specific failure temperature scaling cannot fix."""

    def test_platt_beats_temperature_on_a_shifted_head(self):
        """`class_weight="balanced"` shifts a rare attribute's log-odds by the
        base-rate offset. Temperature rescales the logit and cannot move it, so
        it leaves the error in place; Platt fits an intercept too."""
        from training.prompt_decomposition.metrics import apply_temperature, fit_temperature
        from training.prompt_decomposition.metrics import expected_calibration_error as ece

        rng = np.random.default_rng(7)
        truth = rng.normal(-2.5, 1.0, 4000)                  # ~8% positive
        y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-truth))).astype(int)
        shifted = 1 / (1 + np.exp(-(truth + 2.5)))           # the balanced-weight offset

        temperature = fit_temperature(np.column_stack([1 - shifted, shifted]), y)
        tempered = apply_temperature(np.column_stack([1 - shifted, shifted]), temperature)[:, 1]

        model = AttributeModel(names=("code",))
        model.heads["code"] = _ConstantHead(shifted)
        model.calibrate(np.zeros((4000, 1)), {"code": y.astype(float)})
        platt = model.predict_proba(np.zeros((4000, 1)))["code"]

        correct = y.astype(float)
        assert ece(platt, correct) < ece(tempered, correct)
        assert ece(platt, correct) < 0.05


class _ConstantHead:
    """Stands in for a fitted head so the calibrator can be tested alone."""

    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba, self._proba])


class TestStaleArtifacts:
    def test_an_artifact_from_an_older_version_is_refused_by_name(self, tmp_path):
        """It happened: a run saved before the calibration change unpacked into
        a bare `unexpected keyword argument 'temperatures'`, which is true and
        tells the reader nothing about what to do."""
        import pickle

        with (tmp_path / "attributes.pkl").open("wb") as fh:
            pickle.dump({"names": ("code",), "heads": {}, "temperatures": {"code": 1.0}}, fh)
        with pytest.raises(ValueError, match="incompatible version"):
            AttributeModel.load(tmp_path)


class TestScoring:
    def test_base_rate_travels_with_every_score(self):
        """AUC hides prevalence, and these attributes span 0.075 to 0.819."""
        y = np.array([1.0] * 90 + [0.0] * 10)
        out = evaluate_head(y, np.linspace(0, 1, 100))
        assert out["base_rate"] == pytest.approx(0.9)
        assert "average_precision" in out and "ece" in out

    def test_unlabelled_rows_are_excluded_from_the_score(self):
        y = np.array([1.0, 0.0, np.nan, 1.0])
        assert evaluate_head(y, np.array([0.9, 0.1, 0.5, 0.8]))["n"] == 3

    def test_it_reports_a_threshold_that_reaches_the_recall_target(self):
        y = np.array([0.0] * 50 + [1.0] * 50)
        out = evaluate_head(y, np.linspace(0, 1, 100), recall_target=0.5)
        assert "threshold" in out
        assert out["precision@recall50"] > 0.9


class TestPrediction:
    def _prediction(self):
        return AttributePrediction({"code": 0.9, "math": 0.2, "complexity": 0.7})

    def test_gates_and_criteria_are_separable_by_the_caller(self):
        p = self._prediction()
        assert set(p.gates) == {"code", "math"}
        assert set(p.criteria) == {"complexity"}

    def test_above_returns_the_strongest_first(self):
        assert self._prediction().above(0.5) == ["code", "complexity"]

    def test_the_vector_keeps_its_width_when_a_head_is_missing(self):
        """A missing head must not silently shorten the handoff vector."""
        v = self._prediction().vector(("code", "math", "creative_writing"))
        assert len(v) == 3 and np.isnan(v[2])


class TestServingHead:
    def _run_dir(self, tmp_path, separable):
        X, y = separable
        AttributeModel(names=("code", "complexity")).fit(X, y).calibrate(X, y).save(tmp_path)
        (tmp_path / "attributes.json").write_text(json.dumps(
            {"encoder_model": None, "feature_set": "surface"}), encoding="utf-8")
        return tmp_path

    def test_it_serves_a_probability_per_fitted_attribute(self, tmp_path, separable):
        X, y = separable
        # the surface extractor produces 23 features, so refit on that width
        from decompose.classifiers.prompt_decomposition.features import build_features

        prompts = ["write me a function foo(bar)", "what is the capital of peru"] * 60
        feats = build_features(prompts, "surface")
        labels = {"code": np.array([1.0, 0.0] * 60), "complexity": np.array([1.0, 0.0] * 60)}
        AttributeModel(names=("code", "complexity")).fit(feats, labels).save(tmp_path)
        (tmp_path / "attributes.json").write_text(json.dumps(
            {"encoder_model": None, "feature_set": "surface"}), encoding="utf-8")

        head = AttributeHead(tmp_path)
        assert set(head.names) == {"code", "complexity"}
        p = head.predict("write me a function foo(bar)")
        assert 0.0 <= p["code"] <= 1.0
        assert p["code"] > head.predict("what is the capital of peru")["code"]

    def test_an_empty_batch_is_not_an_error(self, tmp_path, separable):
        assert AttributeHead(self._run_dir(tmp_path, separable)).predict_batch([]) == []
