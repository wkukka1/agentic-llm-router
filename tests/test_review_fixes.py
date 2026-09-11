"""Regression tests for the defects found in review of PR #1.

Each test names the failure it prevents. They are grouped here rather than
scattered into the per-module files because what they have in common is how
they failed: every one produced a *plausible* wrong answer rather than an
error, which is the class of bug that survives a green test suite.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from router.analysis import confusion_table, top_up_metrics
from router.metrics import expected_calibration_error, top_k_accuracy


class TestConfusionMatrixKeepsPredictedClasses:
    """A class predicted but never true used to lose its column, and the rows
    renormalised afterwards so the table still summed to 1 with the evidence
    gone."""

    @staticmethod
    def _preds(true, pred):
        return pd.DataFrame({"y_true": true, "y_pred": pred})

    def test_a_class_only_ever_predicted_still_gets_a_column(self):
        # `extract` never occurs but the model predicts it twice -- exactly the
        # run you would want to see for a class with three rows in the eval.
        table = confusion_table(self._preds(
            ["answer"] * 4, ["answer", "answer", "extract", "extract"]))
        assert "extract" in table.columns
        assert table.loc["answer", "extract"] == pytest.approx(0.5)

    def test_rows_do_not_renormalise_over_the_dropped_mass(self):
        """The tell that the old bug was invisible: the row still summed to 1."""
        table = confusion_table(self._preds(
            ["answer"] * 4, ["answer", "answer", "extract", "extract"]))
        assert table.loc["answer", "answer"] == pytest.approx(0.5)

    def test_an_explicit_label_list_spans_classes_absent_from_both_axes(self):
        table = confusion_table(
            self._preds(["a", "a"], ["a", "b"]), labels=["a", "b", "c"])
        assert list(table.columns) == list(table.index) == ["a", "b", "c"]
        assert table.loc["c"].sum() == 0.0

    def test_square_and_ordered(self):
        table = confusion_table(self._preds(["b", "a"], ["a", "c"]))
        assert list(table.index) == list(table.columns) == ["a", "b", "c"]


class TestTopUpMetricsDoesNotOverwrite:
    """`update()` used to recompute accuracy by argmax and overwrite the stored
    value, and refit calibration on the test split under the same key."""

    @staticmethod
    def _predictions():
        return pd.DataFrame({
            "y_true": ["a", "b", "a", "b"],
            "y_pred": ["a", "b", "b", "b"],
            "p_a": [0.9, 0.2, 0.4, 0.1],
            "p_b": [0.1, 0.8, 0.6, 0.9],
            "uid": list("wxyz"),
            "confidence": [0.9, 0.8, 0.6, 0.9],
        })

    def test_a_stored_accuracy_survives(self):
        """The stored value came from the model's own decision rule. A head with
        a threshold or a prior correction does not predict by argmax, and the
        recomputed number would silently disagree with the confusion matrix in
        the same report."""
        test = {"labels": ["a", "b"], "accuracy": 0.123}
        top_up_metrics(test, self._predictions())
        assert test["accuracy"] == 0.123

    def test_calibration_is_never_refit_on_test(self):
        test = {"labels": ["a", "b"], "temperature": 0.5}
        added = top_up_metrics(test, self._predictions())
        assert test["temperature"] == 0.5
        assert "temperature" not in added
        assert "ece_calibrated" not in added

    def test_genuinely_missing_metrics_are_filled_in(self):
        test = {"labels": ["a", "b"]}
        added = top_up_metrics(test, self._predictions())
        assert "ece" in added and "log_loss" in added

    def test_a_run_with_no_labels_is_left_alone_rather_than_raising(self):
        test = {}
        assert top_up_metrics(test, self._predictions()) == {}
        assert test == {}


class TestLoadRunGuard:
    """`.get("test", {})` implied test could be absent, then indexed into it."""

    def test_a_metrics_file_without_a_test_block_loads(self, tmp_path):
        from router.analysis import load_run

        run = tmp_path / "run"
        run.mkdir()
        pd.DataFrame({"y_true": ["a"], "y_pred": ["a"], "uid": ["1"],
                      "confidence": [0.9]}).to_parquet(run / "test_predictions.parquet")
        (run / "metrics.json").write_text(json.dumps({"runtime": {}}), encoding="utf-8")
        _, metrics = load_run("run", out_dir=tmp_path)
        assert metrics["test"] == {}


class TestMetricEdgeCases:
    def test_ece_bins_include_a_confidence_of_exactly_zero(self):
        """A sample belonging to no bin makes a calibration number quietly wrong
        rather than loudly broken."""
        conf = np.array([0.0, 0.0])
        assert expected_calibration_error(conf, np.array([0.0, 0.0])) == pytest.approx(0.0)
        assert expected_calibration_error(conf, np.array([1.0, 1.0])) == pytest.approx(1.0)

    def test_ece_is_zero_for_a_perfectly_calibrated_split(self):
        conf = np.array([0.9] * 10)
        correct = np.array([1.0] * 9 + [0.0])
        assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-9)

    def test_top_k_is_one_when_k_covers_every_class(self):
        """Degenerate but not undefined. NaN propagated into leaderboards as a
        blank cell that reads like a failed run."""
        proba = np.array([[0.6, 0.4], [0.3, 0.7]])
        idx = np.array([0, 1])
        assert top_k_accuracy(proba, idx, 2) == 1.0
        assert top_k_accuracy(proba, idx, 3) == 1.0

    def test_top_k_still_discriminates_below_the_class_count(self):
        proba = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
        assert top_k_accuracy(proba, np.array([0, 2]), 1) == 1.0
        assert top_k_accuracy(proba, np.array([2, 0]), 1) == 0.0
        assert top_k_accuracy(proba, np.array([2, 0]), 2) == 0.0


class TestAuditGatingIsHonest:
    def test_clean_names_the_checks_it_actually_covers(self):
        """The docstring said five checks; `clean` gates on two."""
        from router.overfit import AuditResult

        assert AuditResult.gating_checks == ("permutation", "near-duplicates")

    def test_permutation_p_is_bounded_by_the_number_of_permutations(self):
        """Five permutations could never report better than p=0.17 however
        separated the real score was, which is why the default is 100."""
        from router.overfit import AuditResult

        r = AuditResult(name="x", n=10, n_classes=2, majority_rate=0.5,
                        train_acc=1.0, test_acc=0.9, fold_sd=0.0,
                        shuffled_mean=0.5, shuffled_sd=0.05,
                        shuffled_scores=[0.5] * 5)
        assert r.permutation_p == pytest.approx(1 / 6)

    def test_permutation_p_counts_ties_against_the_real_score(self):
        from router.overfit import AuditResult

        r = AuditResult(name="x", n=10, n_classes=2, majority_rate=0.5,
                        train_acc=1.0, test_acc=0.5, fold_sd=0.0,
                        shuffled_mean=0.5, shuffled_sd=0.0,
                        shuffled_scores=[0.5] * 9)
        assert r.permutation_p == pytest.approx(1.0)

    def test_default_permutation_count_is_enough_to_estimate_a_null(self):
        import inspect

        from router.overfit import audit

        assert inspect.signature(audit).parameters["permutations"].default >= 100


class TestEncoderParamMismatchIsAnError:
    """A head fitted on one encoder's vectors, served another's, is silently
    wrong -- the failure this raises on."""

    def test_mismatched_encoder_settings_raise_on_load(self, tmp_path):
        import pickle

        from router.models import build

        path = tmp_path / "model"
        path.mkdir()
        with (path / "head.pkl").open("wb") as fh:
            pickle.dump({"head": object(), "labels": ["a"],
                         "params": {"encoder_model": "A", "pooling": "mean",
                                    "max_length": 256}}, fh)
        model = build("embed_logreg", encoder_model="B", pooling="mean", max_length=256)
        with pytest.raises(ValueError, match="different encoder settings"):
            model.load(path)

    def test_matching_settings_load_cleanly(self, tmp_path):
        import pickle

        from router.models import build

        path = tmp_path / "model"
        path.mkdir()
        params = {"encoder_model": "A", "pooling": "mean", "max_length": 256}
        with (path / "head.pkl").open("wb") as fh:
            pickle.dump({"head": object(), "labels": ["a"], "params": params}, fh)
        model = build("embed_logreg", **params)
        model.load(path)
        assert model.labels == ["a"]


class TestEmbeddingCacheIsAtomic:
    def test_no_temp_file_survives_a_successful_write(self, tmp_path, monkeypatch):
        """np.save is not atomic; two runs encoding the same rows can interleave
        and leave a truncated array that loads without error."""
        from router.embeddings import EmbeddingEncoder

        enc = EmbeddingEncoder.__new__(EmbeddingEncoder)
        monkeypatch.setattr(EmbeddingEncoder, "signature", "sig", raising=False)
        monkeypatch.setattr(EmbeddingEncoder, "encode",
                            lambda self, texts: np.ones((len(texts), 3)))
        out = enc.encode_cached(["a", "b"], tag="t/u", cache_dir=tmp_path)
        assert out.shape == (2, 3)
        assert not list(tmp_path.glob("*.tmp.npy"))
        assert len(list(tmp_path.glob("*.npy"))) == 1

    def test_a_second_call_hits_the_cache(self, tmp_path, monkeypatch):
        from router.embeddings import EmbeddingEncoder

        calls = []
        enc = EmbeddingEncoder.__new__(EmbeddingEncoder)
        monkeypatch.setattr(EmbeddingEncoder, "signature", "sig", raising=False)

        def fake(self, texts):
            calls.append(texts)
            return np.ones((len(texts), 3))

        monkeypatch.setattr(EmbeddingEncoder, "encode", fake)
        enc.encode_cached(["a"], tag="t", cache_dir=tmp_path)
        enc.encode_cached(["a"], tag="t", cache_dir=tmp_path)
        assert len(calls) == 1


class TestSharedSettings:
    def test_seed_is_shared_rather_than_defaulted_per_call_site(self):
        from router.config import ExperimentConfig, ModelConfig
        from router.settings import settings

        cfg = ExperimentConfig(name="x", model=ModelConfig(name="tfidf_logreg"))
        assert cfg.seed == settings().seed

    def test_environment_overrides_the_default(self, monkeypatch):
        from router.settings import settings

        monkeypatch.setenv("ROUTER_SEED", "4242")
        settings.cache_clear()
        try:
            assert settings().seed == 4242
        finally:
            settings.cache_clear()

    def test_dotenv_is_read_when_the_environment_is_silent(self, tmp_path, monkeypatch):
        from router.settings import settings

        env = tmp_path / ".env"
        env.write_text("ROUTER_SEED=77\n# a comment\nROUTER_DATA_DIR='/tmp/d'\n")
        monkeypatch.delenv("ROUTER_SEED", raising=False)
        settings.cache_clear()
        try:
            s = settings(env)
            assert s.seed == 77
            assert str(s.data_dir) == "/tmp/d"
        finally:
            settings.cache_clear()

    def test_environment_beats_dotenv(self, tmp_path, monkeypatch):
        from router.settings import settings

        env = tmp_path / ".env"
        env.write_text("ROUTER_SEED=77\n")
        monkeypatch.setenv("ROUTER_SEED", "88")
        settings.cache_clear()
        try:
            assert settings(env).seed == 88
        finally:
            settings.cache_clear()
