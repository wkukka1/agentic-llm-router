"""Regression tests for the defects found in review of PR #1.

Each test names the failure it prevents. They are grouped here rather than
scattered into the per-module files because what they have in common is how
they failed: every one produced a *plausible* wrong answer rather than an
error, which is the class of bug that survives a green test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestEncoderParamMismatchIsAnError:
    """A head fitted on one encoder's vectors, served another's, is silently
    wrong -- the failure this raises on."""

    def test_mismatched_encoder_settings_raise_on_load(self, tmp_path):
        import pickle

        from decompose.classifiers.prompt_decomposition.models import build

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

        from decompose.classifiers.prompt_decomposition.models import build

        path = tmp_path / "model"
        path.mkdir()
        params = {"encoder_model": "A", "pooling": "mean", "max_length": 256}
        with (path / "head.pkl").open("wb") as fh:
            pickle.dump({"head": object(), "labels": ["a"], "params": params}, fh)
        model = build("embed_logreg", **params)
        model.load(path)
        assert model.labels == ["a"]


class TestEmbeddingCacheIsAtomic:
    """Needs `router.embeddings.encoder`, which imports torch. CI installs the light
    core deliberately -- the encoders are a ~1GB extra -- so these skip there
    and run wherever `.[encoders]` is installed."""

    def test_no_temp_file_survives_a_successful_write(self, tmp_path, monkeypatch):
        """np.save is not atomic; two runs encoding the same rows can interleave
        and leave a truncated array that loads without error."""
        pytest.importorskip("torch")
        from decompose.classifiers.prompt_decomposition.encoder import EmbeddingEncoder

        enc = EmbeddingEncoder.__new__(EmbeddingEncoder)
        monkeypatch.setattr(EmbeddingEncoder, "signature", "sig", raising=False)
        monkeypatch.setattr(EmbeddingEncoder, "encode",
                            lambda self, texts: np.ones((len(texts), 3)))
        out = enc.encode_cached(["a", "b"], tag="t/u", cache_dir=tmp_path)
        assert out.shape == (2, 3)
        assert not list(tmp_path.glob("*.tmp.npy"))
        assert len(list(tmp_path.glob("*.npy"))) == 1

    def test_a_second_call_hits_the_cache(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        from decompose.classifiers.prompt_decomposition.encoder import EmbeddingEncoder

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
    def test_environment_overrides_the_default(self, monkeypatch):
        from decompose.classifiers.prompt_decomposition.settings import settings

        monkeypatch.setenv("ROUTER_SEED", "4242")
        settings.cache_clear()
        try:
            assert settings().seed == 4242
        finally:
            settings.cache_clear()

    def test_dotenv_is_read_when_the_environment_is_silent(self, tmp_path, monkeypatch):
        from decompose.classifiers.prompt_decomposition.settings import settings

        env = tmp_path / ".env"
        env.write_text("ROUTER_SEED=77\n# a comment\nROUTER_DATA_DIR='some/dir'\n")
        monkeypatch.delenv("ROUTER_SEED", raising=False)
        monkeypatch.delenv("ROUTER_DATA_DIR", raising=False)
        settings.cache_clear()
        try:
            s = settings(env)
            assert s.seed == 77
            # Compared as a Path, not a string: `Path("a/b")` stringifies with
            # backslashes on Windows, so a string comparison here passes on the
            # Linux runner and fails on the Windows one -- which is the entire
            # reason Windows is in the CI matrix.
            assert s.data_dir == Path("some/dir")
            assert s.handlabelled_dir == Path("some/dir") / "handlabelled"
        finally:
            settings.cache_clear()

    def test_environment_beats_dotenv(self, tmp_path, monkeypatch):
        from decompose.classifiers.prompt_decomposition.settings import settings

        env = tmp_path / ".env"
        env.write_text("ROUTER_SEED=77\n")
        monkeypatch.setenv("ROUTER_SEED", "88")
        settings.cache_clear()
        try:
            assert settings(env).seed == 88
        finally:
            settings.cache_clear()


class TestCalibrationProvenanceIsRecorded:
    """The temperature was always fitted on validation, but it was *stored*
    under the `test` key, which reads as though it were fitted there."""

    @staticmethod
    def _run(tmp_path, metrics):
        import yaml

        from decompose.classifiers.prompt_decomposition.models import build

        m = build("tfidf_logreg", char_ngrams=None, min_df=1)
        m.fit(["alpha one", "alpha two", "beta one", "beta two"], ["a", "a", "b", "b"])
        d = tmp_path / "run"
        d.mkdir()
        m.save(d / "model")
        (d / "config.yaml").write_text(yaml.safe_dump(
            {"model": {"name": "tfidf_logreg",
                       "params": {"char_ngrams": None, "min_df": 1}}}), encoding="utf-8")
        (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return d

    def test_the_calibration_block_wins_over_the_legacy_location(self, tmp_path):
        from decompose.classifiers.prompt_decomposition import DomainHead

        run = self._run(tmp_path, {
            "test": {"temperature": 9.0},
            "calibration": {"temperature": 2.0, "fitted_on": "validation"},
        })
        head = DomainHead(run)
        assert head.temperature == 2.0
        assert head.temperature_fitted_on == "validation"

    def test_an_older_run_still_loads_from_the_test_key(self, tmp_path):
        """Artifacts written before the calibration block existed were
        validation-fitted too; they just did not record it."""
        from decompose.classifiers.prompt_decomposition import DomainHead

        head = DomainHead(self._run(tmp_path, {"test": {"temperature": 3.0}}))
        assert head.temperature == 3.0
        assert head.temperature_fitted_on == "validation"
