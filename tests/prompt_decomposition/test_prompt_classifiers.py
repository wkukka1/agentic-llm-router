"""The heads, wired into Will's `PromptDecomposer` as concrete `Classifier`s.

These tests are about the *contract*, not the accuracy -- the heads' own tests
cover that. What matters here is that a signal arrives with the name a consumer
expects, carries its provenance, and that the decomposer merges several
classifiers without collision.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from decompose.classifiers.base import ClassificationInput
from decompose.classifiers.prompt_heads import SurfaceClassifier
from decompose.decomposer import PromptDecomposer
from decompose.signals import Signal


def test_the_surface_classifier_needs_no_artifact():
    """It is the one head with no model behind it, so it always works -- which
    makes it the right smoke test for the wiring itself."""
    signals = SurfaceClassifier().classify(ClassificationInput(prompt="write me a poem"))
    assert len(signals) == 1
    assert signals[0].name == "surface"
    assert signals[0].value["log_words"] > 0


def test_a_signal_carries_which_classifier_and_version_produced_it():
    """Provenance is the point of `produced_by`: a signal that cannot be traced
    to a run is one nobody can debug."""
    signal = SurfaceClassifier().classify(ClassificationInput(prompt="hello there"))[0]
    assert signal.produced_by == "surface:1"


def test_the_decomposer_merges_classifiers_into_one_signal_bag():
    class Constant:
        name, version = "constant", "1"

        def classify(self, input):
            return [Signal(name="constant", value=1.0, produced_by="constant:1")]

    signals = PromptDecomposer([SurfaceClassifier(), Constant()]).extract_signals(
        ClassificationInput(prompt="summarise this document"))
    assert set(signals.signals) == {"surface", "constant"}
    assert signals.get("surface").value["log_chars"] > 0


def test_two_classifiers_claiming_one_name_is_an_error_not_a_silent_overwrite():
    """`PromptSignals.add` raises on a duplicate. Worth a test because the
    failure it prevents -- one head quietly shadowing another -- is invisible."""
    with pytest.raises(ValueError, match="already exists"):
        PromptDecomposer([SurfaceClassifier(), SurfaceClassifier()]).extract_signals(
            ClassificationInput(prompt="anything"))


class TestTrainedHeadAdapters:
    """The adapters that wrap a run directory, exercised on a stub artifact."""

    def _length_run(self, tmp_path):
        from decompose.classifiers.prompt_decomposition.features import build_features
        from decompose.classifiers.prompt_decomposition.heads.length_model import LengthModel

        prompts = ["write a detailed essay about databases", "yes or no"] * 60
        y = np.array([7.0, 4.0] * 60)
        model = LengthModel(alpha=1.0, feature_set="surface").fit(
            build_features(prompts, "surface"), y)
        model.save(tmp_path)
        (tmp_path / "length.json").write_text(json.dumps(
            {"feature_set": "surface", "encoder_model": None, "variant": "multi_turn"}))
        return tmp_path

    def test_the_length_adapter_reports_its_target_variant(self, tmp_path):
        """A single-turn head and a multi-turn head disagree by 2.8x on the same
        prompt, so which target an artifact was fitted on travels with every
        signal it emits."""
        from decompose.classifiers.prompt_heads import LengthClassifier

        signals = LengthClassifier(self._length_run(tmp_path)).classify(
            ClassificationInput(prompt="write a detailed essay about databases"))
        by_name = {s.name: s for s in signals}
        assert by_name["expected_tokens"].metadata["variant"] == "multi_turn"
        assert by_name["expected_tokens"].value > 0
        assert by_name["length_bucket"].value in ("short", "medium", "long", "very_long")

    def test_the_length_signal_does_not_claim_more_confidence_than_it_has(self, tmp_path):
        """One estimate is within 2x of the truth about 60% of the time. A
        confidence of 1.0 would invite a caller to budget against it."""
        from decompose.classifiers.prompt_heads import LengthClassifier

        signals = LengthClassifier(self._length_run(tmp_path)).classify(
            ClassificationInput(prompt="yes or no"))
        assert all(s.confidence < 1.0 for s in signals)
