"""``default_prompt_decomposer`` -- builds a real, multi-classifier
``PromptDecomposer`` from whichever prompt-heads artifacts are on hand.

Complements tests/prompt_decomposition/test_prompt_classifiers.py, which
tests one classifier at a time; this is the first test proving a
*factory-built*, *multi*-classifier decomposer merges correctly."""

from __future__ import annotations

import json

import numpy as np

from decompose.classifiers.base import ClassificationInput
from decompose.decomposer import PromptDecomposer, default_prompt_decomposer


def _length_run(tmp_path):
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


def test_default_factory_with_no_artifacts_is_just_the_surface_classifier():
    decomposer = default_prompt_decomposer()
    assert isinstance(decomposer, PromptDecomposer)
    signals = decomposer.extract_signals(ClassificationInput(prompt="write me a poem"))
    assert set(signals.signals) == {"surface"}


def test_default_factory_wires_a_supplied_artifact_alongside_surface(tmp_path):
    decomposer = default_prompt_decomposer({"length": _length_run(tmp_path)})
    signals = decomposer.extract_signals(
        ClassificationInput(prompt="write a detailed essay about databases"))
    assert set(signals.signals) == {"surface", "expected_tokens", "length_bucket"}
    assert signals.get("surface").value["log_words"] > 0
    assert signals.get("expected_tokens").value > 0
