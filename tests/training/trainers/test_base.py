"""``RouterModelTrainer``/``TrainingConfig`` (training/trainers/base.py)."""

from __future__ import annotations

import pytest

from training.trainers.base import RouterModelTrainer, TrainingConfig, translate_training_config


def test_training_config_defaults():
    config = TrainingConfig(run_name="r")
    assert config.seed == 42
    assert config.max_epochs is None
    assert config.hyperparameters == {}
    assert config.preprocessing == {}


def test_router_model_trainer_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        RouterModelTrainer()


def test_a_concrete_trainer_starts_with_no_last_result():
    class _Trainer(RouterModelTrainer):
        name = "toy"

        def train(self, dataset, config):
            raise NotImplementedError

    assert _Trainer().last_result is None


# --------------------------------------------------------------------------- #
# translate_training_config (TN-01/TN-02: shared by NIRTTrainer and
# BaselineNIRTTrainer, previously two byte-identical private copies)          #
# --------------------------------------------------------------------------- #
def test_translate_training_config_applies_typed_overrides():
    config = TrainingConfig(
        run_name="r", seed=7, max_epochs=3, early_stopping_patience=5,
        preprocessing={"scale": "standard"},
        hyperparameters={"model": {"hidden": 8}},
    )
    cfg = translate_training_config(config)
    assert cfg["seed"] == 7
    assert cfg["train"] == {"epochs": 3, "patience": 5}
    assert cfg["data"] == {"scale": "standard"}
    assert cfg["model"] == {"hidden": 8}


def test_translate_training_config_handles_an_explicit_null_train_section():
    """TN-01: a real YAML shape ('train:' with no children) parses to
    hyperparameters["train"] = None; dict.setdefault("train", {}) does not
    replace an already-present None, so cfg["train"]["epochs"] = ... used to
    raise TypeError. Both wrapped fit() functions already defend against this
    same shape with `... or {}`."""
    config = TrainingConfig(
        run_name="r", max_epochs=2, hyperparameters={"train": None},
    )
    cfg = translate_training_config(config)
    assert cfg["train"] == {"epochs": 2}


def test_translate_training_config_handles_an_explicit_null_data_section():
    config = TrainingConfig(
        run_name="r", preprocessing={"scale": "standard"}, hyperparameters={"data": None},
    )
    cfg = translate_training_config(config)
    assert cfg["data"] == {"scale": "standard"}
