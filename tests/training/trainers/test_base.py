"""``RouterModelTrainer``/``TrainingConfig`` (training/trainers/base.py)."""

from __future__ import annotations

import pytest

from training.trainers.base import RouterModelTrainer, TrainingConfig


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
