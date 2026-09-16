"""``TrainingHarness`` (training/pipeline.py) -- runs all three wrapped
trainers (NIRT, baseline, MLP router) through one interface."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    FakeTrainingData,
    baseline_cfg,
    make_split_obs,
    make_synthetic_irt,
    nirt_cfg,
    nirt_datasets,
)

torch = pytest.importorskip("torch")

from router.llm.cost import CostModel
from router.models.registry import RouterModelFactory
from router.routing.routers import NIRTRouter
from training.nirt.baseline.trainer import BaselineNIRTTrainer
from training.nirt.trainer import NIRTTrainer
from training.pipeline import TrainingHarness, TrainingJob
from training.trainers.base import TrainingConfig
from training.trainers.mlp_router_trainer import MLPRouterTrainer


def test_harness_train_runs_nirt_and_loads_through_the_factory(tmp_path):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=20)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)

    harness = TrainingHarness()
    trainer = NIRTTrainer(runs_dir=tmp_path)
    config = TrainingConfig(run_name="harness-nirt", hyperparameters=nirt_cfg())
    run = harness.train(trainer, (train_ds, val_ds), config)

    assert run.run_id == "harness-nirt"
    assert run.trainer_name == "nirt"
    assert run.artifact is not None
    assert run.training_metrics.training_time > 0
    assert np.isfinite(run.validation_metrics.bce) and run.validation_metrics.bce != 0.0

    factory = RouterModelFactory()
    factory.register("nirt", NIRTRouter)
    router = factory.create(run.artifact, CostModel())
    assert isinstance(router, NIRTRouter)


def test_harness_run_accepts_a_training_job(tmp_path):
    train_obs, val_obs, q_store, m_store = make_synthetic_irt(n_queries=200, seed=21)
    train_ds, val_ds = nirt_datasets(train_obs, val_obs, q_store, m_store)

    trainer = NIRTTrainer(runs_dir=tmp_path)
    config = TrainingConfig(run_name="harness-job", hyperparameters=nirt_cfg())
    job = TrainingJob(run_id="harness-job", trainer=trainer, dataset=(train_ds, val_ds), config=config)
    run = TrainingHarness().run(job)
    assert run.run_id == "harness-job"


def test_harness_train_runs_baseline():
    from training.nirt.baseline.synthetic import make_synthetic, to_arrays

    syn = make_synthetic(n_queries=300, n_models=6, K=2, seed=22)
    tr, va = to_arrays(syn, seed=22)

    harness = TrainingHarness()
    trainer = BaselineNIRTTrainer()
    config = TrainingConfig(run_name="harness-baseline", hyperparameters=baseline_cfg(k=2, epochs=10))
    run = harness.train(trainer, (tr, va), config)

    assert run.trainer_name == "baseline"
    assert run.artifact.router_model_name == "baseline"
    assert run.training_metrics.training_time > 0


def test_harness_train_runs_mlp_router(tmp_path):
    d = FakeTrainingData(make_split_obs(seed=23), query_dim=16)

    harness = TrainingHarness()
    trainer = MLPRouterTrainer(runs_dir=tmp_path)
    config = TrainingConfig(
        run_name="harness-mlp", hyperparameters={"hidden": 16, "epochs": 3, "pathway": "irt"},
    )
    run = harness.train(trainer, d, config)

    assert run.trainer_name == "mlp"
    assert run.artifact.router_model_name == "mlp"
    # MLPRouterTrainer.last_result has no val_metrics/history -> zero defaults,
    # not a crash
    assert run.validation_metrics.bce == 0.0
    assert run.training_metrics.training_time > 0
