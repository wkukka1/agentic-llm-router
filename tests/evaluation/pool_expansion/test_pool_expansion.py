"""Pool-expansion battery: derived headline numbers, ablation pool arithmetic,
ledger row shape / rendering, routing / provenance on toy data.  No checkpoint
loading -- prediction frames are injected."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evaluation.pool_expansion import (
    LEDGER_COLUMNS,
    ablation_pools,
    derived_headline,
    ledger_row,
    render_ledger_md,
)
from evaluation.pool_expansion import battery
from evaluation.pool_expansion.battery import (
    _frontier_saving,
    _pool_matrices,
    _val_selected_saving,
    ablation_block,
    prediction_matrices,
    provenance,
    routing_block,
)
from helpers import FakeTrainingData, make_split_obs
from router.config import Config, load_config
from router.nirt.frames import pivot_qm
from router.provenance import file_digest


# --------------------------------------------------------------------------- #
# toy pool                                                                    #
# --------------------------------------------------------------------------- #
MODEL_IDS = ["A", "B", "C"]
TRUE = np.array([
    [1.0, 1.0, 0.0],   # q0: A,B correct
    [1.0, 0.0, 0.0],   # q1: A only
    [0.0, 1.0, 1.0],   # q2: B,C
    [1.0, 1.0, 1.0],   # q3: all
])
COST = np.array([
    [1.0, 0.1, 0.5],
    [1.0, 0.1, 0.5],
    [1.0, 0.1, 0.5],
    [1.0, 0.1, 0.5],
])
TQ_ACC = {"A": 0.9, "B": 0.5, "C": 0.6}  # A is the best single model
FRONTIER = pd.DataFrame([
    {"lam": 0.0, "accuracy": 0.75, "cost_per_1k_queries": 900.0},
    {"lam": 0.5, "accuracy": 0.73, "cost_per_1k_queries": 400.0},
    {"lam": 1.0, "accuracy": 0.60, "cost_per_1k_queries": 150.0},
])
# Validation frontier whose cheapest in-band point is lam=1.0 -- a lambda the
# test frontier would never pick (test acc 0.60 there).
VAL_FRONTIER = pd.DataFrame([
    {"lam": 0.0, "accuracy": 0.80, "cost_per_1k_queries": 950.0},
    {"lam": 0.5, "accuracy": 0.70, "cost_per_1k_queries": 380.0},
    {"lam": 1.0, "accuracy": 0.795, "cost_per_1k_queries": 140.0},
])
VAL_BEST_ACC = 0.80


def _derived(**kw):
    return derived_headline(TRUE, COST, TRUE, model_ids=MODEL_IDS, tq_acc=TQ_ACC,
                            frontier=FRONTIER, reference="A", **kw)


def test_derived_oracle_ceiling_and_offbest():
    d = _derived()
    assert d["best_single_model"] == "A"
    assert d["best_single_test_accuracy"] == pytest.approx(0.75)
    # oracle = cheapest model at per-query max score = [B, A, B, B]; cost 0.325 vs A's 1.0
    assert d["oracle_cost_per_1k_queries"] == pytest.approx(325.0)
    assert d["oracle_cost_saving_ceiling"] == pytest.approx(0.675)
    assert d["oracle_offbest_fraction"] == pytest.approx(0.75)
    assert d["oracle_model_mix"] == {"A": 1, "B": 3}


def test_derived_naive_argmax_oracle_recorded_for_comparison():
    d = _derived()
    na = d["naive_argmax_oracle"]
    # true.argmax(1) column-order tie-break = [A, A, B, A] -> overpays
    assert na["model_mix"] == {"A": 3, "B": 1}
    assert na["cost_per_1k_queries"] == pytest.approx(775.0)
    assert na["cost_saving_ceiling"] == pytest.approx(0.225)
    assert na["offbest_fraction"] == pytest.approx(0.25)
    # the tie-break fix must not hurt the ceiling
    assert d["oracle_cost_saving_ceiling"] >= na["cost_saving_ceiling"]


def test_derived_test_ceiling_savings():
    d = _derived()
    # best_acc 0.75; ref cost 1000/1k
    assert d["router_saving_at_minus1pt_test_ceiling"]["cost_saving_vs_ref"] == pytest.approx(0.10)
    assert d["router_saving_at_minus1pt_test_ceiling"]["lam"] == 0.0
    assert d["router_saving_at_minus3pt_test_ceiling"]["cost_saving_vs_ref"] == pytest.approx(0.60)
    assert d["router_saving_at_minus3pt_test_ceiling"]["lam"] == 0.5


def test_derived_escalation_rate_test_ceiling():
    d = _derived()
    # lam0: route == argmax(true) == oracle == [A,A,B,A] -> 25% off A
    assert d["zoib_escalation_rate_lam0"] == pytest.approx(0.25)
    # test-chosen -3pt lambda = 0.5 (accuracy 0.73 closest to 0.72)
    assert d["minus3pt_lambda_test_ceiling"] == 0.5
    # U = true - 0.5*cost/cmax -> choices [B,A,B,B] -> 75% off A
    assert d["zoib_escalation_rate_at_minus3pt_test_ceiling"] == pytest.approx(0.75)


def test_derived_without_validation_frontier_has_no_headline_saving():
    d = _derived()
    assert d["lambda_selection"] == "unavailable"
    assert d["router_saving_at_minus1pt"] is None
    assert d["router_saving_at_minus3pt"] is None
    assert d["minus3pt_lambda"] is None
    assert d["zoib_escalation_rate_at_minus3pt"] is None


def test_derived_headline_lambda_is_chosen_on_validation_and_measured_on_test():
    d = _derived(val_frontier=VAL_FRONTIER, val_best_acc=VAL_BEST_ACC)
    assert d["lambda_selection"] == "validation"
    s1 = d["router_saving_at_minus1pt"]
    # validation band >= 0.79 -> {lam 0.0, lam 1.0}; cheapest on validation = lam 1.0
    assert s1["lam"] == 1.0
    assert s1["val_accuracy"] == pytest.approx(0.795)
    # ... reported with the *test* row at lam 1.0, not the test-optimal lam 0.0
    assert s1["accuracy"] == pytest.approx(0.60)
    assert s1["d_accuracy_vs_best_single"] == pytest.approx(-0.15)
    assert s1["cost_saving_vs_ref"] == pytest.approx(0.85)
    assert d["router_saving_at_minus1pt_test_ceiling"]["lam"] == 0.0
    # -3pt lambda: validation accuracy closest to 0.77 is lam 1.0 (0.795)
    assert d["minus3pt_lambda"] == 1.0
    # U = true - cost -> [B, A, B, B] -> 75% off A
    assert d["zoib_escalation_rate_at_minus3pt"] == pytest.approx(0.75)


def test_val_selected_saving_rejects_lambda_off_the_test_grid():
    val = VAL_FRONTIER.assign(lam=[0.0, 0.5, 7.0])
    with pytest.raises(ValueError, match="not on the test frontier grid"):
        _val_selected_saving(val, FRONTIER, 1000.0, val_best_acc=VAL_BEST_ACC,
                             test_best_acc=0.75, margin=0.01)


def test_frontier_saving_returns_none_when_nothing_qualifies():
    assert _frontier_saving(FRONTIER, 1000.0, best_acc=0.99, margin=0.01) is None


# --------------------------------------------------------------------------- #
# ablation pool arithmetic                                                    #
# --------------------------------------------------------------------------- #
def test_ablation_removes_exactly_the_named_models():
    full = ["A", "B", "C", "D"]
    added, reduced = ablation_pools(["C", "D"], full)
    assert added == ["C", "D"]
    assert reduced == ["A", "B"]


def test_ablation_ignores_names_not_in_pool_and_preserves_order():
    full = ["gpt-4", "yi-34b-chat", "mistral-7b"]
    added, reduced = ablation_pools(["yi-34b-chat", "not-in-pool"], full)
    assert added == ["yi-34b-chat"]
    assert reduced == ["gpt-4", "mistral-7b"]


def test_ablation_empty_when_no_added_models():
    added, reduced = ablation_pools([], ["A", "B"])
    assert added == []
    assert reduced == ["A", "B"]


# --------------------------------------------------------------------------- #
# routing / ablation on toy observations (injected predictions)               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy_data():
    return FakeTrainingData(make_split_obs(n_queries=120, n_models=3, seed=1))


def _oracle_preds(d, split):
    obs = d.nirt_observations()
    return pivot_qm(obs[obs["split"] == split], "target")


@pytest.fixture
def no_checkpoint_loads(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("checkpoint load attempted")

    monkeypatch.setattr(battery, "_pred_matrix", _boom)
    monkeypatch.setattr(battery, "_nirt_matrix", _boom)


def test_routing_and_ablation_reuse_injected_predictions(toy_data, no_checkpoint_loads):
    test_pred = _oracle_preds(toy_data, "test")
    preds = {"ZOIB  E[Y]": test_pred, "baseline NIRT (Bernoulli)": test_pred}
    val_ey = _oracle_preds(toy_data, "validation")
    pool = ["m0", "m1", "m2"]

    routing = routing_block(toy_data, "test", preds=preds, val_zoib_ey=val_ey,
                            pool_models=pool, reference="m2")
    assert routing["model_ids"] == pool
    assert routing["derived"]["lambda_selection"] == "validation"
    assert routing["zoib_frontier_validation"] is not None
    assert routing["nirt_error"] is None

    abl = ablation_block(toy_data, "test", added_models=["m1", "m2"], full_pool=pool,
                         full_derived=routing["derived"], preds=preds,
                         val_zoib_ey=val_ey, reference="m2")
    assert abl["status"] == "ran"
    assert abl["all_removed"]["reduced_pool"] == ["m0"]
    assert set(abl["leave_one_out"]) == {"m1", "m2"}


def test_routing_without_validation_predictions_is_marked_unavailable(toy_data, no_checkpoint_loads):
    preds = {"ZOIB  E[Y]": _oracle_preds(toy_data, "test")}
    routing = routing_block(toy_data, "test", preds=preds, pool_models=["m0", "m1"])
    der = routing["derived"]
    assert der["lambda_selection"] == "unavailable"
    assert "no validation" in der["lambda_selection_reason"]
    assert der["router_saving_at_minus3pt"] is None


def test_pool_matrices_fails_fast_on_model_missing_from_split(toy_data):
    with pytest.raises(ValueError, match="ghost-model"):
        _pool_matrices(toy_data, "test", ["m0", "ghost-model"])


def test_pool_matrices_fails_fast_when_no_query_is_dense(toy_data):
    obs = toy_data.nirt_observations()
    test_q = sorted(obs.loc[obs["split"] == "test", "query_id"].unique())
    half = set(test_q[: len(test_q) // 2])
    # m0 only on the first half of test queries, m1 only on the second half
    drop = ((obs["model_id"] == "m0") & ~obs["query_id"].isin(half)) | (
        (obs["model_id"] == "m1") & obs["query_id"].isin(half)
    )
    d = FakeTrainingData(obs[~(drop & (obs["split"] == "test"))])
    with pytest.raises(ValueError, match="no 'test' query"):
        _pool_matrices(d, "test", ["m0", "m1"])


def test_prediction_matrices_records_nirt_failure(monkeypatch):
    frame = pd.DataFrame({"m0": [0.5]}, index=["q0"])
    monkeypatch.setattr(battery, "_pred_matrix", lambda *a, **k: frame)

    def _fail(*a, **k):
        raise RuntimeError("query-features dim mismatch")

    monkeypatch.setattr(battery, "_nirt_matrix", _fail)
    with pytest.warns(UserWarning, match="nirt-x"):
        preds, err = prediction_matrices(None, None, "test", zoib="z", bernoulli="b",
                                         nirt_run="nirt-x")
    assert "dim mismatch" in err
    assert not any(k.startswith("NIRT") for k in preds)


# --------------------------------------------------------------------------- #
# provenance                                                                  #
# --------------------------------------------------------------------------- #
def test_provenance_hashes_the_configs_own_splits_and_config(tmp_path):
    splits = tmp_path / "custom_splits"
    splits.mkdir()
    for name in ("train.json", "validation.json", "test.json", "ood.json"):
        (splits / name).write_text(json.dumps({"query_ids": [name]}), encoding="utf-8")
    cfg_file = tmp_path / "irt_router.yaml"
    cfg_file.write_text("paths:\n  splits: custom_splits\n", encoding="utf-8")
    cfg = Config({"paths": {"splits": "custom_splits"}}, root=tmp_path, source_path=cfg_file)

    prov = provenance(cfg, phase="E9", pool_models=["m0"], checkpoints={"zoib": "ck"})
    h = prov["artefact_hashes"]
    assert prov["splits_dir"] == str(splits)
    assert h["splits/ood.json"] == file_digest(splits / "ood.json")
    assert h["splits/test.json"] == file_digest(splits / "test.json")
    assert h["splits/cold_start_models.json"] is None  # expected but absent -> recorded
    assert h["config/irt_router.yaml"] == file_digest(cfg_file)
    assert h["config_resolved"]
    assert "config/phase0.yaml" not in h


def test_provenance_skips_absent_ood(tmp_path):
    (tmp_path / "s").mkdir()
    cfg = Config({"paths": {"splits": "s"}}, root=tmp_path)
    h = provenance(cfg, phase="E9", pool_models=[], checkpoints={})["artefact_hashes"]
    assert "splits/ood.json" not in h
    assert "splits/test.json" in h


def test_load_config_records_its_source_path():
    cfg = load_config()
    assert cfg.source_path is not None and cfg.source_path.exists()


# --------------------------------------------------------------------------- #
# ledger row shape + rendering                                                #
# --------------------------------------------------------------------------- #
def _fake_result(phase="E9"):
    return {
        "phase": phase,
        "sources": ["routerbench"],
        "notes": "toy",
        "pool": {"n_warm_models": 9, "gpt4_cheapest_cost_ratio": 73.1},
        "prediction": {"zoib": {"nll_mean": 0.388, "mae": 0.337, "mean_calibration_ece": 0.012}},
        "routing": {
            "zoib_aiq": {"aiq_improvement": 0.21},
            "derived": {
                "oracle_cost_saving_ceiling": 0.76,
                "oracle_offbest_fraction": 0.94,
                "router_saving_at_minus1pt": {"cost_saving_vs_ref": 0.12},
                "router_saving_at_minus3pt": {"cost_saving_vs_ref": 0.44},
                "router_saving_at_minus1pt_test_ceiling": {"cost_saving_vs_ref": 0.15},
                "router_saving_at_minus3pt_test_ceiling": {"cost_saving_vs_ref": 0.49},
            },
        },
        "cold_start": {"status": "ran", "beats_global_mean_on_bce": False},
    }


def test_ledger_row_has_exactly_the_ledger_columns():
    row = ledger_row(_fake_result())
    assert set(row) == set(LEDGER_COLUMNS)
    assert row["zoib_saving_at_minus3pt"] == 0.44
    assert row["zoib_saving_at_minus3pt_test_ceiling"] == 0.49
    assert row["coldstart_beats_global_mean"] is False


def test_ledger_row_none_saving_when_frontier_empty():
    r = _fake_result()
    r["routing"]["derived"]["router_saving_at_minus1pt"] = None
    assert ledger_row(r)["zoib_saving_at_minus1pt"] is None


def test_ledger_row_notes_surface_nirt_failure():
    r = _fake_result()
    r["routing"].update(nirt_run="nirt-2d-projected", nirt_error="RuntimeError: boom")
    notes = ledger_row(r)["notes"]
    assert notes.startswith("toy; ")
    assert "nirt-2d-projected" in notes and "boom" in notes


def test_render_ledger_md_writes_table(tmp_path, monkeypatch):
    from evaluation import pool_expansion as pe

    art = tmp_path / "artifacts" / "pool_expansion"
    art.mkdir(parents=True)
    (art / "ledger.json").write_text(json.dumps([ledger_row(_fake_result("E0"))]))
    (tmp_path / "docs").mkdir()

    class _Cfg:
        root = tmp_path

    monkeypatch.setattr(pe.battery, "load_config", lambda *a, **k: _Cfg())
    out = pe.render_ledger_md(cfg=_Cfg())
    text = out.read_text()
    assert "| phase |" in text
    assert "E0" in text
    assert text.count("\n|") >= 3  # header + separator + one row
