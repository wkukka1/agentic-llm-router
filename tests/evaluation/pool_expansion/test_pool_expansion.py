"""Pool-expansion battery: derived headline numbers, ablation pool arithmetic,
ledger row shape / rendering.  Pure-function tests -- no checkpoint loading."""

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
from evaluation.pool_expansion.battery import _frontier_saving


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


def test_derived_oracle_ceiling_and_offbest():
    d = derived_headline(TRUE, COST, TRUE, model_ids=MODEL_IDS, tq_acc=TQ_ACC,
                         frontier=FRONTIER, reference="A")
    assert d["best_single_model"] == "A"
    assert d["best_single_test_accuracy"] == pytest.approx(0.75)
    # oracle = cheapest model at per-query max score = [B, A, B, B]; cost 0.325 vs A's 1.0
    assert d["oracle_cost_per_1k_queries"] == pytest.approx(325.0)
    assert d["oracle_cost_saving_ceiling"] == pytest.approx(0.675)
    assert d["oracle_offbest_fraction"] == pytest.approx(0.75)
    assert d["oracle_model_mix"] == {"A": 1, "B": 3}


def test_derived_naive_argmax_oracle_recorded_for_comparison():
    d = derived_headline(TRUE, COST, TRUE, model_ids=MODEL_IDS, tq_acc=TQ_ACC,
                         frontier=FRONTIER, reference="A")
    na = d["naive_argmax_oracle"]
    # true.argmax(1) column-order tie-break = [A, A, B, A] -> overpays
    assert na["model_mix"] == {"A": 3, "B": 1}
    assert na["cost_per_1k_queries"] == pytest.approx(775.0)
    assert na["cost_saving_ceiling"] == pytest.approx(0.225)
    assert na["offbest_fraction"] == pytest.approx(0.25)
    # the tie-break fix must not hurt the ceiling
    assert d["oracle_cost_saving_ceiling"] >= na["cost_saving_ceiling"]


def test_derived_matched_accuracy_savings():
    d = derived_headline(TRUE, COST, TRUE, model_ids=MODEL_IDS, tq_acc=TQ_ACC,
                         frontier=FRONTIER, reference="A")
    # best_acc 0.75; ref cost 1000/1k
    assert d["router_saving_at_minus1pt"]["cost_saving_vs_ref"] == pytest.approx(0.10)
    assert d["router_saving_at_minus1pt"]["lam"] == 0.0
    assert d["router_saving_at_minus3pt"]["cost_saving_vs_ref"] == pytest.approx(0.60)
    assert d["router_saving_at_minus3pt"]["lam"] == 0.5


def test_derived_escalation_rate():
    d = derived_headline(TRUE, COST, TRUE, model_ids=MODEL_IDS, tq_acc=TQ_ACC,
                         frontier=FRONTIER, reference="A")
    # lam0: route == argmax(true) == oracle == [A,A,B,A] -> 25% off A
    assert d["zoib_escalation_rate_lam0"] == pytest.approx(0.25)
    # -3pt lambda picked = 0.5 (accuracy 0.73 closest to 0.72)
    assert d["minus3pt_lambda"] == 0.5
    # U = true - 0.5*cost/cmax -> choices [B,A,B,B] -> 75% off A
    assert d["zoib_escalation_rate_at_minus3pt"] == pytest.approx(0.75)


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
                "router_saving_at_minus1pt": {"cost_saving_vs_ref": 0.15},
                "router_saving_at_minus3pt": {"cost_saving_vs_ref": 0.49},
            },
        },
        "cold_start": {"status": "ran", "beats_global_mean_on_bce": False},
    }


def test_ledger_row_has_exactly_the_ledger_columns():
    row = ledger_row(_fake_result())
    assert set(row) == set(LEDGER_COLUMNS)
    assert row["zoib_saving_at_minus3pt"] == 0.49
    assert row["coldstart_beats_global_mean"] is False


def test_ledger_row_none_saving_when_frontier_empty():
    r = _fake_result()
    r["routing"]["derived"]["router_saving_at_minus1pt"] = None
    assert ledger_row(r)["zoib_saving_at_minus1pt"] is None


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
