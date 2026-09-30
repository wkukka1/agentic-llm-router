"""MOD-001 (training.nirt): an explicit relative runs_dir must resolve the
same way the baseline trainer's explicit relative out_dir does -- against
phase0_cfg's root, not the process's current working directory."""

from __future__ import annotations

from router.config import Config
from training.nirt.train import _resolve_runs_dir


def test_explicit_relative_runs_dir_joins_against_phase0_root(tmp_path):
    phase0_cfg = Config({}, root=tmp_path)

    resolved = _resolve_runs_dir({}, phase0_cfg, runs_dir="my_runs")

    assert resolved == tmp_path / "my_runs"


def test_explicit_absolute_runs_dir_is_left_untouched(tmp_path):
    absolute = tmp_path / "abs_runs"
    phase0_cfg = Config({}, root=tmp_path / "elsewhere")

    resolved = _resolve_runs_dir({}, phase0_cfg, runs_dir=absolute)

    assert resolved == absolute


def test_no_explicit_runs_dir_falls_back_to_nirt_cfg_unchanged(tmp_path):
    """The pre-existing default-branch behaviour must not shift."""
    phase0_cfg = Config({}, root=tmp_path)

    resolved = _resolve_runs_dir({"runs_dir": "cfg_runs"}, phase0_cfg)

    assert resolved == tmp_path / "cfg_runs"


def test_explicit_relative_runs_dir_with_no_phase0_cfg_falls_back_to_repo_root():
    from router.config import REPO_ROOT

    resolved = _resolve_runs_dir({}, None, runs_dir="my_runs")

    assert resolved == REPO_ROOT / "my_runs"
