"""``training.nirt.baseline.checkpoint``: save/load round trip and the
``format`` tag guarding against a NIRT / Phase 1-2 baseline checkpoint mix-up
(XA-05 / contracts.md's checkpoint-format mismatches)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.nirt.baseline.checkpoint import (
    FORMAT,
    load_baseline_run,
    load_checkpoint,
    save_checkpoint,
)
from training.nirt.baseline.model import build_baseline_model


def _model():
    return build_baseline_model(
        {"theta_dim": 2, "model_params": "free", "use_length_head": False},
        n_models=3, query_dim=4, profile_dim=4,
    )


def test_save_checkpoint_round_trips(tmp_path):
    m = _model()
    d = save_checkpoint(
        tmp_path / "run", model=m,
        config={"model": {"theta_dim": 2, "model_params": "free"}},
        model_index={"a": 0, "b": 1, "c": 2}, query_dim=4, profile_dim=4, relevance_dim=0,
    )
    loaded, blob = load_checkpoint(d)
    assert blob["format"] == FORMAT
    # arch / dim / model_params were pure duplicates of model_cfg -- dropped
    assert "arch" not in blob and "dim" not in blob and "model_params" not in blob
    assert loaded.model_params == "free"

    model, blob2, settings = load_baseline_run(d)
    assert settings["pathway"] == "irt"   # config.yaml had no data.pathway -> default


def test_load_checkpoint_rejects_nirt_format(tmp_path):
    """Pointing the baseline loader at a NIRT / IRT-Router checkpoint must
    raise a clear, actionable error -- not an opaque KeyError."""
    d = tmp_path / "not-baseline"
    d.mkdir()
    torch.save({"format": "nirt", "state_dict": {}, "config": {}, "n_models": 0,
                "model_index": {}, "query_dim": 4, "profile_dim": 4}, d / "model.pt")
    with pytest.raises(ValueError, match="router.nirt.checkpoint.load_run"):
        load_checkpoint(d)


def test_load_checkpoint_wraps_missing_key_as_value_error(tmp_path):
    """A checkpoint with no format tag (pre-fix, or genuinely malformed) still
    fails with a message naming the likely mismatch, not a bare KeyError."""
    d = tmp_path / "bad"
    d.mkdir()
    torch.save({"state_dict": {}}, d / "model.pt")
    with pytest.raises(ValueError, match="does not look like a Phase 1/2 baseline"):
        load_checkpoint(d)
