from __future__ import annotations

import pandas as pd
import pytest

from router.config import load_config
from router.models.profiles import (
    build_model_profiles,
    empirical_stats,
    read_model_profiles,
    render_profile_text,
    write_model_profiles,
)
from router.data.response_matrix import build_tables


class _CfgWrap:
    """Real config for profile settings, but responses passed in directly."""

    def __init__(self):
        self._c = load_config()

    def __getattr__(self, k):
        return getattr(self._c, k)

    def get(self, k, d=None):
        return self._c.get(k, d)

    def resolve(self, p):
        return self._c.resolve(p)

    def path(self, k):
        return self._c.path(k)


@pytest.fixture
def corrected(toy_responses):
    return build_tables(cfg=_ChanceCfg(), responses=toy_responses)["responses"]


class _ChanceCfg:
    def get(self, k, d=None):
        return {"chance_correction": {"method": "normalized", "clip": True,
                                      "warn_on_missing_choices": False}}.get(k, d)


def test_empirical_stats_shapes(corrected):
    cfg = _CfgWrap()
    s = empirical_stats(corrected, "gpt-4-1106-preview", cfg)
    assert s["n_observations"] > 0
    assert "overall_correctness" in s
    assert 0.0 <= s["overall_correctness"] <= 1.0
    # gpt-4 has an arena battle in the toy data
    assert "pairwise" in s


def test_empirical_stats_empty_for_unknown_model(corrected):
    assert empirical_stats(corrected, "no-such-model", _CfgWrap()) == {}


def test_render_uses_curated_feature_when_present():
    feature, full, structured = render_profile_text(
        "gpt-4-1106-preview",
        {"feature": "GPT-4 Turbo is strong.", "context_window": 128000},
        {"overall_correctness": 0.8, "n_datasets": 5},
        include_empirical=True,
    )
    assert feature == "GPT-4 Turbo is strong."
    assert "128000-token context window" in full
    assert "overall correctness 0.80" in full


def test_render_template_fallback_no_curated():
    feature, full, _ = render_profile_text("mixtral-8x7b-instruct", None, {},
                                           include_empirical=False)
    assert "mixtral" in feature.lower() or "Mixtral" in feature
    assert full  # non-empty


def test_include_empirical_toggle():
    stats = {"overall_correctness": 0.7, "n_datasets": 3}
    _, with_emp, _ = render_profile_text("m", {"feature": "A model."}, stats, True)
    _, without, _ = render_profile_text("m", {"feature": "A model."}, stats, False)
    assert "Observed behaviour" in with_emp
    assert "Observed behaviour" not in without


def test_build_model_profiles_covers_all_models(corrected):
    df = build_model_profiles(_CfgWrap(), responses=corrected)
    assert set(df["model_id"]) == set(corrected["model_id"].unique())
    assert df["profile_text"].str.len().gt(0).all()
    assert df.loc[df["model_id"] == "gpt-4-1106-preview", "has_curated"].iloc[0]


def test_profiles_parquet_roundtrip(tmp_path, corrected, monkeypatch):
    cfg = _CfgWrap()
    df = build_model_profiles(cfg, responses=corrected)
    monkeypatch.setattr(cfg, "path", lambda k: tmp_path)
    p = write_model_profiles(df, cfg)
    back = read_model_profiles(cfg)
    assert p.exists()
    assert len(back) == len(df)
    assert isinstance(back.iloc[0]["structured"], dict)
    assert isinstance(back.iloc[0]["empirical"], dict)
