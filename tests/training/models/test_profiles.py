from __future__ import annotations

import pytest

from router.config import load_config
from training.models.profiles import (
    build_model_profiles,
    build_profile_embeddings,
    empirical_stats,
    read_model_profiles,
    render_profile_text,
    write_model_profiles,
)


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
def corrected(toy_tables):
    return toy_tables["responses"]


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


def test_build_model_profiles_has_a_text_column_per_ablation_variant(corrected):
    from training.models.profiles import VARIANTS

    df = build_model_profiles(_CfgWrap(), responses=corrected)
    for variant in VARIANTS:
        col = f"profile_text_{variant}"
        assert col in df.columns, f"missing {col}"
        assert df[col].str.len().gt(0).all()
    # P3 is the full/default variant -- same text as the plain profile_text column
    assert (df["profile_text_P3"] == df["profile_text"]).all()


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


def test_build_profile_embeddings_covers_all_models(isolated_training_data):
    """Profile text already built + written: build_profile_embeddings should
    just embed it, keyed by model_id."""
    df = build_model_profiles(isolated_training_data.cfg, responses=isolated_training_data.responses)
    write_model_profiles(df, isolated_training_data.cfg)
    try:
        store = build_profile_embeddings(isolated_training_data.cfg, "irt")
    except Exception as exc:  # model not cached / offline
        pytest.skip(f"encoder unavailable: {exc}")
    assert set(store._index) == set(isolated_training_data.responses["model_id"].unique())


def test_render_benchmarks_sentence_and_structured():
    benchmarks = {
        "mmlu": {"value": 70.6, "unit": "percent", "source": "hf_open_llm_leaderboard",
                 "version": "v1", "snapshot_date": None},
        "arena_elo": {"value": 1116, "unit": "elo", "source": "lmarena_hf",
                      "version": None, "snapshot_date": "2023-12-15"},
    }
    feature, full, structured = render_profile_text(
        "mixtral-8x7b-instruct", {"feature": "Mixtral is strong."}, {}, include_empirical=False,
        benchmarks=benchmarks,
    )
    assert "MMLU 70.6" in full
    assert "Chatbot Arena Elo 1116 (2023-12-15 snapshot)" in full
    assert structured["benchmarks"] == benchmarks
    assert structured["benchmark_scores"] == {"mmlu": 70.6, "arena_elo": 1116}
    assert structured["benchmark_scores_source"] == {
        "mmlu": "hf_open_llm_leaderboard", "arena_elo": "lmarena_hf",
    }


def test_render_benchlm_attribution_appears_once():
    benchmarks = {"overall": {"value": 38.45, "unit": "benchlm_score", "source": "benchlm",
                               "version": "v1", "snapshot_date": "2026-10-06"}}
    _, full, _ = render_profile_text("claude-sonnet-4", {"feature": "x."}, {}, include_empirical=False,
                                     benchmarks=benchmarks)
    assert full.count("Data from BenchLM.ai") == 1


def test_render_strengths_weaknesses():
    entry = {"feature": "A model.", "strengths": ["good at X"], "weaknesses": ["bad at Y"]}
    _, full, structured = render_profile_text("m", entry, {}, include_empirical=False)
    assert "Strengths: good at X." in full
    assert "Weaknesses: bad at Y." in full
    assert structured["strengths"] == ["good at X"]
    assert structured["weaknesses"] == ["bad at Y"]
    assert structured["strengths_source"] == "editorial"
    assert structured["weaknesses_source"] == "editorial"


def test_render_no_benchmarks_or_editorial_unchanged():
    _, full, structured = render_profile_text("m", {"feature": "A model."}, {}, include_empirical=False)
    assert "benchmark scores" not in full.lower()
    assert structured["benchmarks"] == {}
    assert structured["strengths"] == []
    assert structured["weaknesses"] == []


def test_resolve_benchmarks_calls_each_configured_source(monkeypatch):
    import training.models.profiles as profiles_mod

    calls = []

    def fake_hf(repo_id, cfg=None):
        calls.append(("hf", repo_id))
        return {"mmlu": {"value": 1.0, "unit": "percent", "source": "hf_open_llm_leaderboard",
                          "version": "v1", "snapshot_date": None}}

    def fake_arena(name, cfg=None, snapshot_date=None):
        calls.append(("arena", name))
        return {"arena_elo": {"value": 2.0, "unit": "elo", "source": "lmarena_hf",
                               "version": None, "snapshot_date": "2023-12-15"}}

    def fake_benchlm(slug, cfg=None):
        calls.append(("benchlm", slug))
        return None

    monkeypatch.setattr(profiles_mod, "hf_open_llm_leaderboard", fake_hf)
    monkeypatch.setattr(profiles_mod, "arena_elo_rating", fake_arena)
    monkeypatch.setattr(profiles_mod, "benchlm_scores", fake_benchlm)

    entry = {"hf_repo_id": "org/model", "arena_name": "Model-Name", "benchlm_slug": "model-slug"}
    result = profiles_mod.resolve_benchmarks(entry, cfg=None)

    assert ("hf", "org/model") in calls
    assert ("arena", "Model-Name") in calls
    assert ("benchlm", "model-slug") in calls
    assert result["mmlu"]["value"] == 1.0
    assert result["arena_elo"]["value"] == 2.0


def test_resolve_benchmarks_skips_unconfigured_sources(monkeypatch):
    import training.models.profiles as profiles_mod

    calls = []
    monkeypatch.setattr(profiles_mod, "hf_open_llm_leaderboard",
                        lambda *a, **k: calls.append("hf") or None)
    monkeypatch.setattr(profiles_mod, "arena_elo_rating",
                        lambda *a, **k: calls.append("arena") or None)
    monkeypatch.setattr(profiles_mod, "benchlm_scores",
                        lambda *a, **k: calls.append("benchlm") or None)

    result = profiles_mod.resolve_benchmarks({}, cfg=None)
    assert calls == []
    assert result == {}


def test_resolve_benchmarks_manual_fallback_fills_gaps_only():
    import training.models.profiles as profiles_mod

    entry = {"benchmark_scores": {"humaneval": 90.2}}
    result = profiles_mod.resolve_benchmarks(entry, cfg=None)
    assert result["humaneval"] == {"value": 90.2, "unit": "percent", "source": "manual",
                                    "version": None, "snapshot_date": None}


def test_resolve_benchmarks_openlm_scores_tagged_source():
    import training.models.profiles as profiles_mod

    entry = {"openlm_scores": {"aaii": 55.0}}
    result = profiles_mod.resolve_benchmarks(entry, cfg=None)
    assert result["aaii"] == {"value": 55.0, "unit": "percent", "source": "openlm",
                               "version": None, "snapshot_date": None}


def test_resolve_benchmarks_model_card_scores_tagged_source():
    import training.models.profiles as profiles_mod

    entry = {"model_card_scores": {"mmlu": 88.7}}
    result = profiles_mod.resolve_benchmarks(entry, cfg=None)
    assert result["mmlu"] == {"value": 88.7, "unit": "percent", "source": "model_card",
                               "version": None, "snapshot_date": None}


def test_resolve_benchmarks_precedence_openlm_beats_everything_on_collision(monkeypatch):
    """Tier order: OpenLM (1) > official model card (2) > BenchLM (3) >
    HF Open LLM Leaderboard (4) > generic manual fallback (5). Arena Elo is a
    separate, orthogonal key (never part of this chain) -- this test forces
    all five tiers to report the SAME key to prove the order, which would
    never happen in practice (the automated sources use disjoint key names)."""
    import training.models.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "benchlm_scores",
                        lambda *a, **k: {"x": {"value": 3.0, "unit": "benchlm_score",
                                                "source": "benchlm", "version": None,
                                                "snapshot_date": None}})
    monkeypatch.setattr(profiles_mod, "hf_open_llm_leaderboard",
                        lambda *a, **k: {"x": {"value": 4.0, "unit": "percent",
                                                "source": "hf_open_llm_leaderboard",
                                                "version": "v1", "snapshot_date": None}})
    entry = {
        "openlm_scores": {"x": 1.0}, "model_card_scores": {"x": 2.0},
        "benchlm_slug": "s", "hf_repo_id": "r", "benchmark_scores": {"x": 5.0},
    }
    result = profiles_mod.resolve_benchmarks(entry, cfg=None)
    assert result["x"]["value"] == 1.0 and result["x"]["source"] == "openlm"


def test_resolve_benchmarks_precedence_model_card_beats_benchlm_and_below(monkeypatch):
    import training.models.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "benchlm_scores",
                        lambda *a, **k: {"x": {"value": 3.0, "unit": "benchlm_score",
                                                "source": "benchlm", "version": None,
                                                "snapshot_date": None}})
    entry = {"model_card_scores": {"x": 2.0}, "benchlm_slug": "s"}
    result = profiles_mod.resolve_benchmarks(entry, cfg=None)
    assert result["x"]["value"] == 2.0 and result["x"]["source"] == "model_card"


_ABLATION_BENCHMARKS = {
    "arena_elo": {"value": 1116, "unit": "elo", "source": "lmarena_hf",
                  "version": None, "snapshot_date": "2023-12-15"},
    "mmlu": {"value": 70.6, "unit": "percent", "source": "hf_open_llm_leaderboard",
             "version": "v1", "snapshot_date": None},
}
_ABLATION_ENTRY = {"feature": "Mixtral 8x7B Instruct is Mistral AI's model.",
                    "strengths": ["good at X"], "weaknesses": ["bad at Y"]}


def test_variant_p0_has_neither_benchmarks_nor_editorial():
    _, full, _ = render_profile_text("m", _ABLATION_ENTRY, {}, include_empirical=False,
                                     benchmarks=_ABLATION_BENCHMARKS, variant="P0")
    assert "benchmark scores" not in full.lower()
    assert "Strengths" not in full and "Weaknesses" not in full


def test_variant_p1_has_benchmarks_not_editorial():
    _, full, _ = render_profile_text("m", _ABLATION_ENTRY, {}, include_empirical=False,
                                     benchmarks=_ABLATION_BENCHMARKS, variant="P1")
    assert "MMLU 70.6" in full
    assert "Strengths" not in full


def test_variant_p2_has_editorial_not_benchmarks():
    _, full, _ = render_profile_text("m", _ABLATION_ENTRY, {}, include_empirical=False,
                                     benchmarks=_ABLATION_BENCHMARKS, variant="P2")
    assert "benchmark scores" not in full.lower()
    assert "Strengths: good at X." in full


def test_variant_p3_is_full_default():
    _, full, _ = render_profile_text("m", _ABLATION_ENTRY, {}, include_empirical=False,
                                     benchmarks=_ABLATION_BENCHMARKS)  # no variant passed
    assert "MMLU 70.6" in full and "Strengths: good at X." in full


def test_variant_p4_excludes_arena_elo_only():
    _, full, _ = render_profile_text("m", _ABLATION_ENTRY, {}, include_empirical=False,
                                     benchmarks=_ABLATION_BENCHMARKS, variant="P4")
    assert "MMLU 70.6" in full
    assert "Chatbot Arena Elo" not in full
    assert "Strengths: good at X." in full


def test_variant_p5_includes_arena_elo_only_no_editorial():
    _, full, _ = render_profile_text("m", _ABLATION_ENTRY, {}, include_empirical=False,
                                     benchmarks=_ABLATION_BENCHMARKS, variant="P5")
    assert "Chatbot Arena Elo 1116" in full
    assert "MMLU" not in full
    assert "Strengths" not in full


def test_variant_p6_redacts_name_but_keeps_other_facts():
    entry = dict(_ABLATION_ENTRY, context_window=32768)
    feature, full, _ = render_profile_text("m", entry, {}, include_empirical=False,
                                            benchmarks=_ABLATION_BENCHMARKS, variant="P6")
    assert "Mixtral" not in feature and "Mistral" not in feature
    assert "32768-token context window" in full
    assert "MMLU 70.6" in full
    assert "Strengths: good at X." in full


def test_variant_benchmarks_p4_excludes_arena_elo():
    from training.models.profiles import variant_benchmarks

    out = variant_benchmarks(_ABLATION_BENCHMARKS, "P4")
    assert "arena_elo" not in out
    assert "mmlu" in out


def test_variant_benchmarks_p0_is_empty():
    from training.models.profiles import variant_benchmarks

    assert variant_benchmarks(_ABLATION_BENCHMARKS, "P0") == {}


def test_structured_dict_is_variant_independent():
    """`structured` is the complete record regardless of which text variant
    was rendered -- only the rendered text differs per variant (1d)."""
    _, _, structured_p0 = render_profile_text("m", _ABLATION_ENTRY, {}, include_empirical=False,
                                              benchmarks=_ABLATION_BENCHMARKS, variant="P0")
    _, _, structured_p3 = render_profile_text("m", _ABLATION_ENTRY, {}, include_empirical=False,
                                              benchmarks=_ABLATION_BENCHMARKS, variant="P3")
    assert structured_p0["benchmarks"] == structured_p3["benchmarks"] == _ABLATION_BENCHMARKS
    assert structured_p0["strengths"] == structured_p3["strengths"] == ["good at X"]


def test_build_variant_profile_embeddings_uses_distinct_store_per_variant(isolated_training_data):
    """E3's ablation sweep needs each P0-P6 variant to land in its OWN
    persistent store directory -- build_profile_embeddings keys storage only
    on (kind, pathway), so two variants under the same pathway would
    collide/overwrite each other if this didn't give each its own dir."""
    from training.models.profiles import build_variant_profile_embeddings
    from router.embeddings import default_store_dir

    df = build_model_profiles(isolated_training_data.cfg, responses=isolated_training_data.responses)
    write_model_profiles(df, isolated_training_data.cfg)
    try:
        store_p0 = build_variant_profile_embeddings(isolated_training_data.cfg, "irt", "P0")
        store_p4 = build_variant_profile_embeddings(isolated_training_data.cfg, "irt", "P4")
    except Exception as exc:  # model not cached / offline
        pytest.skip(f"encoder unavailable: {exc}")
    assert set(store_p0._index) == set(isolated_training_data.responses["model_id"].unique())
    p0_dir = default_store_dir(isolated_training_data.cfg, "model_profile_P0", "irt")
    p4_dir = default_store_dir(isolated_training_data.cfg, "model_profile_P4", "irt")
    assert p0_dir != p4_dir
    assert p0_dir.exists() and p4_dir.exists()


def test_build_profile_embeddings_auto_builds_missing_profiles(isolated_training_data, monkeypatch):
    """No model_profiles.parquet on disk yet -- build_profile_embeddings must
    fall back to building (and persisting) it from responses, same as the
    former CLI script did."""
    monkeypatch.setattr(
        "training.models.profiles.read_responses",
        lambda cfg: isolated_training_data.responses,
    )
    try:
        store = build_profile_embeddings(isolated_training_data.cfg, "irt")
    except Exception as exc:  # model not cached / offline
        pytest.skip(f"encoder unavailable: {exc}")
    assert set(store._index) == set(isolated_training_data.responses["model_id"].unique())
    # the fallback must have persisted model_profiles.parquet, not just embedded in-memory
    assert (isolated_training_data.cfg.path("processed") / "model_profiles.parquet").exists()
