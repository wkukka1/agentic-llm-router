"""Bad gateway configs fail at load time, not on someone's first request."""

from __future__ import annotations

import pytest

from aar_gateway.config import load_settings


def write(tmp_path, text):
    path = tmp_path / "gateway.yaml"
    path.write_text(text)
    return path


def test_defaults_without_a_config():
    s = load_settings(None)
    assert s["router"]["enabled"] is False
    assert s["strongest"] == "claude-opus-5-5"  # the first listed, not Opus 5, which costs more
    assert s["models"]["claude-haiku-4-5"]["prices"]["input"] == 1.0


def test_openai_lane_needs_a_base_url(tmp_path):
    with pytest.raises(ValueError, match="base_url"):
        load_settings(write(tmp_path, "interceptor: {models: {m: {lane: openai}}}"))


def test_missing_api_key_env_fails_at_load(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    cfg = write(tmp_path, "interceptor: {models: {m: {lane: openai, base_url: 'http://x', api_key_env: SOME_KEY}}}")
    with pytest.raises(ValueError, match="SOME_KEY"):
        load_settings(cfg)


def test_unknown_lane(tmp_path):
    with pytest.raises(ValueError, match="lane"):
        load_settings(write(tmp_path, "interceptor: {models: {m: {lane: carrier-pigeon}}}"))


def test_router_settings_merge_over_defaults(tmp_path):
    s = load_settings(write(tmp_path, "interceptor: {router: {margin: 0.1}}"))
    assert s["router"]["margin"] == 0.1 and s["router"]["horizon_requests"] == 20


def test_sonnet_reads_a_stand_in_between_opus_and_haiku_and_old_configs_are_moved(tmp_path):
    """Seen live: Sonnet's stand-in, claude-sonnet-4, scores below Haiku's (gemini-2.5-flash)
    on real prompts, so the router never picked Sonnet. Configs written at setup still name
    the old default: they get the new stand-in; a stand-in someone chose is kept."""
    from aar_gateway.config import DEFAULT_MODELS, SONNET_STAND_IN, load_settings
    assert DEFAULT_MODELS["claude-sonnet-5-5"]["proxy"] == SONNET_STAND_IN == "qwen3-235b-a22b-2507"
    old = tmp_path / "old.yaml"
    old.write_text("interceptor:\n  models:\n"
                   "    claude-sonnet-5-5: {lane: anthropic, proxy: claude-sonnet-4}\n"
                   "    claude-opus-5-5: {lane: anthropic, proxy: gpt-5}\n")
    assert load_settings(old)["models"]["claude-sonnet-5-5"]["proxy"] == SONNET_STAND_IN
    chosen = tmp_path / "chosen.yaml"
    chosen.write_text("interceptor:\n  models:\n    claude-sonnet-5-5: {lane: anthropic, proxy: glm-4.6}\n")
    assert load_settings(chosen)["models"]["claude-sonnet-5-5"]["proxy"] == "glm-4.6"
