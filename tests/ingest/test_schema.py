"""The collection server accepts exactly what the gateway sends and nothing that could carry text."""

from __future__ import annotations

import pytest
from aar_gateway import share

from ingest.schema import validate_record

LOGGED = {"ts": "2026-10-18T10:00:00+00:00", "requested_model": "claude-opus-5-5", "routed_model": "claude-haiku-5-5",
          "reason": "cheapest within 0.05 utility of the best", "status": 200, "subagent": False, "agent_role": None,
          "input_tokens": 12, "output_tokens": 7, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 3,
          "cost_usd": 0.0123, "cost_if_requested_usd": 0.05, "latency_s": 1.4, "correction": True, "thread": "continue",
          "error": "invalid_request_error: messages.3: a long free-text error"}


@pytest.fixture(autouse=True)
def _no_encoder(monkeypatch):
    monkeypatch.setattr(share, "_embed", lambda text: [(-1) ** i * (i % 127) for i in range(share.DIMS)])


def test_what_the_gateway_really_sends_is_accepted():
    """The client/server contract: build_record's output, with and without the embedding."""
    for embed in (True, False):
        row = validate_record(share.build_record(LOGGED, "some typed message", embed=embed))
        assert row["v"] == 1 and row["day"] == "2026-10-18"
    assert validate_record(share.build_record({**LOGGED, "error": None, "cost_usd": None}, None))["prompt_chars"] is None


def _row(**changes):
    return {**share.build_record(LOGGED, "typed message"), **changes}


@pytest.mark.parametrize("bad", [
    {"prompt": "the user's message"},                                    # any unknown key, text above all
    {"text": "x"}, {"cwd": "/home/me/project"}, {"conversation": "abc123"},
    {"v": 2}, {"v": "1"},
    {"day": "2026-10-18T10:00:00"}, {"day": "yesterday"},
    {"requested_model": "x" * 200}, {"routed_model": "two words and a newline\n"}, {"routed_model": 5},
    {"error_type": "a free text sentence with spaces, and: punctuation that is clearly a message"},
    {"status": "200"}, {"status": 99}, {"status": 700},
    {"input_tokens": -1}, {"input_tokens": 1.5}, {"cost_usd": -0.01}, {"latency_s": "fast"},
    {"correction": "yes"},
    {"embedding": [1] * 63}, {"embedding": [1] * 65}, {"embedding": [200] + [0] * 63}, {"embedding": ["a"] * 64},
    {"embedding": "a whole sentence of text pretending to be an embedding"},
    {"prompt_chars": 0}, {"prompt_chars": 100},                       # not a power of two
    {"embedding_model": "x" * 300},
])
def test_anything_that_could_carry_text_or_breaks_the_format_is_refused(bad):
    with pytest.raises(ValueError):
        validate_record(_row(**bad))


def test_a_missing_field_is_refused_a_null_optional_is_fine():
    row = _row()
    del row["status"]
    with pytest.raises(ValueError):
        validate_record(row)
    validate_record(_row(thread=None, subagent_role=None, latency_s=None, embedding=None, prompt_chars=None))


def test_the_server_does_not_depend_on_the_gateway_package():
    import pathlib

    import ingest

    for path in pathlib.Path(ingest.__file__).parent.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import router" not in text and "from router" not in text, path.name


@pytest.mark.parametrize("error", [
    "Request too large for model: tokens exceeded",       # free text before the first colon
    "overloaded_error: Overloaded", "invalid_request_error: messages.3: x", "", None,
    "ReadTimeout", "a" * 200, "Ünïcode errör: x",
])
def test_whatever_error_text_a_provider_returns_the_record_still_passes_the_server(error):
    row = share.build_record({**LOGGED, "error": error}, "typed message")
    validate_record(row)  # never rejected
    assert row["error_type"] is None or len(row["error_type"]) <= 64
    if error and ":" in error and error.split(":")[0].isidentifier():
        assert row["error_type"] == error.split(":")[0]  # a real error type is kept as is


@pytest.mark.parametrize("field,value", [("requested_model", "My Local Model (v2)"), ("routed_model", "модель"),
                                         ("agent_role", "a role with spaces"), ("thread", "weird thread mode!")])
def test_unusual_names_are_mapped_into_the_format_not_sent_raw(field, value):
    row = share.build_record({**LOGGED, field: value}, "typed message")
    validate_record(row)  # never rejected by the server
    assert value not in str(row.values())
