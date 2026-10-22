from evaluation.gateway import multiturn as mt
from evaluation.gateway import pool_eval as pe


def _user(text, origin="human"):
    return {"type": "user", "origin": {"kind": origin}, "message": {"content": [{"type": "text", "text": text}]}}


def _assistant(text):
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


TOOL_RESULT = {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}}
FIRST = "Please add a retry to the http client in fetch.py"
SECOND = "Now make the retry count configurable from settings.yaml as well"


def test_turns_skip_first_message_and_tool_results_and_route_on_the_message():
    conv = mt.claude_code_conversation([_user(FIRST), _assistant("Added retry with backoff."),
                                        TOOL_RESULT, _user(SECOND)])
    rows = mt.conversation_turns(conv, source="own", group="s1")
    assert [r["route_text"] for r in rows] == [SECOND]  # first message has no history; tool result is no turn
    assert FIRST in rows[0]["prompt"] and "Added retry" in rows[0]["prompt"]
    assert rows[0]["prompt"].endswith(SECOND)
    # pool_eval recovers the routed text from the graded prompt
    assert rows[0]["prompt"].rpartition("</earlier conversation>\n\n")[2] == SECOND


def test_transcript_without_origin_counts_typed_messages_only():
    old = lambda t: {"type": "user", "message": {"content": t}}  # noqa: E731
    conv = mt.claude_code_conversation([old(FIRST), TOOL_RESULT, old(SECOND)])
    assert conv == [("user", FIRST), ("user", None), ("user", SECOND)]


def test_context_is_capped():
    prior = [("user", "x" * 5000), ("assistant", "y" * 5000)]
    out = mt.with_context(prior, "the message")
    assert len(out) <= mt.CONTEXT_CHARS + len("<earlier conversation>\n\n</earlier conversation>\n\nthe message")
    assert "y" * 100 in out  # newest text kept, oldest cut


def test_sample_caps_turns_per_session():
    rows = [mt._row(f"prompt {i}", group="same") for i in range(5)] + [mt._row("other", group="b")]
    picked = mt.sample(rows, 10, seed=0)
    assert sum(r["group"] == "same" for r in picked) == mt.PER_SESSION and len(picked) == 3


def test_load_prompts_keeps_single_turn_prompts_whole(tmp_path):
    g = tmp_path / "graded.jsonl"
    g.write_text("".join(
        f'{{"prompt_hash": "p", "prompt": "plain q", "model": "{t}", "score": 1, "usage_usd": 0.1}}\n'
        for t in pe.TIERS))
    assert pe.load_prompts(g)[0]["route_text"] == "plain q"
