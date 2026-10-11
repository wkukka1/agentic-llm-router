"""Anthropic Messages <-> OpenAI Chat Completions, *with* tools.

What lets the interceptor send a Claude Code request to a non-Anthropic model
(OpenAI, Gemini's OpenAI-compatible endpoint, Ollama, vLLM, OpenRouter ...)
without the agent losing its tools: tool definitions, ``tool_use`` /
``tool_result`` turns, images and streaming are all carried across.

The rule is "lossless or not at all". :func:`untranslatable_reason` names
anything that would not survive the trip -- too many tools, a server tool,
a document block, an image inside a tool result -- and the router then
leaves that model out for that request instead of silently dropping part of
it. Two things are dropped by design, because no other provider can use
them: ``thinking`` blocks (Anthropic's own API also ignores another model's
thinking blocks) and ``cache_control`` markers (cost-only).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Iterable, Mapping, Optional

_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_PASSABLE_BLOCKS = {"text", "image", "tool_use", "tool_result", "thinking", "redacted_thinking"}
_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
         "function_call": "tool_use", "content_filter": "refusal"}


def _blocks(content) -> list:
    if content is None:
        return []
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def untranslatable_reason(body: Mapping, *, max_tools: Optional[int] = None) -> Optional[str]:
    """``None`` if ``body`` survives translation intact, else why not."""
    tools = body.get("tools") or []
    if max_tools is not None and len(tools) > max_tools:
        return f"{len(tools)} tools > provider limit {max_tools}"
    for t in tools:
        if t.get("type") not in (None, "custom") or "input_schema" not in t:
            return f"server/built-in tool {t.get('name') or t.get('type')!r}"
        if not _TOOL_NAME.match(t.get("name", "")):
            return f"tool name {t.get('name')!r} not valid for OpenAI-style APIs"
    for m in body.get("messages", []):
        if m.get("role") not in ("user", "assistant", "system"):
            return f"message role {m.get('role')!r}"
        for b in _blocks(m.get("content")):
            kind = b.get("type")
            if kind not in _PASSABLE_BLOCKS:
                return f"content block {kind!r}"
            if kind == "image" and (b.get("source") or {}).get("type") not in ("base64", "url"):
                return "image source type"
            if kind == "tool_result" and any(
                    x.get("type") != "text" for x in _blocks(b.get("content"))):
                return "non-text tool result (e.g. an image from Read)"
    for b in _blocks(body.get("system")):
        if b.get("type") != "text":
            return f"system block {b.get('type')!r}"
    return None


def _text(blocks) -> str:
    return "".join(b.get("text", "") for b in _blocks(blocks) if b.get("type") == "text")


def _image_url(source: Mapping) -> str:
    if source.get("type") == "url":
        return source["url"]
    return f"data:{source['media_type']};base64,{source['data']}"


def anthropic_to_openai(body: Mapping, model: str, *, max_tokens_param: str = "max_tokens") -> dict:
    msgs: list[dict] = []
    if body.get("system"):
        msgs.append({"role": "system", "content": _text(body["system"])})
    for m in body.get("messages", []):
        blocks = _blocks(m.get("content"))
        if m["role"] == "system":  # mid-conversation operator message
            msgs.append({"role": "system", "content": _text(blocks)})
            continue
        if m["role"] == "assistant":
            out: dict = {"role": "assistant", "content": _text(blocks) or None}
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                     for b in blocks if b.get("type") == "tool_use"]
            if calls:
                out["tool_calls"] = calls
            if out["content"] is not None or calls:
                msgs.append(out)
            continue
        parts: list[dict] = []
        for b in blocks:
            kind = b.get("type")
            if kind == "tool_result":
                text = _text(b.get("content"))
                msgs.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                             "content": f"Error: {text}" if b.get("is_error") else text})
            elif kind == "text":
                parts.append({"type": "text", "text": b["text"]})
            elif kind == "image":
                parts.append({"type": "image_url", "image_url": {"url": _image_url(b["source"])}})
        if parts:
            only_text = all(p["type"] == "text" for p in parts)
            msgs.append({"role": "user",
                         "content": "".join(p["text"] for p in parts) if only_text else parts})

    req: dict = {"model": model, "messages": msgs, "stream": bool(body.get("stream"))}
    if req["stream"]:
        req["stream_options"] = {"include_usage": True}
    if body.get("max_tokens"):
        req[max_tokens_param] = body["max_tokens"]
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("stop_sequences", "stop")):
        if body.get(src) is not None:
            req[dst] = body[src]
    if body.get("tools"):
        req["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
            for t in body["tools"]]
        choice = body.get("tool_choice") or {}
        kind = choice.get("type")
        if kind == "any":
            req["tool_choice"] = "required"
        elif kind == "tool":
            req["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif kind == "none":
            req["tool_choice"] = "none"
        if choice.get("disable_parallel_tool_use"):
            req["parallel_tool_calls"] = False
    return req


def openai_usage(usage: Optional[Mapping]) -> dict:
    """OpenAI usage in Anthropic's shape: ``input_tokens`` is the *uncached*
    remainder, same convention as Anthropic's own usage."""
    usage = usage or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {"input_tokens": max((usage.get("prompt_tokens") or 0) - cached, 0),
            "output_tokens": usage.get("completion_tokens") or 0,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0}


def _tool_input(arguments: Optional[str]) -> dict:
    try:
        parsed = json.loads(arguments or "{}")
    except ValueError:
        return {"_raw_arguments": arguments}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def openai_to_anthropic(resp: Mapping, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        content.append({"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": tc["function"]["name"],
                        "input": _tool_input(tc["function"].get("arguments"))})
    stop = _STOP.get(choice.get("finish_reason"), "end_turn")
    if stop == "end_turn" and any(b["type"] == "tool_use" for b in content):
        stop = "tool_use"  # some servers (Ollama) say "stop" alongside tool calls
    return {"id": _msg_id(), "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": openai_usage(resp.get("usage"))}


def sse(event: str, data: Mapping) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class OpenAIStreamTranslator:
    """OpenAI ``chat.completion.chunk`` stream in, Anthropic SSE events out.

    Text becomes a ``text`` block; each streamed tool call becomes a
    ``tool_use`` block whose arguments arrive as ``input_json_delta``
    fragments -- the same event shapes Anthropic's own API sends, so the
    client's tool loop runs unchanged."""

    def __init__(self, model: str):
        self.model = model
        self.usage: dict = {}
        self._started = False
        self._open: Optional[tuple[str, object]] = None  # ("text", None) | ("tool", openai index)
        self._index = -1
        self._finish: Optional[str] = None
        self._saw_tool = False

    def _start(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        return [sse("message_start", {"type": "message_start", "message": {
            "id": _msg_id(), "type": "message", "role": "assistant", "model": self.model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}})]

    def _close(self) -> list[bytes]:
        if self._open is None:
            return []
        self._open = None
        return [sse("content_block_stop", {"type": "content_block_stop", "index": self._index})]

    def _open_block(self, kind: str, key, block: dict) -> list[bytes]:
        out = self._close()
        self._index += 1
        self._open = (kind, key)
        out.append(sse("content_block_start", {"type": "content_block_start",
                                               "index": self._index, "content_block": block}))
        return out

    def feed(self, chunk: Mapping) -> list[bytes]:
        out = self._start()
        if chunk.get("usage"):
            self.usage = openai_usage(chunk["usage"])
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if self._open is None or self._open[0] != "text":
                    out += self._open_block("text", None, {"type": "text", "text": ""})
                out.append(sse("content_block_delta", {
                    "type": "content_block_delta", "index": self._index,
                    "delta": {"type": "text_delta", "text": delta["content"]}}))
            for tc in delta.get("tool_calls") or []:
                key = tc.get("index", 0)
                fn = tc.get("function") or {}
                if self._open != ("tool", key):
                    self._saw_tool = True
                    out += self._open_block("tool", key, {
                        "type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": fn.get("name", ""), "input": {}})
                if fn.get("arguments"):
                    out.append(sse("content_block_delta", {
                        "type": "content_block_delta", "index": self._index,
                        "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}}))
            if choice.get("finish_reason"):
                self._finish = choice["finish_reason"]
        return out

    def finish(self) -> list[bytes]:
        out = self._start() + self._close()
        stop = _STOP.get(self._finish, "end_turn")
        if stop == "end_turn" and self._saw_tool:
            stop = "tool_use"
        out.append(sse("message_delta", {
            "type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": self.usage.get("output_tokens", 0),
                      "input_tokens": self.usage.get("input_tokens", 0)}}))
        out.append(sse("message_stop", {"type": "message_stop"}))
        return out


def iter_openai_sse(lines: Iterable[str]) -> Iterable[dict]:
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue
