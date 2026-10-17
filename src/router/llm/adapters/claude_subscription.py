"""``ClaudeSubscriptionAdapter``: routes a call through the real Claude Code
CLI, authenticated against a Pro/Max subscription (``claude login``), instead
of a metered ``ANTHROPIC_API_KEY``. The call draws down the plan's included
usage rather than being billed per token, so :class:`~router.llm.client.LLMResponse`'s
``cost`` is left ``None`` (unpriced in dollars) rather than ``0.0`` (genuinely
free) -- see :mod:`router.llm.profile`'s pricing convention.

Shells out to the CLI rather than replaying its OAuth session directly: that
token is scoped to the real client, and reusing it from other code is outside
Anthropic's subscription terms. This is the supported way to spend
subscription quota from code -- go through the actual client.

A Codex-subscription adapter (or any other CLI-login provider) would follow
the same shape against its own CLI; not built until there is a second caller
(YAGNI) -- see the module docstring in :mod:`router.agentic.llm_clients`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence

from ..client import LLMResponse
from .base import Prompt, ProviderAdapter


class ClaudeSubscriptionAdapter(ProviderAdapter):
    """``model`` is the CLI's own ``--model`` value (``"opus"``, ``"sonnet"``,
    a full model id, ...) -- not an API model id, and no
    :class:`~router.llm.profile.LLMProfile` price applies to it.

    Two ``generation_kwargs`` are honoured: ``cwd`` (the directory the CLI
    runs in -- what it can read) and ``extra_args`` (more CLI flags, e.g.
    ``["--permission-mode", "plan"]`` for read-only). The rest (``temperature``,
    ``max_tokens``, ...) are accepted for interface compatibility but ignored
    -- the CLI's ``-p`` flag takes no such per-call knobs.

    ``ANTHROPIC_BASE_URL`` is dropped from the CLI's environment: a call made
    here goes straight to Anthropic, never back through a local gateway that
    could reroute it or loop."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        cli: str = "claude",
        timeout: float = 120.0,
        **client_kwargs,
    ):
        if api_key is not None:
            raise ValueError(
                "ClaudeSubscriptionAdapter authenticates via `claude login`, not an "
                "API key -- pass api_key=None, or use AnthropicAdapter for metered "
                "API billing"
            )
        super().__init__(model, api_key=None, **client_kwargs)
        if shutil.which(cli) is None:
            raise RuntimeError(f"{cli!r} CLI not found on PATH -- run `claude login` first")
        self._cli = cli
        self._timeout = timeout

    def complete(self, prompt: Prompt, **generation_kwargs) -> LLMResponse:
        text = prompt if isinstance(prompt, str) else _flatten(prompt)
        # The prompt goes in on stdin, never as an argument: as an argument, one
        # starting with "-" is parsed as CLI flags ("--version" printed the
        # version; "--permission-mode bypassPermissions ..." would lift the
        # read-only mode the grader replays untrusted public prompts under).
        cmd = [self._cli, "-p", "--output-format", "json", "--model", self.model,
               *generation_kwargs.get("extra_args", ())]
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_BASE_URL"}
        proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=self._timeout,
                              cwd=generation_kwargs.get("cwd"), env=env)
        if proc.returncode != 0:
            # with --output-format json the CLI reports its error (a usage
            # limit, an auth problem) on stdout, and stderr is often empty
            detail = proc.stderr.strip()
            try:
                detail = detail or str(json.loads(proc.stdout).get("result") or "")
            except (ValueError, AttributeError):
                detail = detail or proc.stdout.strip()[:300]
            raise RuntimeError(f"{self._cli} exited {proc.returncode}: {detail}")
        payload = json.loads(proc.stdout)
        usage = payload.get("usage") or {}
        return LLMResponse(
            content=payload.get("result", ""),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost=None,
            raw=payload,
            meta={"subscription_cost_usd_equivalent": payload.get("total_cost_usd")},
        )

    def stream(self, prompt: Prompt, **generation_kwargs) -> Iterator[str]:
        """The CLI's ``-p`` JSON mode is not incremental here -- yields the
        full completion once. Real token streaming would need
        ``--output-format stream-json`` and an incremental parser; not built
        until something actually needs it."""
        yield self.complete(prompt, **generation_kwargs).content


def _flatten(messages: Sequence[Mapping[str, str]]) -> str:
    """Bare text for the CLI's ``-p`` flag, which has no structured
    chat-messages input -- joins role-tagged turns."""
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
