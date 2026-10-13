"""Gateway settings: the ``interceptor:`` section of a gateway config (YAML)
merged over the defaults below -- which models Claude Code's requests may be
routed to, how each is reached, its prices, and the router's knobs. See
``configs/gateway.example.yaml``. No secrets live in the YAML: an ``openai``
lane model names the environment variable its key is read from.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

#: what the interceptor may route Claude Code's requests to when no config
#: says otherwise. ``proxy`` is the router's pool model whose predicted quality
#: stands in for this one -- the defaults are LLMRouterBench's 2025 pool
#: (``configs/llmrouterbench.yaml``): no public dataset scores today's Claude
#: tiers on shared prompts, so Opus reads GPT-5 (the pool's strongest), Sonnet
#: reads Claude Sonnet 4 (the same family, one generation back) and Haiku reads
#: Gemini 2.5 Flash (the pool's fast tier).
DEFAULT_MODELS: dict[str, dict] = {
    "claude-opus-5-5": {"lane": "anthropic", "proxy": "gpt-5", "context_window": 1_000_000,
                        "rejects": ["thinking_disabled"]},
    "claude-opus-5": {"lane": "anthropic", "proxy": "gpt-5", "context_window": 1_000_000},
    "claude-sonnet-5-5": {"lane": "anthropic", "proxy": "claude-sonnet-4", "context_window": 1_000_000,
                          "rejects": ["thinking_disabled"]},
    "claude-sonnet-5": {"lane": "anthropic", "proxy": "claude-sonnet-4", "context_window": 1_000_000},
    "claude-haiku-5-5": {"lane": "anthropic", "proxy": "gemini-2.5-flash", "context_window": 1_000_000},
    "claude-haiku-4-5": {"lane": "anthropic", "proxy": "gemini-2.5-flash", "context_window": 200_000,
                         "rejects": ["effort"]},
}
DEFAULT_ROUTER = {"enabled": False, "kind": "knn", "data_config": "configs/llmrouterbench.yaml", "k": 25,
                  "run": "lrb-nirt-2d-projected", "runs_dir": None, "margin": 0.05,
                  "expected_output_tokens": 800, "cache_ttl": "1h", "horizon_requests": 20,
                  "switch_margin": 0.05, "turn_smoothing": 0.5}
#: Claude Code subagents (``router.gateway.subagents``): ``roles`` maps a role
#: to a model (``null`` keeps the session's), merged over that module's
#: defaults; ``enabled: false`` routes subagents like any other turn.
DEFAULT_SUBAGENTS = {"enabled": True, "roles": {}}


def gateway_home() -> Path:
    """Where the gateway keeps its files: the request log's folder
    (``AAR_GATEWAY_LOG``), else ``~/.aar-gateway``."""
    log = os.environ.get("AAR_GATEWAY_LOG")
    return Path(log).expanduser().parent if log else Path.home() / ".aar-gateway"


def config_path() -> Path | None:
    """The gateway config in use: ``AAR_GATEWAY_CONFIG``, else
    ``~/.aar-gateway/config.yaml`` if it exists, else none (the defaults)."""
    named = os.environ.get("AAR_GATEWAY_CONFIG")
    if named:
        return Path(named).expanduser()
    default = gateway_home() / "config.yaml"
    return default if default.exists() else None


def local_models_path() -> Path:
    return gateway_home() / "local-models.yaml"


def local_models() -> dict[str, dict]:
    """Models added with ``aar-gateway local add`` -- ``{}`` when none."""
    path = local_models_path()
    if not path.exists():
        return {}
    return {str(k): dict(v) for k, v in ((yaml.safe_load(path.read_text()) or {}).get("models") or {}).items()}


def load_settings(config_path: str | Path | None = None) -> dict:
    """The interceptor's settings: the ``interceptor:`` section of a gateway
    config merged over the defaults above. Fails at load time, not on a
    request, for a bad lane, an ``openai`` lane without ``base_url``, or an
    ``api_key_env`` that isn't set."""
    from .pricing import CLAUDE_PRICES

    spec = (yaml.safe_load(Path(config_path).read_text()) or {}) if config_path else {}
    section = spec.get("interceptor") or {}
    models = {k: dict(v) for k, v in (section.get("models") or DEFAULT_MODELS).items()}
    # opt-in local models (`aar-gateway local add`), kept in a file of their own
    # so the main config is never rewritten; they never replace a configured model
    for model_id, m in local_models().items():
        models.setdefault(model_id, dict(m))
    for model_id, m in models.items():
        lane = m.setdefault("lane", "anthropic")
        if lane not in ("anthropic", "openai"):
            raise ValueError(f"gateway config: model {model_id!r} has unknown lane {lane!r}")
        if lane == "openai" and not m.get("base_url"):
            raise ValueError(f"gateway config: openai-lane model {model_id!r} needs base_url")
        key_env = m.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            raise ValueError(f"gateway config: model {model_id!r} names api_key_env={key_env!r} "
                             "but it is not set in the environment")
        if "prices" not in m and model_id in CLAUDE_PRICES:
            m["prices"] = CLAUDE_PRICES[model_id]
    # The "always the strongest" baseline: the config's choice, else the first
    # priced Claude model listed (the defaults list Opus 5.5 first). Not the
    # priciest: Opus 5 costs more than the newer, stronger Opus 5.5, and
    # measuring against it overstated the saving.
    anthropic_priced = [k for k, m in models.items() if m["lane"] == "anthropic" and m.get("prices")]
    strongest = section.get("strongest") or next(iter(anthropic_priced), None)
    return {"models": models, "strongest": strongest,
            "capture_prompts": bool(section.get("capture_prompts", False)),
            "router": {**DEFAULT_ROUTER, **(section.get("router") or {})},
            "subagents": {**DEFAULT_SUBAGENTS, **(section.get("subagents") or {})}}
