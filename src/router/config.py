"""Configuration loading for Phase 0.

A single YAML file (``configs/phase0.yaml`` by default) drives every stage.
Access is via attribute-style dotted lookup on :class:`Config`, e.g.
``cfg.embedding.model_name`` or ``cfg.get("clustering.umap.n_neighbors")``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels up from this file (src/router/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "phase0.yaml"


class Config:
    """Read-only, attribute-accessible view over a nested dict."""

    def __init__(self, data: dict[str, Any], root: Path = REPO_ROOT):
        self._data = data
        self._root = root

    # -- access --------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Config(value, self._root) if isinstance(value, dict) else value

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node, self._root) if isinstance(node, dict) else node

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self._data!r})"

    # -- paths -------------------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    def path(self, key: str) -> Path:
        """Resolve a ``paths.<key>`` entry against the repo root."""
        rel = self._data["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else self._root / p

    def resolve(self, rel: str | os.PathLike) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self._root / p


def section(cfg: Any, dotted: str, default: dict | None = None) -> dict:
    """A config subtree as a plain ``dict`` (``default`` or ``{}`` if absent).

    Saves every caller from repeating the
    ``x.to_dict() if hasattr(x, "to_dict") else (x or {})`` dance. Works with a
    :class:`Config` or any object exposing a ``get(key, default)`` method.
    """
    node = cfg.get(dotted)
    if isinstance(node, Config):
        return node.to_dict()
    if isinstance(node, dict):
        return node
    return dict(default or {})


def coerce_hidden(value: Any) -> int | None:
    """Hidden-width config value -> ``int`` or ``None``.

    ``None`` / ``"none"`` / ``"None"`` / ``0`` -- an explicit *linear* head, i.e.
    ``query_hidden: null`` in the YAML -- become ``None``; anything else is cast to
    ``int``. Shared by every model ``from_config``. A *missing* key still means
    "default width": callers pass ``coerce_hidden(cfg.get("query_hidden", 64))``,
    so a missing key resolves to ``64`` while an explicit ``null`` stays a linear
    head.
    """
    if value in ("none", "None", 0, None):
        return None
    return int(value)


def coerce_auto_bool(value: Any, auto: bool) -> bool:
    """Tri-state config flag: ``"auto"`` / ``None`` -> ``auto``, else ``bool(value)``."""
    return bool(auto) if value in ("auto", None) else bool(value)


def require_choice(value: Any, allowed, *, field: str) -> Any:
    """Assert ``value`` is one of ``allowed`` (any container / dict); raise a
    uniform ``ValueError`` otherwise. Returns ``value`` so it can wrap an assign."""
    if value not in allowed:
        raise ValueError(f"{field} must be one of {tuple(allowed)}, got {value!r}")
    return value


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)
