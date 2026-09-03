"""Canonical model identity and confidence-based alias resolution.

Backed by the human-reviewed ``configs/model_registry.yaml``. Only aliases
marked ``confidence: high`` are merged to a ``canonical_id``; ``medium`` / ``low``
aliases are recorded (queryable via :func:`alias_confidence`) but pass through
as their own slug so distinct checkpoints are never conflated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ..config import REPO_ROOT

DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "model_registry.yaml"


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    model_name: str
    provider: str
    family: str | None = None
    version: str | None = None


@dataclass
class _Registry:
    path: Path
    canonical: dict[str, ModelInfo] = field(default_factory=dict)
    merge_aliases: dict[str, str] = field(default_factory=dict)      # native.lower() -> canonical_id
    alias_conf: dict[str, str] = field(default_factory=dict)         # native.lower() -> confidence


def _slug(native: str) -> str:
    return str(native).strip().lower().replace("/", "-").replace(" ", "-")


@lru_cache(maxsize=4)
def _load(path_str: str) -> _Registry:
    path = Path(path_str)
    reg = _Registry(path=path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    merge_at = data.get("default_confidence_to_merge", "high")
    order = {"high": 3, "medium": 2, "low": 1}
    threshold = order.get(merge_at, 3)

    for entry in data["models"]:
        cid = entry["canonical_id"]
        name = entry.get("name") or _pretty_name(entry)
        reg.canonical[cid] = ModelInfo(
            model_id=cid,
            model_name=name,
            provider=entry.get("provider") or "unknown",
            family=entry.get("family"),
            version=entry.get("version"),
        )
        for alias in entry.get("aliases", []):
            native = str(alias["native"])
            conf = alias.get("confidence", "high")
            key = native.lower()
            reg.alias_conf[key] = conf
            if order.get(conf, 0) >= threshold:
                reg.merge_aliases[key] = cid
    return reg


def _pretty_name(entry: dict) -> str:
    fam = entry.get("family")
    ver = entry.get("version")
    if fam and ver:
        return f"{fam} {ver}".strip()
    return entry["canonical_id"]


def _registry(path: str | Path | None = None) -> _Registry:
    return _load(str(path or DEFAULT_REGISTRY_PATH))


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def canonical_model_id(native: str, registry_path: str | Path | None = None) -> str:
    """Resolve a source-native model string to a canonical ``model_id``.

    High-confidence aliases are merged; everything else is slugified and passed
    through (ingestion never drops data). Un-registered / non-high-confidence
    models are flagged by the quality checks.
    """
    reg = _registry(registry_path)
    key = str(native).strip().lower()
    if key in reg.merge_aliases:
        return reg.merge_aliases[key]
    return _slug(native)


def alias_confidence(native: str, registry_path: str | Path | None = None) -> str | None:
    return _registry(registry_path).alias_conf.get(str(native).strip().lower())


def model_info(model_id: str, registry_path: str | Path | None = None) -> ModelInfo:
    reg = _registry(registry_path)
    if model_id in reg.canonical:
        return reg.canonical[model_id]
    return ModelInfo(model_id, model_id, "unknown")


def is_canonical(model_id: str, registry_path: str | Path | None = None) -> bool:
    return model_id in _registry(registry_path).canonical


def all_canonical_ids(registry_path: str | Path | None = None) -> list[str]:
    return sorted(_registry(registry_path).canonical)
