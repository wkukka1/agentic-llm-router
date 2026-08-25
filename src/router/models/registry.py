"""Name -> classifier factory, so experiment YAML can stay declarative."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from router.models.base import DomainClassifier

_REGISTRY: dict[str, Callable[..., DomainClassifier]] = {}


def register(name: str) -> Callable[[type[DomainClassifier]], type[DomainClassifier]]:
    def decorator(cls: type[DomainClassifier]) -> type[DomainClassifier]:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build(name: str, **params: Any) -> DomainClassifier:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def available() -> list[str]:
    return sorted(_REGISTRY)
