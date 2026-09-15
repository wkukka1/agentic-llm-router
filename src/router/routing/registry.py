"""A tiny name -> :class:`~router.routing.base.Router` subclass registry.

The extension point for new strategies: decorate a subclass with :func:`register`
and it becomes reachable by its ``kind`` string via :func:`build_router` /
:data:`REGISTRY` (config-driven construction, CLI ``--router`` flags, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Router

__all__ = ["REGISTRY", "register", "build_router"]

REGISTRY: dict[str, type["Router"]] = {}


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def register(cls: type["Router"]) -> type["Router"]:
    """Class decorator: index ``cls`` under its ``kind``. Returns ``cls`` unchanged."""
    kind = getattr(cls, "kind", None)
    if not kind or kind == "router":
        raise ValueError(f"{cls.__name__} must set a distinct `kind` before @register")
    if kind in REGISTRY and REGISTRY[kind] is not cls:
        # a module reload makes a new class object with the same qualified name
        # -- that's a re-registration, not a collision
        if _qualname(REGISTRY[kind]) != _qualname(cls):
            raise ValueError(
                f"router kind {kind!r} already registered to {_qualname(REGISTRY[kind])}"
            )
    REGISTRY[kind] = cls
    return cls


def build_router(kind: str, /, **kwargs) -> "Router":
    """Construct a registered router by ``kind``.

    Passes ``**kwargs`` straight through to the class. If ``run=`` is given and
    the class exposes a ``from_run`` classmethod (:class:`NIRTRouter`), that path
    is used instead of ``__init__``.
    """
    try:
        cls = REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"unknown router kind {kind!r}; registered: {sorted(REGISTRY)}"
        ) from None
    if "run" in kwargs and hasattr(cls, "from_run"):
        run = kwargs.pop("run")
        return cls.from_run(run, **kwargs)
    return cls(**kwargs)
