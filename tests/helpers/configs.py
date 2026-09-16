"""Minimal ``router.config.Config`` stand-ins for tests that pass data in directly.

Real code only ever calls ``.get(dotted_key, default)`` on these paths, so a dict
wrapper is enough and keeps the tests off the on-disk config.
"""

from __future__ import annotations


class ChanceCfg:
    """``cfg`` for ``build_tables`` / ``apply_chance_correction`` / profile builders.

    Only ``cfg.get("chance_correction", ...)`` is consulted on that path.
    """

    def __init__(self, method: str = "normalized", clip: bool = True, warn: bool = False):
        self._d = {
            "chance_correction": {
                "method": method,
                "clip": clip,
                "warn_on_missing_choices": warn,
            }
        }

    def get(self, key, default=None):
        return self._d.get(key, default)


def chance_cfg(method: str = "normalized", clip: bool = True, warn: bool = False) -> ChanceCfg:
    """Convenience constructor mirroring the common keyword names."""
    return ChanceCfg(method=method, clip=clip, warn=warn)


class FamilyCfg:
    """``cfg`` for ``training.data.families`` -- serves the ``profiles`` block only."""

    def __init__(self, families: dict):
        self._f = families

    def get(self, dotted, default=None):
        return self._f if dotted == "profiles" else default
